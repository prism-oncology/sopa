import logging
import warnings
from pathlib import Path

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata
from spatialdata import SpatialData
from spatialdata.models import SpatialElement
from spatialdata.transformations import (
    Affine,
    BaseTransformation,
    Identity,
    Sequence,
    get_transformation,
    get_transformation_between_coordinate_systems,
)
from xarray import DataArray, DataTree

from .. import settings
from ..constants import ATTRS_KEY, SopaAttrs, SopaFiles, SopaKeys

log = logging.getLogger(__name__)


def set_boundaries_attrs(sdata: SpatialData, key: str):
    """Sets the boundaries attribute in the SpatialData object.

    Args:
        sdata: A SpatialData object.
        key: Key of the shapes containing the cell boundaries.
    """
    assert key in sdata.shapes, f"No '{key}' found in sdata.shapes"
    sdata.attrs[SopaAttrs.BOUNDARIES] = key
    _save_attrs(sdata)


def get_boundaries(
    sdata: SpatialData,
    return_key: bool = False,
    warn: bool = False,
    key: str | None = None,
    table_key: str | None = None,
) -> gpd.GeoDataFrame | tuple[str, gpd.GeoDataFrame] | None:
    """Gets cell segmentation boundaries of a SpatialData object after running Sopa.

    Args:
        sdata: A SpatialData object
        return_key: Whether to return the key of the shapes or not.
        warn: If `True`, prints a warning if no boundary is found. Else, raises an error.
        key: A valid `shapes_key` or None.
        table_key: Name of the table used to find the corresponding boundaries.

    Returns:
        A `GeoDataFrame` containing the boundaries, or a tuple `(shapes_key, geo_df)`
    """
    assert key is None or table_key is None, "Provide only one of `key` or `table_key`"

    if table_key is not None:
        key = sdata.tables[table_key].uns[ATTRS_KEY]["region"]
        assert isinstance(key, str)
        return get_spatial_element(sdata.shapes, key=key, return_key=return_key)

    key = key or sdata.attrs.get(SopaAttrs.BOUNDARIES)

    if key is not None:
        return get_spatial_element(sdata.shapes, key=key, return_key=return_key)

    VALID_BOUNDARIES = [
        SopaKeys.PROSEG_BOUNDARIES,
        SopaKeys.BAYSOR_BOUNDARIES,
        SopaKeys.STARDIST_BOUNDARIES,
        SopaKeys.COMSEG_BOUNDARIES,
        SopaKeys.CELLPOSE_BOUNDARIES,
    ]
    for key in VALID_BOUNDARIES:
        res = _try_get_boundaries(sdata, key, return_key)
        if res is not None:
            return res

    error_message = "sdata object has no valid segmentation boundary. Consider running Sopa segmentation first."

    if not warn:
        raise ValueError(error_message)

    log.warning(error_message)
    return (None, None) if return_key else None


def _try_get_boundaries(sdata: SpatialData, key: str, return_key: bool) -> gpd.GeoDataFrame | None:
    """Try to get a cell boundaries for a given `shapes_key`"""
    if key in sdata.shapes:
        return (key, sdata[key]) if return_key else sdata[key]


def to_intrinsic(
    sdata: SpatialData, element: SpatialElement | str, target_element: SpatialElement | str
) -> SpatialElement:
    """Transforms a `SpatialElement` into the intrinsic coordinate system of another `SpatialElement`

    Args:
        sdata: A SpatialData object
        element: `SpatialElement` to transform, or its key. We recommend it to choose a vector element (for instance, points or shapes).
        target_element: `SpatialElement` of the target coordinate system, or its key.

    Returns:
        The `element` with coordinates transformed to the intrinsic coordinate system of `target_element`.
    """
    element = sdata[element] if isinstance(element, str) else element
    target_element = sdata[target_element] if isinstance(target_element, str) else target_element

    for cs, transformation in get_transformation(element, get_all=True).items():
        if isinstance(transformation, Identity):
            target_transformations = get_transformation(target_element, get_all=True)
            if isinstance(target_transformations.get(cs), Identity):
                return element  # no transformation needed
            break

    try:
        transformation = get_transformation_between_coordinate_systems(sdata, element, target_element)
    except:
        transformations1 = get_transformation(element, get_all=True)
        transformations2 = get_transformation(target_element, get_all=True)

        common_keys = list(set(transformations1.keys()) & set(transformations2.keys()))

        if not common_keys:
            raise ValueError("No common coordinate system found between the two elements")

        cs = "global" if "global" in common_keys else common_keys.pop()

        transformation = Sequence([transformations1[cs], transformations2[cs].inverse()])

    return spatialdata.transform(element, transformation=transformation, maintain_positioning=True)


def _make_affine_2d(affine: Affine, remove_name: str = "z") -> Affine:
    X = affine.matrix
    i, j = affine.input_axes.index(remove_name), affine.output_axes.index(remove_name)

    assert X[j, i] == 1 and len(X[:, i].nonzero()[0]) == 1, "Cannot reduce 3D affine to 2D"

    return Affine(
        np.delete(np.delete(X, i, axis=1), j, axis=0),
        input_axes=[name for name in affine.input_axes if name != remove_name],
        output_axes=[name for name in affine.output_axes if name != remove_name],
    )


def _has_3d_name(affine: Affine, name: str) -> bool:
    in_input, in_output = name in affine.input_axes, name in affine.output_axes
    assert in_input == in_output, f"Affine transformation has {name=} in only one of input/output axes: {affine}"
    return in_input


def _ensure_2d_transformation(
    transformation: BaseTransformation | dict[str, BaseTransformation],
) -> BaseTransformation | dict[str, BaseTransformation]:
    if isinstance(transformation, dict):
        return {cs: _ensure_2d_transformation(t) for cs, t in transformation.items()}

    if isinstance(transformation, Affine):
        if _has_3d_name(transformation, "z"):
            return _make_affine_2d(transformation, remove_name="z")
        elif _has_3d_name(transformation, "c"):
            return _make_affine_2d(transformation, remove_name="c")
        return transformation

    elif isinstance(transformation, Sequence):
        return Sequence([_ensure_2d_transformation(t) for t in transformation.transformations])

    return transformation


def copy_transformations(element: SpatialElement, ensure_2d: bool = True) -> BaseTransformation:
    transformation = get_transformation(element, get_all=True).copy()
    return _ensure_2d_transformation(transformation) if ensure_2d else transformation


def get_feature_key(points: dd.DataFrame | SpatialData, raise_error: bool = False) -> str:
    """Get the feature key of a transcript dataframe"""
    if isinstance(points, SpatialData):
        if SopaKeys.TRANSCRIPTS_PATCHES in points.shapes:
            points_key = points[SopaKeys.TRANSCRIPTS_PATCHES][SopaKeys.POINTS_KEY].iloc[0]
        else:
            points_key = points.attrs.get(SopaAttrs.TRANSCRIPTS)
            assert points_key is not None, "No transcripts found in the SpatialData object"

        points = points[points_key]

    assert isinstance(points, dd.DataFrame), "points must be a Dask DataFrame"

    feature_key = points.attrs.get(ATTRS_KEY, {}).get("feature_key")

    if raise_error and feature_key is None:
        raise ValueError(
            f"No gene column name found in points['{ATTRS_KEY}']['feature_key']. Provide the `gene_column` argument."
        )

    return feature_key


def get_intensities(sdata: SpatialData, table_key: str = SopaKeys.TABLE) -> pd.DataFrame | None:
    """Gets the intensity dataframe of shape `n_obs x n_channels`

    Args:
        sdata: A `SpatialData` object.
        table_key: Key of `sdata` containing to table from which intensities will be extracted.

    Returns:
        A pandas DataFrame containing the intensities, or `None` if no intensities are found.
    """
    assert table_key in sdata.tables, f"No '{table_key}' found in sdata.tables"

    adata = sdata.tables[table_key]

    if not adata.uns[SopaKeys.UNS_KEY][SopaKeys.UNS_HAS_INTENSITIES]:
        return None

    if adata.uns[SopaKeys.UNS_KEY][SopaKeys.UNS_HAS_TRANSCRIPTS]:
        return adata.obsm[SopaKeys.INTENSITIES_OBSM]

    return adata.to_df()


def get_spatial_element(
    element_dict: dict[str, SpatialElement],
    key: str | None = None,
    return_key: bool = False,
    as_spatial_image: bool = False,
    scale: str | None = None,
) -> SpatialElement | tuple[str, SpatialElement]:
    """Gets an element from a SpatialData object.

    Args:
        element_dict: Dictionnary whose values are spatial elements (e.g., `sdata.images`).
        key: Optional element key. If `None`, returns the only element (if only one).
        return_key: Whether to also return the key of the element.
        as_spatial_image: Whether to return the element as a `SpatialImage` (if it is a `DataTree`)
        scale: If the element is a `DataTree`, the scale to be used.


    Returns:
        If `return_key` is False, only the element is returned, else a tuple `(element_key, element)`
    """
    assert len(element_dict), "No spatial element was found in the dict."

    if key is not None:
        assert key in element_dict, f"Spatial element '{key}' not found."
        return _return_element(element_dict, key, return_key, as_spatial_image, scale=scale)

    assert len(element_dict) > 0, (
        "No spatial element found. Provide an element key to denote which element you want to use."
    )
    assert len(element_dict) == 1, (
        f"Multiple valid elements found: {', '.join(element_dict.keys())}. Provide an element key to denote which element you want to use."
    )

    key = next(iter(element_dict.keys()))

    return _return_element(element_dict, key, return_key, as_spatial_image, scale=scale)


def get_spatial_image(
    sdata: SpatialData,
    key: str | None = None,
    return_key: bool = False,
    scale: str | None = None,
    valid_attr: str = SopaAttrs.CELL_SEGMENTATION,
) -> DataArray | tuple[str, DataArray]:
    """Gets a DataArray from a SpatialData object (if the image has multiple scale, the `scale0` is returned)

    Args:
        sdata: SpatialData object.
        key: Optional image key. If `None`, returns the only image (if only one), or tries to find an image with `valid_attr`.
        return_key: Whether to also return the key of the image.
        scale: If the image is a `DataTree`, the scale to be used. By default, uses the highest resolution scale, i.e., `"scale0"`.
        valid_attr: Attribute that the image must have to be considered valid.

    Returns:
        If `return_key` is False, only the image is returned, else a tuple `(image_key, image)`
    """
    return get_spatial_element(
        sdata.images,
        key=key or sdata.attrs.get(valid_attr),
        return_key=return_key,
        as_spatial_image=True,
        scale=scale,
    )


def _return_element(
    element_dict: dict[str, SpatialElement],
    key: str,
    return_key: bool,
    as_spatial_image: bool,
    scale: str | None = None,
) -> SpatialElement | tuple[str, SpatialElement]:
    element = element_dict[key]

    if as_spatial_image and isinstance(element, DataTree):
        if scale is not None and scale != "scale0":
            assert scale in element, f"Scale '{scale}' not found. Available scales: {', '.join(element.keys())}"
        element = next(iter(element[scale or "scale0"].values()))

    return (key, element) if return_key else element


def add_spatial_element(
    sdata: SpatialData,
    element_name: str,
    element: SpatialElement,
    overwrite: bool = True,
):
    assert isinstance(element_name, str)
    assert overwrite or element_name not in sdata._shared_keys, (
        f"Trying to add {element_name=} but it is already existing and {overwrite=}"
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*already exists. Overwriting it in-memory.")
        sdata[element_name] = element

    if sdata.is_backed() and settings.auto_save_on_disk:
        try:
            sdata.write_element(element_name, overwrite=overwrite)
        except Exception as e:
            if overwrite:  # force writing the element
                sdata.delete_element_from_disk(element_name)
                sdata.write_element(element_name, overwrite=overwrite)
            else:
                raise ValueError(f"Error while saving {element_name} on disk with {overwrite=}: {e}")


def set_sopa_attrs(
    sdata: SpatialData,
    cell_segmentation_key: str | None = None,
    tissue_segmentation_key: str | None = None,
    transcripts_key: str | None = None,
    boundaries_key: str | None = None,
    bins_table_key: str | None = None,
):
    """Stores in the `SpatialData` object the keys of the main elements used in Sopa.
    This allows Sopa to retreive with elements should be used for each operation.

    !!! info
        The attrs are already stored in `sdata.attrs` when reading data with `sopa.io`.
        Use this function only if you already stored on disk a SpatialData object without the attrs (with `sopa<2.0.0`).

    Args:
        sdata: A `SpatialData` object.
        cell_segmentation_key: Name of the image to be used for cell segmentation (highest resolution image).
        tissue_segmentation_key: Name of the image to be used for tissue segmentation (medium/low resolution image).
        transcripts_key: Name of the points containing the transcripts.
        boundaries_key: Name of the shapes containing the cell boundaries.
        bins_table_key: Name of the table containing the bins (e.g., for Visium HD data).
    """
    if cell_segmentation_key is not None:
        assert cell_segmentation_key in sdata.images
        sdata.attrs[SopaAttrs.CELL_SEGMENTATION] = cell_segmentation_key

    if tissue_segmentation_key is not None:
        assert tissue_segmentation_key in sdata.images
        sdata.attrs[SopaAttrs.TISSUE_SEGMENTATION] = tissue_segmentation_key

    if transcripts_key is not None:
        assert transcripts_key in sdata.points
        sdata.attrs[SopaAttrs.TRANSCRIPTS] = transcripts_key

    if boundaries_key is not None:
        assert boundaries_key in sdata.shapes
        sdata.attrs[SopaAttrs.BOUNDARIES] = boundaries_key

    if bins_table_key is not None:
        assert bins_table_key in sdata.tables
        sdata.attrs[SopaAttrs.BINS_TABLE] = bins_table_key

    _save_attrs(sdata)


def _save_attrs(sdata: SpatialData):
    if sdata.is_backed() and settings.auto_save_on_disk:
        try:
            sdata.write_attrs()
        except Exception as e:
            raise ValueError(f"Error while saving attrs on disk: {e}")


HOME_CACHE_DIR = Path.home() / SopaFiles.SOPA_CACHE_DIR


def get_cache_dir(sdata: SpatialData) -> Path:
    """Get the cache directory for a SpatialData object.

    Args:
        sdata: A `SpatialData` object.

    Returns:
        A `Path` to the cache directory.
    """
    if sdata.is_backed():  # inside the zarr directory
        cache_dir = sdata.path.resolve() / SopaFiles.SOPA_CACHE_DIR
    elif SopaAttrs.UID in sdata.attrs:  # existing cache in the home directory
        cache_dir = HOME_CACHE_DIR / sdata.attrs[SopaAttrs.UID]
    else:  # create a new cache directory in the home directory
        import uuid

        uid = str(uuid.uuid4())
        sdata.attrs[SopaAttrs.UID] = uid
        cache_dir = HOME_CACHE_DIR / str(uid)

    cache_dir.mkdir(exist_ok=True, parents=True)

    return cache_dir


def delete_cache(sdata: SpatialData | None = None) -> None:
    """Delete the cache directory (the entire cache, or the cache of one specific SpatialData object).

    Args:
        sdata: The SpatialData object whose cache is to be deleted. If None, the entire cache is deleted.
    """
    import shutil

    if sdata is not None:
        cache_dir = get_cache_dir(sdata)
        shutil.rmtree(cache_dir)
        return

    for sub_dir in list(HOME_CACHE_DIR.iterdir()):
        if sub_dir.is_dir():
            shutil.rmtree(sub_dir)


def get_transcripts_patches_dirs(sdata: SpatialData, index: int | None = None) -> list[Path] | Path:
    """Get the list of directories containing the transcript patches

    Args:
        sdata: A `SpatialData` object containing the transcript patches.

    Returns:
        If `index` is None, a list of `Path` to the directories containing the transcript patches. Else, the `Path` to the directory of the patch with the given index.
    """
    assert SopaKeys.TRANSCRIPTS_PATCHES in sdata.shapes, "Transcript patches not found in the SpatialData object"

    cache_dir = get_cache_dir(sdata) / SopaFiles.TRANSCRIPT_CACHE_DIR

    if index is not None:
        assert index in sdata.shapes[SopaKeys.TRANSCRIPTS_PATCHES].index, f"Patch index {index} not found"
        return cache_dir / str(index)

    return [cache_dir / str(index) for index in sdata.shapes[SopaKeys.TRANSCRIPTS_PATCHES].index]


def delete_transcripts_patches_dirs(sdata: SpatialData):
    """Delete the cache directories containing the transcript patches (for instance, for Baysor or ComSeg)

    Args:
        sdata: A `SpatialData` object.
    """
    import shutil

    for patch_dir in get_transcripts_patches_dirs(sdata):
        shutil.rmtree(patch_dir)
