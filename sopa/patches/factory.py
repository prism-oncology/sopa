import logging
from typing import Literal

from spatialdata import SpatialData

from ..constants import SopaAttrs, SopaKeys
from ..utils import get_spatial_element, get_spatial_image
from .patches import Patches2D
from .transcripts import OnDiskTranscriptPatches

log = logging.getLogger(__name__)


def make_image_patches(
    sdata: SpatialData,
    patch_width: int | None = 2000,
    patch_overlap: int = 50,
    image_key: str | None = None,
    roi_key: str | None = SopaKeys.ROI,
    scale: str | None = None,
    key_added: str | None = None,
):
    """Create overlapping patches on an image. This can be used for image-based segmentation methods such as Cellpose, which will run on each patch.

    Args:
        sdata: A `SpatialData` object.
        patch_width: Width of the patches, in pixels. If `None`, creates only one patch.
        patch_overlap: Number of pixels of overlap between patches.
        image_key: Optional key of the image on which the patches will be made. If not provided, it is found automatically.
        roi_key: Optional name of the shapes that need to touch the patches. Patches that do not touch any shape will be ignored during segmentation. By default, uses `"region_of_interest"` if existing. If `None`, all patches will be used.
        scale: If the image is a DataTree, the scale to be used for patching (e.g., `"scale1"`). By default, uses the highest resolution scale, i.e., `"scale0"`.
        key_added: Optional name of the patches to be saved. By default, uses `"image_patches"`.
    """
    image_key, _ = get_spatial_image(sdata, key=image_key, return_key=True)

    patches = Patches2D(
        sdata,
        image_key,
        patch_width=patch_width,
        patch_overlap=patch_overlap,
        roi_key=roi_key,
        scale=scale,
    )

    patches.add_shapes(key_added=key_added)


def make_transcript_patches(
    sdata: SpatialData,
    patch_width: float | int | None = 2000,
    patch_overlap: int = 50,
    points_key: str | None = None,
    prior_shapes_key: Literal["auto"] | str | None = None,
    unassigned_value: int | str | None = None,
    min_points_per_patch: int = 4000,
    write_cells_centroids: bool = False,
    roi_key: str | None = SopaKeys.ROI,
    key_added: str | None = None,
    **kwargs: int,
):
    """Create overlapping patches on a transcripts dataframe, and save it in a cache. This can be used for trancript-based segmentation methods such as Baysor or Proseg.

    !!! info "Prior segmentation usage"
        To save assign a prior segmentation to each transcript, you can use the `prior_shapes_key` argument:

        - If a segmentation has already been performed (for example an existing 10X-Genomics segmentation), use `prior_shapes_key="auto"` to use it (or, provide manually the column name and the `unassigned_value` argument).
        - If you have already run segmentation with Sopa, use `prior_shapes_key` to denote the name of the shapes (GeoDataFrame) containing the boundaries, e.g. `prior_shapes_key="cellpose_boundaries"`.

    Args:
        sdata: A `SpatialData` object.
        patch_width: Width of the patches, in microns. If `None`, creates only one patch.
        patch_overlap: Number of microns of overlap between patches.
        points_key: Optional key of the points on which the patches will be made. If not provided, it is found automatically.
        prior_shapes_key: Optional key of `sdata` containing the shapes with the prior segmentation, or column of the points dataframe. If `"auto"`, use the prior column from the technology.
        unassigned_value: If `prior_shapes_key` has been provided and corresponds to a points column: this argument is the value given to the transcript that are not inside any cell.
        min_points_per_patch: Minimum number of points/transcripts for a patch to be considered for segmentation.
        write_cells_centroids: If `True`, the centroids of the prior cells will be saved. This is useful for some segmentation tools such as ComSeg.
        roi_key: Optional name of the shapes that need to touch the patches. Patches that do not touch any shape will be ignored during segmentation. By default, uses `"region_of_interest"` if existing. If `None`, all patches will be used.
        key_added: Optional name of the patches to be saved. By default, uses `"transcripts_patches"`.
        **kwargs: Additional arguments passed to the `OnDiskTranscriptPatches` class.
    """
    assert not write_cells_centroids or prior_shapes_key, "write_cells_centroids argument requires prior_shapes_key"

    points_key, _ = get_spatial_element(
        sdata.points,
        key=points_key or sdata.attrs.get(SopaAttrs.TRANSCRIPTS),
        return_key=True,
    )

    if prior_shapes_key == "auto":
        assert SopaAttrs.PRIOR_TUPLE_KEY in sdata.attrs, (
            f"prior_shapes_key='auto' requires a prior segmentation to be present in the points dataframe ('{SopaAttrs.PRIOR_TUPLE_KEY}' must be in `sdata.attrs`)."
        )
        prior_shapes_key, unassigned_value = sdata.attrs[SopaAttrs.PRIOR_TUPLE_KEY]

    patches = OnDiskTranscriptPatches(
        sdata,
        points_key,
        patch_width=patch_width,
        patch_overlap=patch_overlap,
        prior_shapes_key=prior_shapes_key,
        unassigned_value=unassigned_value,
        min_points_per_patch=min_points_per_patch,
        write_cells_centroids=write_cells_centroids,
        roi_key=roi_key,
        **kwargs,
    )

    patches.write()
    patches.add_shapes(key_added=key_added)
