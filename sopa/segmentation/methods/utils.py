import json
import logging
import shutil
from pathlib import Path

import anndata
import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from shapely.geometry import Polygon, shape
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel
from tqdm import tqdm

from ... import shapes
from ...aggregation import count_transcripts
from ...constants import SopaKeys
from ...utils import add_spatial_element, copy_transformations, get_feature_key, get_transcripts_patches_dirs
from .. import solve_conflicts

log = logging.getLogger(__name__)


def resolve(
    sdata: SpatialData,
    patches_dirs: list[str | Path],
    gene_column: str | None = None,
    min_area: float = 0,
    key_added: str = SopaKeys.BAYSOR_BOUNDARIES,
):
    """Concatenate all the per-patch segmentation runs and resolve the conflicts. Resulting cells boundaries are saved in the `SpatialData` object.

    Args:
        sdata: A `SpatialData` object
        gene_column: Column of the transcript dataframe containing the genes names
        patches_dirs: Optional list of subdirectories inside `temp_dir` to be read. By default, read all.
        min_area: Minimum area (in microns^2) for a cell to be kept
        key_added: Name of the spatial element that will be added, containing the cell boundaries
    """
    if min_area > 0:
        log.info(f"Cells whose area is less than {min_area} microns^2 will be removed")

    patches_cells, adatas = _read_all_segmented_patches(patches_dirs, min_area)
    geo_df, cells_indices, new_ids = _resolve_patches(patches_cells, adatas)

    points_key = sdata[SopaKeys.TRANSCRIPTS_PATCHES][SopaKeys.POINTS_KEY].iloc[0]
    points = sdata[points_key]
    transformations = copy_transformations(points)

    geo_df = ShapesModel.parse(geo_df, transformations=transformations)

    table_conflicts = []
    if len(new_ids):
        new_cells = geo_df.geometry[cells_indices == -1]
        geo_df_new = gpd.GeoDataFrame({"geometry": new_cells})
        geo_df_new = ShapesModel.parse(geo_df_new, transformations=transformations)

        log.info("Aggregating transcripts on merged cells")
        gene_column = gene_column or get_feature_key(sdata, raise_error=True)
        table_conflicts = count_transcripts(sdata, gene_column, geo_df=geo_df_new, points_key=points_key)
        table_conflicts.obs_names = new_ids
        table_conflicts = [table_conflicts]

    valid_ids = set(geo_df.index)
    table = anndata.concat(
        [adata[list(valid_ids & set(adata.obs_names))] for adata in adatas] + table_conflicts,
        join="outer",
    )
    table.obs.dropna(axis="columns", inplace=True)

    geo_df = geo_df.loc[table.obs_names]

    table.obsm["spatial"] = np.array([[centroid.x, centroid.y] for centroid in geo_df.centroid])
    table.obs[SopaKeys.REGION_KEY] = pd.Series(key_added, index=table.obs_names, dtype="category")
    table.obs[SopaKeys.INSTANCE_KEY] = geo_df.index

    table = TableModel.parse(
        table,
        region_key=SopaKeys.REGION_KEY,
        region=key_added,
        instance_key=SopaKeys.INSTANCE_KEY,
    )

    add_spatial_element(sdata, key_added, geo_df)
    add_spatial_element(sdata, SopaKeys.TABLE, table)

    log.info(f"Added sdata.tables['{SopaKeys.TABLE}'], and {len(geo_df)} cell boundaries to sdata['{key_added}']")


def _read_one_segmented_patch(
    directory: str, min_area: float = 0, min_vertices: int = 4
) -> tuple[list[Polygon], AnnData]:
    directory: Path = Path(directory)
    id_as_string, polygon_file = _find_polygon_file(directory)

    loom_file = directory / "segmentation_counts.loom"
    if loom_file.exists():
        adata = anndata.io.read_loom(directory / "segmentation_counts.loom", obs_names="Name", var_names="Name")
    else:
        adata = anndata.io.read_h5ad(directory / "segmentation_counts.h5ad")

    adata.obs.rename(columns={"area": SopaKeys.ORIGINAL_AREA_OBS}, inplace=True)

    cells_ids = pd.Series(
        adata.obs_names if id_as_string else adata.obs["CellID"].astype(int),
        index=adata.obs_names,
    )
    del adata.obs["CellID"]

    with open(polygon_file) as f:
        polygons_dict = json.load(f)
        polygons_dict = {c["cell"]: c for c in polygons_dict["geometries"]}

    def _keep_cell(ID: str | int):
        if ID not in polygons_dict:
            return False
        return len(polygons_dict[ID]["coordinates"][0]) >= min_vertices

    cells_ids = cells_ids[cells_ids.map(_keep_cell)]

    geo_df = gpd.GeoDataFrame(index=cells_ids.index, geometry=[shape(polygons_dict[ID]) for ID in cells_ids])
    geo_df = shapes.to_valid_polygons(geo_df)

    ratio_filtered = (geo_df.area <= min_area).mean()
    if ratio_filtered > 0.2:
        log.warning(f"{ratio_filtered:.2%} of cells will be filtered due to {min_area=}")

    geo_df = geo_df[geo_df.area > min_area]

    return geo_df.geometry.values, adata[geo_df.index].copy()


def _find_polygon_file(directory: Path) -> tuple[bool, Path]:
    old_baysor_path = directory / "segmentation_polygons.json"
    if old_baysor_path.exists():
        return False, old_baysor_path
    new_baysor_path = directory / "segmentation_polygons_2d.json"
    assert new_baysor_path.exists(), f"Could not find the segmentation polygons file in {directory}"
    return True, new_baysor_path


def _read_all_segmented_patches(
    patches_dirs: list[str],
    min_area: float = 0,
) -> tuple[list[list[Polygon]], list[AnnData]]:
    outs = [
        _read_one_segmented_patch(path, min_area)
        for path in tqdm(patches_dirs, desc="Reading transcript-segmentation outputs")
    ]

    patches_cells, adatas = zip(*outs)

    return patches_cells, adatas


def _resolve_patches(
    patches_cells: list[list[Polygon]], adatas: list[AnnData]
) -> tuple[gpd.GeoDataFrame, np.ndarray, np.ndarray]:
    """Resolve the segmentation conflits on the patches overlaps.

    Args:
        patches_cells: List of polygons segmented on each patch
        adatas: List of AnnData objects corresponding to each patch

    Returns:
        The new GeoDataFrame, the new cells indices (-1 for merged cells), and the ids of the merged cells.
    """
    patch_ids = [adata.obs_names for adata in adatas]

    patch_indices = np.arange(len(patches_cells)).repeat([len(cells) for cells in patches_cells])
    cells = [cell for cells in patches_cells for cell in cells]
    segmentation_ids = np.array([cell_id for ids in patch_ids for cell_id in ids])

    cells_resolved, cells_indices = solve_conflicts(cells, patch_indices=patch_indices, return_indices=True)

    existing_ids = segmentation_ids[cells_indices[cells_indices >= 0]]
    new_ids = np.char.add("merged_cell_", np.arange((cells_indices == -1).sum()).astype(str))
    cells_resolved.index = np.concatenate([existing_ids, new_ids])

    return cells_resolved, cells_indices, new_ids


def _check_transcript_patches(sdata: SpatialData, with_prior: bool = False):
    assert SopaKeys.TRANSCRIPTS_PATCHES in sdata.shapes, (
        "Transcript patches not found in the SpatialData object. Run `sopa.make_transcript_patches(...)` first."
    )

    directories: list[Path] = get_transcripts_patches_dirs(sdata)

    assert all(directory.exists() for directory in directories), (
        "Some patch directories are missing. "
        "This can happen if you already finished a segmentation which deleted the cache. "
        "You must re-run `sopa.make_transcript_patches`. "
        "Next time, use `delete_cache=False` during the segmentation to keep the cache."
    )

    if with_prior:
        assert SopaKeys.PRIOR_SHAPES_KEY in sdata[SopaKeys.TRANSCRIPTS_PATCHES].columns, (
            "You need to run `sopa.make_transcript_patches` with a `prior_shapes_key`. "
            "You can provide `prior_shapes_key='auto'` if your technology has a prior segmentation, or `prior_shapes_key='cellpose_boundaries'` if you ran cellpose segmentation first."
        )


def _get_executable_path(name: str, default_dir: str) -> Path | str:
    if shutil.which(name) is not None:
        return name

    default_path = Path.home() / default_dir / "bin" / name
    if default_path.exists():
        return default_path

    bin_path = Path.home() / ".local" / "bin" / name
    raise FileNotFoundError(
        f"Please install {name} and ensure that either `{default_path}` executes {name}, or that `{name}` is an existing command (add it to your PATH, or create a symlink at {bin_path})."
    )
