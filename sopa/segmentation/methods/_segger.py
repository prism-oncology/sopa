import logging
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from spatialdata import SpatialData
from spatialdata.models import ShapesModel

from ...aggregation.table import add_parsed_table
from ...constants import SopaKeys
from ...utils import (
    copy_transformations,
    delete_transcripts_patches_dirs,
    get_feature_key,
    get_transcripts_patches_dirs,
    set_boundaries_attrs,
    to_intrinsic,
)
from .utils import _check_transcript_patches

log = logging.getLogger(__name__)


def segger(
    sdata: SpatialData,
    delete_cache: bool = True,
    prior_shapes_key: str | None = None,
    key_added: str = SopaKeys.SEGGER_BOUNDARIES,
    **kwargs,
):
    """Run [segger](https://github.com/dpeerlab/segger) segmentation on a SpatialData object, and add the corresponding cell boundaries and `AnnData` table with counts.

    !!! warning "Segger installation"
        Make sure to install segger (`pip install 'sopa[segger]'` or `pip install segger`).
        Segger requires CUDA and a compatible GPU.

    !!! info "Segger usage"
        Segger runs on the full dataset (a single transcript patch). Run
        [`sopa.make_transcript_patches`](../patches/#sopa.make_transcript_patches) with `patch_width=None`
        and a `prior_shapes_key` before running segger.

    Args:
        sdata: A `SpatialData` object.
        delete_cache: Whether to delete the cache after segmentation.
        prior_shapes_key: Key of `sdata` containing the prior cell boundaries (inferred from transcript patches by default).
        key_added: Name of the shapes element to be added to `sdata.shapes`.
        **kwargs: Additional keyword arguments passed to `segger.segment()` (e.g. `n_epochs`).
    """
    _check_transcript_patches(sdata, with_prior=True)

    patches_dirs = get_transcripts_patches_dirs(sdata)
    assert len(patches_dirs) == 1, (
        "Segger works on a single patch. Re-run `sopa.make_transcript_patches` with `patch_width=None` and a `prior_shapes_key`."
    )
    patch_dir = Path(patches_dirs[0])

    points_key = sdata[SopaKeys.TRANSCRIPTS_PATCHES][SopaKeys.POINTS_KEY].iloc[0]
    if prior_shapes_key is None:
        prior_shapes_key = sdata[SopaKeys.TRANSCRIPTS_PATCHES][SopaKeys.PRIOR_SHAPES_KEY].iloc[0]

    transcripts, boundaries = _prepare_segger_inputs(sdata, patch_dir, points_key, prior_shapes_key)

    with tempfile.TemporaryDirectory() as output_dir:
        _run_segger(transcripts, boundaries, output_dir, **kwargs)
        adata, geo_df = _read_segger_outputs(output_dir, copy_transformations(sdata[points_key]))

    add_parsed_table(sdata, adata, geo_df, key_added, SopaKeys.TABLE)

    set_boundaries_attrs(sdata, key_added)

    if delete_cache:
        delete_transcripts_patches_dirs(sdata)

    log.info(f"Segger segmentation completed: {len(geo_df)} cells added to sdata['{key_added}'].")


def _prepare_segger_inputs(
    sdata: SpatialData,
    patch_dir: Path,
    points_key: str,
    prior_shapes_key: str,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    transcripts = pd.read_csv(patch_dir / "transcripts.csv")

    feature_key = get_feature_key(sdata[points_key], raise_error=True)

    rename_map = {}
    if feature_key != "feature_name":
        rename_map[feature_key] = "feature_name"
    if SopaKeys.SOPA_PRIOR in transcripts.columns:
        rename_map[SopaKeys.SOPA_PRIOR] = "cell_id"

    if rename_map:
        transcripts = transcripts.rename(columns=rename_map)

    assert "cell_id" in transcripts.columns, (
        "Transcripts must have a cell_id column (from prior segmentation via `sopa.make_transcript_patches`)."
    )

    transcripts["cell_id"] = transcripts["cell_id"].astype(str)

    prior_shapes = to_intrinsic(sdata, prior_shapes_key, points_key)

    has_nucleus = "overlaps_nucleus" in transcripts.columns or "CellComp" in transcripts.columns

    boundaries_records = []
    for i, (_, row) in enumerate(prior_shapes.iterrows()):
        cell_id = str(i + 1)
        boundaries_records.append({
            "cell_id": cell_id,
            "boundary_type": "cell",
            "geometry": row.geometry,
        })

    boundaries = gpd.GeoDataFrame(boundaries_records)

    if not has_nucleus:
        log.warning(
            "No nucleus information found in transcripts. "
            "Segger will use cell-level segmentation only (segmentation_graph_mode='cell')."
        )

    return transcripts, boundaries


def _run_segger(
    transcripts: pd.DataFrame,
    boundaries: gpd.GeoDataFrame,
    output_directory: str,
    **kwargs,
):
    try:
        import segger
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Install segger (`pip install 'sopa[segger]'` or `pip install segger`) for this method to work."
        )

    segger.configure_memory()

    segger.segment(
        output_directory=output_directory,
        transcripts=transcripts,
        boundaries=boundaries,
        **kwargs,
    )


def _read_segger_outputs(
    output_directory: str,
    transformations: dict,
) -> tuple:
    from pathlib import Path

    import anndata

    output_dir = Path(output_directory)

    adata_path = output_dir / "anndata.h5ad"
    assert adata_path.exists(), f"Segger output not found at {adata_path}"

    boundaries_path = output_dir / "cell_boundaries.parquet"
    assert boundaries_path.exists(), f"Segger boundaries not found at {boundaries_path}"

    adata = anndata.read_h5ad(adata_path)
    geo_df = gpd.read_parquet(boundaries_path)

    geo_df.crs = None
    geo_df = ShapesModel.parse(geo_df, transformations=transformations)

    return adata, geo_df
