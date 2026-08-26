import logging
from pathlib import Path

import anndata
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata
from anndata import AnnData
from scipy.sparse import csr_matrix
from spatialdata import SpatialData
from spatialdata.models import ShapesModel
from spatialdata.transformations import BaseTransformation, Identity

from ...aggregation.table import add_parsed_table
from ...constants import SopaAttrs, SopaKeys
from ...utils import (
    copy_transformations,
    delete_transcripts_patches_dirs,
    get_cache_dir,
    get_feature_key,
    get_transcripts_patches_dirs,
    set_boundaries_attrs,
    to_intrinsic,
)
from .utils import _check_transcript_patches, _get_executable_path

log = logging.getLogger(__name__)


def proseg(
    sdata: SpatialData,
    delete_cache: bool = True,
    command_line_suffix: str = "",
    infer_presets: bool = True,
    prior_shapes_key: str | None = None,
    bins_key: str | None = None,
    key_added: str = SopaKeys.PROSEG_BOUNDARIES,
):
    """Run [`proseg`](https://github.com/dcjones/proseg) segmentation on a SpatialData object, and add the corresponding cell boundaries and `AnnData` table with counts.

    !!! warning "Proseg installation"
        Make sure to install [`proseg`](https://github.com/dcjones/proseg) separately before running this function.

    !!! info "Proseg usage specificities"
        If you are using Visium HD data, provide the `prior_shapes_key` corresponding to the prior cell boundaries (e.g., `"stardist_boundaries"`).

        If you're using non-Visium HD data, note that `proseg` will only run on one patch. I.e., you need
        to run [`sopa.make_transcript_patches`](../patches/#sopa.make_transcript_patches) with `patch_width=None` and a `prior_shapes_key` before running `proseg`.

        Also, note that aggregation is not necessary after running `proseg` (but can be useful if you want e.g. channel aggregation).

    Args:
        sdata: A `SpatialData` object.
        delete_cache: Whether to delete the cache after segmentation.
        command_line_suffix: Optional suffix to add to the proseg command line.
        infer_presets: Whether to infer the proseg presets based on the columns of the transcripts dataframe.
        prior_shapes_key: **Only for Visium HD data.** Key of `sdata` containing the prior cell boundaries. If `"auto"`, use the latest performed segmentation (e.g., stardist or the 10X Genomics segmentation).
        bins_key: **Only for Visium HD data.** Key of `sdata` with the table corresponding to the bin-by-gene table of gene counts (e.g., for Visium HD data). Inferred by default.
        key_added: Name of the shapes element to be added to `sdata.shapes`.
    """
    if prior_shapes_key is not None:
        bins_key = bins_key or sdata.attrs.get(SopaAttrs.BINS_TABLE)

        assert bins_key is not None, (
            "Using `prior_shapes_key` is specific to Visium HD data, and a `bins_key` must be provided."
        )

        _proseg_bins(
            sdata,
            prior_shapes_key=prior_shapes_key,
            bins_key=bins_key,
            command_line_suffix=command_line_suffix,
            infer_presets=infer_presets,
            key_added=key_added,
        )
    else:
        assert sdata.points, (
            "No points found in `sdata`. If you use Visium HD data, consider providing a `prior_shapes_key`."
        )
        assert bins_key is None, (
            "`bins_key` can only be provided when using `prior_shapes_key` (i.e., for Visium HD data)."
        )

        _proseg_points(
            sdata,
            delete_cache=delete_cache,
            command_line_suffix=command_line_suffix,
            infer_presets=infer_presets,
            key_added=key_added,
        )

    log.info("Proseg table and boundaries added (running `sopa.aggregate` is not mandatory).")


def _proseg_points(
    sdata: SpatialData, delete_cache: bool, command_line_suffix: str, infer_presets: bool, key_added: str
):
    _check_transcript_patches(sdata, with_prior=True)

    points_key = sdata[SopaKeys.TRANSCRIPTS_PATCHES][SopaKeys.POINTS_KEY].iloc[0]

    patches_dirs = get_transcripts_patches_dirs(sdata)
    assert len(patches_dirs) == 1, (
        "Proseg is fast enough to work on a single patch. Re-run `sopa.make_transcript_patches` with `patch_width=None` and a `prior_shapes_key`."
    )
    patch_dir = Path(patches_dirs[0])

    proseg_command = _get_proseg_points_command(sdata, points_key, command_line_suffix, infer_presets)

    _run_proseg(proseg_command, patch_dir)
    adata, geo_df = _read_proseg(patch_dir, copy_transformations(sdata[points_key]))

    add_parsed_table(sdata, adata, geo_df, key_added, SopaKeys.TABLE)

    set_boundaries_attrs(sdata, key_added)

    if delete_cache:
        delete_transcripts_patches_dirs(sdata)


def _proseg_bins(
    sdata: SpatialData,
    prior_shapes_key: str,
    bins_key: str,
    command_line_suffix: str,
    infer_presets: bool,
    key_added: str,
):
    import shutil

    if prior_shapes_key == "auto":
        prior_shapes_key = sdata.attrs.get(SopaAttrs.BOUNDARIES)
        assert prior_shapes_key is not None, (
            "`sdata` does not contain any known prior segmentation. Please provide the `prior_shapes_key` explicitly."
        )

    bins_table: AnnData = sdata.tables[bins_key]

    assert isinstance(bins_table.X, csr_matrix), (
        "The bin table's X matrix must be in CSR format. E.g., you can run `adata.X = adata.X.tocsr()`"
    )

    bins_shapes_key = sdata.get_annotated_regions(bins_table)
    bins_shapes_key = bins_shapes_key[0] if isinstance(bins_shapes_key, list) else bins_shapes_key
    bins_shapes = sdata[bins_shapes_key]

    assert "microns" in bins_shapes.attrs["transform"], "`microns` coordinate system is needed for the shapes"

    centroid_microns = spatialdata.transform(bins_shapes, to_coordinate_system="microns").centroid
    bins_table.obsm["spatial_microns"] = np.array([centroid_microns.x, centroid_microns.y]).T

    prior_aligned = to_intrinsic(sdata, prior_shapes_key, bins_shapes_key)

    sjoin = sdata[bins_shapes_key].sjoin(prior_aligned.reset_index(drop=True))
    sjoin = sjoin[~sjoin.index.duplicated()]

    instance_key_column = sdata.get_instance_key_column(bins_table)
    sjoin.index = sjoin.index.map(pd.Series(instance_key_column.index, index=instance_key_column.values))

    bins_table.obs[SopaKeys.SOPA_PRIOR] = 0
    bins_table.obs.loc[sjoin.index, SopaKeys.SOPA_PRIOR] = sjoin["index_right"] + 1

    work_dir = get_cache_dir(sdata)

    with anndata.settings.override(allow_write_nullable_strings=False):  # fix #441
        bins_table.write_zarr(work_dir / "bins_table.zarr")

    proseg_command = _get_proseg_bins_command(sdata, command_line_suffix, infer_presets)

    _run_proseg(proseg_command, work_dir)

    adata, geo_df = _read_proseg(work_dir, _transformations_from_microns(bins_shapes))

    add_parsed_table(sdata, adata, geo_df, key_added, SopaKeys.TABLE)

    set_boundaries_attrs(sdata, key_added)

    shutil.rmtree(work_dir / "bins_table.zarr")
    shutil.rmtree(work_dir / "proseg-output.zarr")


def _transformations_from_microns(bins_shapes: gpd.GeoDataFrame) -> dict[str, BaseTransformation]:
    transformations = copy_transformations(bins_shapes)
    from_microns = transformations["microns"].inverse()

    transformations = {cs: from_microns.compose_with(t) for cs, t in transformations.items() if cs != "microns"}
    transformations["microns"] = Identity()

    return transformations


def _run_proseg(proseg_command: str, cwd: str | Path):
    import subprocess

    log.info(f"Running proseg with command: `{proseg_command}`")

    result = subprocess.run(
        proseg_command,
        cwd=cwd,
        shell=True,
        capture_output=False,
    )

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=proseg_command,
            output=result.stdout,
            stderr=result.stderr,
        )


def _get_proseg_points_command(
    sdata: SpatialData,
    points_key: str,
    command_line_suffix: str,
    infer_presets: bool,
) -> str:
    proseg_executable = _get_executable_path("proseg", ".cargo")

    feature_key = get_feature_key(sdata[points_key], raise_error=True)

    use_zarr = _use_zarr_output(proseg_executable)

    if infer_presets:
        command_line_suffix = _add_presets(command_line_suffix, sdata[points_key].columns)

    return f"{proseg_executable} transcripts.csv -x x -y y -z z --gene-column {feature_key} --cell-id-column {SopaKeys.SOPA_PRIOR} --cell-id-unassigned 0 {'--exclude-spatialdata-transcripts' if use_zarr else ''} {command_line_suffix}"


def _get_proseg_bins_command(sdata: SpatialData, command_line_suffix: str, infer_presets: bool) -> str:
    proseg_executable = _get_executable_path("proseg", ".cargo")

    if infer_presets and "--voxel-size" not in command_line_suffix:
        command_line_suffix += " --voxel-size 2"

    return f"{proseg_executable} --anndata {get_cache_dir(sdata) / 'bins_table.zarr'} --anndata-coordinate-key spatial_microns --overwrite --cell-id-column {SopaKeys.SOPA_PRIOR} {command_line_suffix}"


def _add_presets(command_line_suffix: str, columns: list[str]) -> str:
    if "fov-column" not in command_line_suffix:
        for column in columns:
            if column in ["fov", "fov_name"]:
                command_line_suffix += f" --fov-column {column}"
                break

    if "qv" in columns and "--qv-column" not in command_line_suffix:
        command_line_suffix += " --qv-column qv"

    if "qv" in columns and "--min-qv" not in command_line_suffix:
        command_line_suffix += " --min-qv 20.0"

    if "compartment-column" not in command_line_suffix:
        if "overlaps_nucleus" in columns:
            command_line_suffix += " --compartment-column overlaps_nucleus --compartment-nuclear 1"
        elif "CellComp" in columns:
            command_line_suffix += " --compartment-column CellComp --compartment-nuclear Nuclear"

    return command_line_suffix


def _read_proseg(cwd: Path, transformations: dict[str, BaseTransformation]) -> tuple[AnnData, gpd.GeoDataFrame]:
    zarr_output = cwd / "proseg-output.zarr"

    if zarr_output.exists():  # in versions >= 3.0.0
        _sdata = spatialdata.read_zarr(zarr_output)
        adata, geo_df = _sdata.tables["table"], _sdata.shapes["cell_boundaries"]

        del geo_df.attrs["transform"]
    else:
        import gzip

        counts = pd.read_csv(cwd / "expected-counts.csv.gz")

        obs = pd.read_csv(cwd / "cell-metadata.csv.gz")
        obs.index = obs.index.map(str)

        adata = AnnData(counts, obs=obs)

        with gzip.open(cwd / "cell-polygons.geojson.gz", "rb") as f:
            geo_df = gpd.read_file(f)

    geo_df.crs = None
    geo_df = ShapesModel.parse(geo_df, transformations=transformations)

    return adata, geo_df


def _use_zarr_output(proseg_executable_path: str) -> bool:
    import subprocess

    from packaging.version import InvalidVersion, Version

    result = subprocess.run(f"{proseg_executable_path} --version", shell=True, capture_output=True, text=True)

    try:
        return Version(result.stdout.split()[1]) >= Version("3.0.0")
    except InvalidVersion:
        log.warning(
            "Could not parse the version of proseg. Assumes proseg<3.0.0. We may assume fallback to a different version in the future."
        )
        return False
