import shutil

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import spatialdata
from shapely import Point
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, PointsModel, ShapesModel

import sopa
from sopa.constants import SopaFiles, SopaKeys
from sopa.patches.patches import Patches1D, Patches2D
from sopa.patches.transcripts import OnDiskTranscriptPatches, _unassigned_to_zero
from sopa.segmentation.methods.utils import _check_transcript_patches


@pytest.fixture
def sdata() -> SpatialData:
    sdata = sopa.io.toy_dataset(length=512, cell_density=1e-3, as_multiscale=True)
    return sdata


def test_patchify_image(sdata: SpatialData):
    sopa.make_image_patches(sdata, 300, 100)
    assert len(sdata[SopaKeys.PATCHES]) == 9

    sopa.make_image_patches(sdata, 512, 0)
    assert len(sdata[SopaKeys.PATCHES]) == 1

    sopa.make_image_patches(sdata, 200, 0, scale="scale0")
    assert len(sdata[SopaKeys.PATCHES]) == 9

    sopa.make_image_patches(sdata, 200, 0, scale="scale1")
    assert len(sdata[SopaKeys.PATCHES]) == 4

    sopa.make_image_patches(sdata, 200, 0, scale="scale2")
    assert len(sdata[SopaKeys.PATCHES]) == 1

    sopa.make_image_patches(sdata, 500, 0, scale="scale0")
    assert len(sdata[SopaKeys.PATCHES]) == 4

    sopa.make_image_patches(sdata, 500, 0, scale="scale1")
    assert len(sdata[SopaKeys.PATCHES]) == 1

    sopa.make_image_patches(sdata, 500, 0, scale="scale2")
    assert len(sdata[SopaKeys.PATCHES]) == 1


def test_patchify_inside_tissue_roi(sdata: SpatialData):
    sopa.make_image_patches(sdata, 80, 0)
    assert len(sdata[SopaKeys.PATCHES]) == 49

    sopa.segmentation.tissue(sdata)

    sopa.make_image_patches(sdata, 80, 0)
    assert len(sdata[SopaKeys.PATCHES]) == 42  # inside the tissue ROI

    del sdata.shapes[SopaKeys.ROI]


def test_pathify_transcripts():
    # reproduce issue #214

    xmax = np.float32(11475.797)
    xmin = np.float32(2.671875)

    Patches1D(xmin=xmin, xmax=xmax, patch_width=300.0, patch_overlap=30.0, tight=True, int_coords=False)


def test_patchify_baysor(sdata: SpatialData):
    with pytest.raises(AssertionError):
        sopa.utils.get_transcripts_patches_dirs(sdata)

    sopa.make_transcript_patches(sdata, 30, 10)
    assert len(sdata[SopaKeys.TRANSCRIPTS_PATCHES]) == 9
    assert len(sopa.utils.get_transcripts_patches_dirs(sdata)) == 9

    sopa.make_transcript_patches(sdata, 52, 0)
    assert len(sdata[SopaKeys.TRANSCRIPTS_PATCHES]) == 1
    assert len(sopa.utils.get_transcripts_patches_dirs(sdata)) == 1

    sopa.utils.delete_cache(sdata)


def test_patchify_baysor_inside_tissue_roi(sdata: SpatialData):
    sopa.make_transcript_patches(sdata, 5, 0, min_points_per_patch=0)

    assert len(sdata[SopaKeys.TRANSCRIPTS_PATCHES]) == 121

    sopa.segmentation.tissue(sdata)

    sopa.make_transcript_patches(sdata, 5, 0, min_points_per_patch=0)

    assert len(sdata[SopaKeys.TRANSCRIPTS_PATCHES]) == 107  # inside the tissue ROI

    sopa.utils.delete_cache(sdata)


def test_gene_exlude_pattern():
    _default_gene_exclude_pattern = sopa.settings.gene_exclude_pattern

    num_to_exclude = 5

    gene_names = [
        pd.NA,  # should be excluded
        np.nan,  # should be excluded
        "bLaNk",  # should be excluded
        "SystemControl",  # should be excluded
        "negcontrol_gene",  # should be excluded
        "gene1",
        "gene2",
        "gene3",
        "CDNAN",  # should not be excluded although it contains 'nan'
    ]

    df = pd.DataFrame({
        "x": np.zeros(len(gene_names)),
        "y": np.zeros(len(gene_names)),
        "z": np.zeros(len(gene_names)),
        "genes": gene_names,
    })
    points = {
        "transcripts": PointsModel.parse(df, feature_key="genes"),
    }

    sdata = SpatialData(points=points)

    sopa.make_transcript_patches(sdata, min_points_per_patch=0)

    df = pd.read_csv(
        sopa.utils.get_cache_dir(sdata) / SopaFiles.TRANSCRIPT_CACHE_DIR / "0" / SopaFiles.TRANSCRIPTS_FILE
    )

    assert len(df) == len(sdata["transcripts"]) - num_to_exclude

    sopa.settings.gene_exclude_pattern = None

    sopa.make_transcript_patches(sdata, min_points_per_patch=0)

    df = pd.read_csv(
        sopa.utils.get_cache_dir(sdata) / SopaFiles.TRANSCRIPT_CACHE_DIR / "0" / SopaFiles.TRANSCRIPTS_FILE
    )

    assert len(df) == len(sdata["transcripts"])

    sopa.settings.gene_exclude_pattern = _default_gene_exclude_pattern


def test_patches_with_and_without_centroids():
    gdf = ShapesModel.parse(gpd.GeoDataFrame(geometry=[Point(100, 100).buffer(80)]))
    im = Image2DModel.parse(np.zeros((1, 200, 200), dtype=np.uint8))

    sdata = SpatialData(images={"im": im}, shapes={"gdf": gdf})

    patches = Patches2D(sdata, element="im", patch_width=67, patch_overlap=0, roi_key="gdf")
    assert len(patches) == 9
    assert patches.geo_df.index[0] == 0

    patches = Patches2D(sdata, element="im", patch_width=67, patch_overlap=0, roi_key="gdf", use_roi_centroids=True)
    assert len(patches) == 1
    assert patches.geo_df.index[0] == 0


def test_unassigned_to_zero():
    df = pd.DataFrame({"col1": ["a", "b", "zero", "b"], "col2": [-1, 0, 1, 2], "col3": [0, 1, 2, 10]})
    df = dd.from_pandas(df, npartitions=1)

    assert (_unassigned_to_zero(df["col1"], "zero").compute() == [1, 2, 0, 2]).all()
    assert (_unassigned_to_zero(df["col2"], -1).compute() == [0, 9223372036854775806, 1, 2]).all()
    assert (_unassigned_to_zero(df["col3"], 0).compute() == [0, 1, 2, 10]).all()


def test_move_sdata_transcript_cache():
    sdata = sopa.io.toy_dataset()

    sopa.segmentation.tissue(sdata)
    sdata.write("tests/test.zarr", overwrite=True)

    sopa.make_transcript_patches(sdata, patch_width=70, patch_overlap=0)

    shutil.move("tests/test.zarr", "tests/test_moved.zarr")

    sdata = spatialdata.read_zarr("tests/test_moved.zarr")
    _check_transcript_patches(sdata)  # the cache should still be detected

    shutil.rmtree("tests/test_moved.zarr")


def test_patch_assignment():
    genes = ["G1", "G2", "G3", "G4", "G5", "G6"]

    df = pd.DataFrame({
        "x": [0, 0.1, 1, 0, 1, 0.9],
        "y": [0, 0.1, 0, 1, 1, 0.9],
        "genes": genes,
    })
    points = {
        "transcripts": PointsModel.parse(df, feature_key="genes"),
    }

    sdata = SpatialData(points=points)

    cache_dir = sopa.utils.get_cache_dir(sdata)

    sopa.make_transcript_patches(sdata, patch_width=-1, min_points_per_patch=0)

    patch_dirs = list((cache_dir / "transcript_patches").iterdir())
    assert len(patch_dirs) == 1
    assert (pd.read_csv(patch_dirs[0] / "transcripts.csv").genes == genes).all()

    sopa.make_transcript_patches(sdata, patch_width=0.5, patch_overlap=0, min_points_per_patch=0)

    patch_dirs = list((cache_dir / "transcript_patches").iterdir())
    assert len(patch_dirs) == 4
    assert (pd.read_csv(cache_dir / "transcript_patches" / "0" / "transcripts.csv").genes == genes[:2]).all()
    assert (pd.read_csv(cache_dir / "transcript_patches" / "1" / "transcripts.csv").genes == genes[2:3]).all()
    assert (pd.read_csv(cache_dir / "transcript_patches" / "2" / "transcripts.csv").genes == genes[3:4]).all()
    assert (pd.read_csv(cache_dir / "transcript_patches" / "3" / "transcripts.csv").genes == genes[4:]).all()


def test_is_full_slide():
    sdata = sopa.io.toy_dataset(length=500)

    patches = OnDiskTranscriptPatches(
        sdata,
        "transcripts",
        patch_width=None,
        patch_overlap=0,
        prior_shapes_key=None,
        unassigned_value=None,
        min_points_per_patch=0,
        write_cells_centroids=False,
        roi_key=None,
    )

    assert patches.is_full_slide

    patches = OnDiskTranscriptPatches(
        sdata,
        "transcripts",
        patch_width=1000,
        patch_overlap=0,
        prior_shapes_key=None,
        unassigned_value=None,
        min_points_per_patch=0,
        write_cells_centroids=False,
        roi_key=None,
    )

    assert patches.is_full_slide

    patches = OnDiskTranscriptPatches(
        sdata,
        "transcripts",
        patch_width=10,
        patch_overlap=0,
        prior_shapes_key=None,
        unassigned_value=None,
        min_points_per_patch=0,
        write_cells_centroids=False,
        roi_key=None,
    )

    assert not patches.is_full_slide

    sopa.segmentation.tissue(sdata)

    patches = OnDiskTranscriptPatches(
        sdata,
        "transcripts",
        patch_width=None,
        patch_overlap=0,
        prior_shapes_key=None,
        unassigned_value=None,
        min_points_per_patch=0,
        write_cells_centroids=False,
        roi_key=SopaKeys.ROI,
    )

    assert not patches.is_full_slide
