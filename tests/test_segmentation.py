import numpy as np
import pytest
import shapely
from geopandas.testing import assert_geodataframe_equal
from shapely import MultiPolygon
from spatialdata import SpatialData
from spatialdata.models import Image2DModel

import sopa
from sopa.constants import SopaKeys
from sopa.segmentation.stainings import _channels_average_within_mask


def test_channels_average_within_mask():
    image = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 20, 30], [40, 50, 60], [70, 80, 90]]])
    mask = np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0]])

    expected = np.array([
        [[1.0, 2.0, 3.0], [4.0, 4.0, 6.0], [4.0, 8.0, 4.0]],
        [[10.0, 20.0, 30.0], [40.0, 40.0, 60.0], [40.0, 80.0, 40.0]],
    ])

    assert (_channels_average_within_mask(image, mask) == expected).all()


@pytest.mark.cellpose_v4
def test_cellpose_segmentation():
    sdata = sopa.io.toy_dataset(length=200)

    sopa.make_image_patches(sdata, patch_width=125, patch_overlap=40)

    assert len(sdata[SopaKeys.PATCHES]) == 4

    sopa.settings.parallelization_backend = None

    sopa.segmentation.cellpose(sdata, "DAPI", 35)
    cells_no_backend = sdata[SopaKeys.CELLPOSE_BOUNDARIES].copy()

    sopa.settings.parallelization_backend = "dask"

    sopa.segmentation.cellpose(sdata, "DAPI", 35)
    cells_dask_backend = sdata[SopaKeys.CELLPOSE_BOUNDARIES].copy()

    assert_geodataframe_equal(cells_no_backend, cells_dask_backend)

    sopa.settings.parallelization_backend = None


def test_tissue_segmentation():
    sdata = sopa.io.toy_dataset(length=500)

    sopa.segmentation.tissue(sdata, key_added="tissue_seg")  # H&E segmentation

    assert "tissue_seg" in sdata.shapes

    sopa.segmentation.tissue(sdata, mode="staining")

    sopa.segmentation.tissue(sdata, mode="staining", level=0)

    m1 = sdata["cells"].union_all()
    m2 = shapely.make_valid(MultiPolygon(sdata["cells"].geometry.values))

    assert m2.intersection(m1).area / m2.area > 0.99

    assert m1.intersection(m2).area / m1.area > 0.5

    sopa.segmentation.tissue(sdata, level=1)

    geo_df = sdata[SopaKeys.ROI].copy()
    m1_default = MultiPolygon(geo_df.geometry.values)

    geo_df = sopa.utils.to_intrinsic(sdata, sdata[SopaKeys.ROI], sdata["he_image"]).copy()
    m1_default_transformed = MultiPolygon(geo_df.geometry.values)

    assert m1_default_transformed.intersection(m1_default).area / m1_default_transformed.area < 0.2

    sopa.segmentation.tissue(sdata, level=0)

    geo_df = sdata[SopaKeys.ROI].copy()
    m1_default_level0 = MultiPolygon(geo_df.geometry.values)

    assert m1_default_transformed.intersection(m1_default_level0).area / m1_default_transformed.area > 0.9


def test_tissue_segmentation_drop_threshold():
    X = np.zeros((1, 50, 50))
    X[0, 20:40, 20:40] = 1  # 400 pixels
    X[0, :10, :10] = 1  # 100 pixels, i.e. 20% of the total area

    sdata = SpatialData(images={"image": Image2DModel.parse(X, c_coords=["DAPI"])})

    sopa.segmentation.tissue(sdata, drop_threshold=0.19)
    assert len(sdata["region_of_interest"]) == 2

    sopa.segmentation.tissue(sdata, drop_threshold=0.21)
    assert len(sdata["region_of_interest"]) == 1
