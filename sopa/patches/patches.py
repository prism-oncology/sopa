import logging
from math import ceil

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, SpatialElement
from xarray import DataArray, DataTree

from ..constants import SopaKeys
from ..shapes import to_valid_polygons
from ..utils import add_spatial_element, copy_transformations, to_intrinsic

log = logging.getLogger(__name__)


class Patches1D:
    def __init__(
        self,
        xmin: float | int,
        xmax: float | int,
        patch_width: float | int,
        patch_overlap: float | int,
        tight: bool,
        int_coords: bool,
    ):
        self.xmin, self.xmax = xmin, xmax
        self.delta = self.xmax - self.xmin

        self.patch_width = int(xmax - xmin + 1) if patch_width == float("inf") else patch_width
        self.patch_overlap = patch_overlap
        self.int_coords = int_coords

        self._count = self.count()
        if tight:
            self.patch_width = self.tight_width()
            assert self._count == self.count(), (
                f"Invalid patching with {self.delta=}, {self.patch_width=} and {self.patch_overlap=}"
            )

    def count(self):
        if self.patch_width >= self.delta:
            return 1
        return ceil((self.delta - self.patch_overlap) / (self.patch_width - self.patch_overlap))

    def update(self, patch_width):
        self.patch_width = patch_width
        assert self._count == self.count()

    def tight_width(self):
        width = (self.delta + (self._count - 1) * self.patch_overlap) / self._count
        return ceil(width) if self.int_coords else width + 1e-4  # add an epsilon to avoid floating point errors (#214)

    def __getitem__(self, i):
        start_delta = i * (self.patch_width - self.patch_overlap)
        x0, x1 = self.xmin + start_delta, self.xmin + start_delta + self.patch_width

        return [int(x0), int(x1)] if self.int_coords else [x0, x1]


class Patches2D:
    """
    Compute 2D-patches with overlaps. This can be done on an image or a DataFrame.

    Attributes:
        geo_df (GeoDataFrame): GeoPandas dataframe representing the patches
        bboxes (np.ndarray): Array of shape `(n_patches, 4)` containing the (xmin, ymin, xmax, ymax) coordinates of the patches bounding boxes
    """

    bboxes: np.ndarray
    geo_df: gpd.GeoDataFrame
    roi: Polygon | MultiPolygon | gpd.GeoDataFrame | None

    def __init__(
        self,
        sdata: SpatialData,
        element: SpatialElement | str,
        patch_width: float | int | None,
        patch_overlap: float | int = 50,
        roi_key: str | None = SopaKeys.ROI,
        use_roi_centroids: bool = False,
        scale: str | None = None,
    ):
        """
        Args:
            sdata: A `SpatialData` object
            element: SpatialElement or name of the element on with patches will be made (image or points)
            patch_width: Width of the patches (in the unit of the coordinate system of the element). If `None`, only one patch will be created
            patch_overlap: Overlap width between the patches
            roi_key: Optional name of the shapes that need to touch the patches. Patches that do not touch any shape will be ignored. If `None`, all patches will be used.
            use_roi_centroids: If `True`, the ROI will be computed from the centroids of the shapes in `roi_key`. If `False`, the ROI will be computed from the shapes themselves.
            scale: If the element is a DataTree, the scale to be used for patching (e.g., `"scale1"`). By default, uses the highest resolution scale, i.e., `"scale0"`.
        """
        patch_width = float("inf") if (patch_width is None or patch_width == -1) else patch_width

        assert patch_width > patch_overlap, f"Argument {patch_width=} must be greater than {patch_overlap=}"

        self.sdata = sdata
        self.element = sdata[element] if isinstance(element, str) else element

        self.scale = scale
        if isinstance(self.element, DataTree):
            assert scale is None or scale in self.element, (
                f"Scale '{scale}' not found. Available scales: {', '.join(self.element.keys())}"
            )
            self.element = next(iter(self.element[scale or "scale0"].values()))
        elif scale is not None:
            raise ValueError("Scales are only supported for DataTree elements.")

        if isinstance(self.element, DataArray):
            xmin, ymin = 0, 0
            xmax, ymax = len(self.element.coords["x"]), len(self.element.coords["y"])
            tight, int_coords = False, True
        elif isinstance(self.element, dd.DataFrame):
            xmin, ymin = self.element.x.min().compute(), self.element.y.min().compute()
            xmax, ymax = self.element.x.max().compute(), self.element.y.max().compute()
            tight, int_coords = True, False
        else:
            raise TypeError(f"Invalid element type: {type(self.element)}")

        self.patch_x = Patches1D(xmin, xmax, patch_width, patch_overlap, tight, int_coords)
        self.patch_y = Patches1D(ymin, ymax, patch_width, patch_overlap, tight, int_coords)

        self.roi = None
        assert roi_key is None or roi_key in sdata.shapes or roi_key == SopaKeys.ROI, f"Invalid {roi_key=}"

        if roi_key is not None and roi_key in sdata.shapes:
            roi_geo_df = to_intrinsic(sdata, sdata[roi_key], self.element)
            self.roi = _get_roi(roi_geo_df, use_roi_centroids)

        self.geo_df = self._init_patches(use_roi_centroids)
        self.bboxes = np.array(self.geo_df[SopaKeys.BOUNDS].to_list())

    def _init_patches(self, use_roi_centroids: bool) -> gpd.GeoDataFrame:
        data = {
            "geometry": [],
            SopaKeys.PATCHES_ILOCS: [],
            SopaKeys.BOUNDS: [],
        }

        for i in range(self.patch_x._count * self.patch_y._count):
            iy, ix = divmod(i, self.patch_x._count)
            bounds = self._bbox_iloc(ix, iy)
            patch = box(*bounds)

            data["geometry"].append(patch)
            data[SopaKeys.PATCHES_ILOCS].append((ix, iy))
            data[SopaKeys.BOUNDS].append(bounds)

        geo_df = gpd.GeoDataFrame(data)

        if self.roi is not None:
            if use_roi_centroids:
                indices = gpd.sjoin(geo_df, self.roi).index.unique()
                geo_df = geo_df.loc[indices]
            else:
                geo_df.geometry = geo_df.geometry.intersection(self.roi)
                geo_df = geo_df[~geo_df.is_empty]

        geo_df = to_valid_polygons(geo_df, simple_polygon=False)
        geo_df = geo_df.reset_index(drop=True)

        assert len(geo_df), "No valid patches found inside the provided region of interest."

        if self.scale is not None:
            geo_df[SopaKeys.SCALE] = self.scale

        return ShapesModel.parse(geo_df, transformations=copy_transformations(self.element))

    def __repr__(self):
        return f"{self.__class__.__name__} object with {len(self)} patches"

    @property
    def shape(self) -> tuple[int, int]:
        return (self.patch_y._count, self.patch_x._count)

    @property
    def is_full_slide(self) -> bool:
        return self.roi is None and len(self) == 1

    def centroids(self) -> np.ndarray:
        """Get the centroids of the patches as a numpy array of shape `(n_patches, 2)`.
        This doesn't take into account the ROI, so it may return centroids outside the ROI."""
        x = (self.bboxes[:, 0] + self.bboxes[:, 2]) / 2
        y = (self.bboxes[:, 1] + self.bboxes[:, 3]) / 2

        return np.column_stack((x, y))

    def _bbox_iloc(self, ix: int, iy: int) -> list[int]:
        """Coordinates of the rectangle bounding box of the patch at the given indices

        Args:
            ix: Patch index in the x-axis
            iy: Patch indes in the y-axis

        Returns:
            A list `[xmin, ymin, xmax, ymax]` representing the bounding box of the patch
        """
        xmin, xmax = self.patch_x[ix]
        ymin, ymax = self.patch_y[iy]
        return [xmin, ymin, xmax, ymax]

    def __len__(self):
        """Number of patches"""
        return len(self.geo_df)

    def add_shapes(self, key_added: str | None = None, overwrite: bool = True) -> gpd.GeoDataFrame:
        key_added = key_added or SopaKeys.PATCHES

        add_spatial_element(self.sdata, key_added, self.geo_df, overwrite=overwrite)

        log.info(f"Added {len(self.geo_df)} patch(es) to sdata['{key_added}']")

        return self.geo_df


def _get_roi(geo_df: gpd.GeoDataFrame, use_roi_centroids: bool) -> Polygon | MultiPolygon | gpd.GeoDataFrame:
    """Merge all geometries into a single region-of-interest"""
    if use_roi_centroids:
        return geo_df.centroid.to_frame()

    roi = geo_df.geometry.union_all()

    if isinstance(roi, GeometryCollection):
        _previous_area = roi.area

        roi = MultiPolygon([geom for geom in roi.geoms if isinstance(geom, Polygon)])
        if roi.area < _previous_area * 0.999:
            raise ValueError("ROI is a GeometryCollection that could not be simplified into polygons.")

    assert isinstance(roi, (Polygon, MultiPolygon)), f"Invalid ROI type: {type(roi)}. Must be Polygon or MultiPolygon"

    return roi
