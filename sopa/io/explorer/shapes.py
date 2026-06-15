import logging
from math import ceil
from pathlib import Path

import geopandas as gpd
import numpy as np
import zarr
from shapely.geometry import MultiPolygon, Polygon
from zarr.storage import ZipStore

from ._constants import FileNames, cell_summary_attrs, group_attrs
from .utils import explorer_file_path, int_cell_id

log = logging.getLogger(__name__)

TOLERANCE_STEP = 0.5


def pad_polygon(polygon: Polygon, max_vertices: int, tolerance: float = TOLERANCE_STEP) -> np.ndarray:
    """Transform the polygon to have the desired number of vertices

    Args:
        polygon: A `shapely` polygon
        max_vertices: The desired number of vertices
        tolerance: The step of tolerance used for simplification. At each step, we increase the tolerance of this value until the polygon is simplified enough.

    Returns:
        A 2D array representing the polygon vertices
    """
    n_vertices = len(polygon.exterior.coords)
    assert n_vertices >= 3

    if tolerance > 50:
        log.warning(
            f"Polygon with {n_vertices} vertices could not be simplified to {max_vertices} vertices. Using a circle instead."
        )
        polygon = polygon.centroid.buffer(np.sqrt(polygon.area) / 2)
        return pad_polygon(polygon, max_vertices, TOLERANCE_STEP)

    coords = polygon.exterior.coords._coords

    if n_vertices == max_vertices:
        return coords.flatten()

    if n_vertices < max_vertices:
        return np.pad(coords, ((0, max_vertices - n_vertices), (0, 0)), mode="edge").flatten()

    polygon = polygon.simplify(tolerance=tolerance)
    return pad_polygon(polygon, max_vertices, tolerance + TOLERANCE_STEP)


def write_polygons(
    path: Path,
    geo_df: gpd.GeoDataFrame,
    geo_df_nucleus: gpd.GeoDataFrame | None,
    max_vertices: int,
    pixel_size: float = 0.2125,
    preserve_ids: bool = True,
    is_dir: bool = True,
) -> None:
    """Write a `cells.zarr.zip` file containing the cell polygonal boundaries

    Args:
        path: Path to the Xenium Explorer directory where the transcript file will be written
        geo_df: A GeoDataFrame containing the `shapely` polygons to be written
        geo_df_nucleus: Same as `geo_df` but for the cells' nucleus
        max_vertices: The number of vertices per polygon (they will be transformed to have the right number of vertices)
        pixel_size: Number of microns in a pixel. Invalid value can lead to inconsistent scales in the Explorer.
        preserve_ids: If `True`, will preserve the cell IDs from `geo_df.index` in the explorer (they need to be valid Xenium Explorer IDs). If `False`, will use new IDs for the output file.
        is_dir: If `False`, then `path` is a path to a single file, not to the Xenium Explorer directory.
    """
    path = explorer_file_path(path, FileNames.SHAPES, is_dir)

    polygons, coordinates = _extract_coordinates(geo_df, max_vertices, pixel_size=pixel_size)
    coordinates_nucleus = (
        coordinates if geo_df_nucleus is None else _extract_coordinates(geo_df_nucleus, max_vertices, pixel_size)[1]
    )

    log.info(f"Writing {len(polygons)} cell polygons")

    num_cells = len(coordinates)
    cells_fourth = ceil(num_cells / 4)
    cells_half = ceil(num_cells / 2)

    GROUP_ATTRS = group_attrs()
    GROUP_ATTRS["number_cells"] = num_cells

    polygon_vertices = np.stack([coordinates_nucleus, coordinates])
    num_points = polygon_vertices.shape[2]
    n_vertices = num_points // 2

    with ZipStore(path, mode="w") as store:
        g = zarr.group(store=store, zarr_format=2, attributes=GROUP_ATTRS)

        g.create_array(
            "polygon_vertices",
            data=polygon_vertices.astype(np.float32),
            chunks=(1, cells_fourth, ceil(num_points / 4)),
        )

        cell_id = np.ones((num_cells, 2))
        cell_id[:, 0] = int_cell_id(geo_df.index) if preserve_ids else np.arange(num_cells)
        g.create_array("cell_id", data=cell_id.astype(np.uint32), chunks=(cells_half, 1))

        cell_summary = np.zeros((num_cells, 7))
        cell_summary[:, 2] = [p.area for p in polygons]
        g.create_array(
            "cell_summary", data=cell_summary.astype(np.float64), chunks=(num_cells, 1), attributes=cell_summary_attrs()
        )

        g.create_array(
            "polygon_num_vertices", data=np.full((2, num_cells), n_vertices, dtype=np.int32), chunks=(1, cells_half)
        )

        g.create_array(
            "seg_mask_value",
            data=np.arange(num_cells, dtype=np.uint32),
            chunks=(cells_half,),
        )


def _extract_coordinates(
    geo_df: gpd.GeoDataFrame, max_vertices: int, pixel_size: float
) -> tuple[list[Polygon], np.ndarray]:
    polygons = geo_df.geometry.values

    assert all(isinstance(geom, (Polygon, MultiPolygon)) for geom in polygons), (
        "All geometries must be a `shapely.Polygon` or `shapely.MultiPolygon`"
    )

    if any(isinstance(geom, MultiPolygon) for geom in polygons):
        log.warning("Some cells are MultiPolygons. Only the polygon with the largest area will be kept for each cell.")
        polygons = [_to_largest_polygon(p) for p in polygons]

    coordinates = np.stack([pad_polygon(p, max_vertices) for p in polygons])
    coordinates *= pixel_size

    return polygons, coordinates


def _to_largest_polygon(polygon: Polygon | MultiPolygon) -> Polygon:
    if isinstance(polygon, Polygon):
        return polygon
    return max(polygon.geoms, key=lambda x: x.area)
