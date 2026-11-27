import logging

import dask
import geopandas as gpd
import numpy as np
import numpy.ma as ma
import pandas as pd
import shapely
from anndata import AnnData
from shapely.geometry import box
from spatialdata import SpatialData
from xarray import DataArray

from ..shapes import expand_radius, pixel_outer_bounds, rasterize
from ..utils import get_boundaries, get_spatial_image, to_intrinsic, validated_channel_names
from .table import parse_table

log = logging.getLogger(__name__)

AVAILABLE_MODES = ["average", "min", "max"]


def aggregate_channels(
    sdata: SpatialData,
    image_key: str | None = None,
    shapes_key: str | None = None,
    expand_radius_ratio: float = 0,
    mode: str = "average",
    no_overlap: bool = False,
) -> AnnData:
    """Aggregate the channel intensities inside the cells boundaries (either `"average"`, or take the `"min"` / `"max"`).

    Args:
        sdata: A `SpatialData` object
        image_key: Key of `sdata` containing the image. If only one `images` element, this does not have to be provided.
        shapes_key: Key of `sdata` containing the cell boundaries. If only one `shapes` element, this does not have to be provided.
        expand_radius_ratio: Cells polygons will be expanded by `expand_radius_ratio * mean_radius`. This help better aggregate boundary stainings.
        mode: Aggregation mode. One of `"average"`, `"min"`, `"max"`. By default, average the intensity inside the cell mask.
        no_overlap: If `True`, the (expanded) cells will not overlap.

    Returns:
        An `AnnData` object with the aggregated channel intensities in `adata.X`
    """
    assert mode in AVAILABLE_MODES, f"Invalid {mode=}. Available modes are {AVAILABLE_MODES}"

    image_key, image = get_spatial_image(sdata, image_key, return_key=True)

    shapes_key, geo_df = get_boundaries(sdata, key=shapes_key, return_key=True)
    geo_df = to_intrinsic(sdata, geo_df, image)
    geo_df = expand_radius(geo_df, expand_radius_ratio, no_overlap=no_overlap)

    adata = _aggregate_channels_aligned(image, geo_df, mode)

    return parse_table(adata, geo_df, shapes_key=shapes_key, image_key=image_key)


def _aggregate_channels_aligned(image: DataArray, geo_df: gpd.GeoDataFrame, mode: str) -> AnnData:
    """Average channel intensities per cell. The image and cells have to be aligned, i.e. be on the same coordinate system.

    Args:
        image: A `DataArray` of shape `(n_channels, y, x)`
        geo_df: A `GeoDataFrame` whose geometries are cell boundaries (polygons)

    Returns:
        An `AnnData` object with the aggregated channel intensities
    """
    from dask.diagnostics import ProgressBar

    log.info(f"Aggregating channels intensity over {len(geo_df)} cells with {mode=}")

    cells = geo_df.geometry.tolist()
    tree = shapely.STRtree(cells)

    n_channels = len(image.coords["c"])
    areas = np.zeros(len(cells))
    if mode == "min":
        aggregation = np.full((len(cells), n_channels), fill_value=np.inf)
    else:
        aggregation = np.zeros((len(cells), n_channels))

    chunk_sizes = image.data.chunks
    offsets_y = np.cumsum(np.pad(chunk_sizes[1], (1, 0), "constant"))
    offsets_x = np.cumsum(np.pad(chunk_sizes[2], (1, 0), "constant"))

    def _average_chunk_inside_cells(chunk, iy, ix):
        ymin, ymax = offsets_y[iy], offsets_y[iy + 1]
        xmin, xmax = offsets_x[ix], offsets_x[ix + 1]

        patch = box(xmin, ymin, xmax, ymax)
        intersections = tree.query(patch, predicate="intersects")

        for index in intersections:
            cell = cells[index]
            bounds = pixel_outer_bounds(cell.bounds)

            sub_image = chunk[
                :,
                max(bounds[1] - ymin, 0) : max(bounds[3] - ymin, 0),
                max(bounds[0] - xmin, 0) : max(bounds[2] - xmin, 0),
            ]

            if sub_image.shape[1] == 0 or sub_image.shape[2] == 0:
                continue

            mask = rasterize(cell, sub_image.shape[1:], (max(bounds[0], xmin), max(bounds[1], ymin)))
            mask_sum = np.sum(mask)
            if mask_sum == 0:
                continue

            areas[index] += mask_sum

            if mode == "min":
                masked_image = ma.masked_array(sub_image, 1 - np.repeat(mask[None], n_channels, axis=0))
                aggregation[index] = np.minimum(aggregation[index], masked_image.min(axis=(1, 2)))
            elif mode in ["average", "max"]:
                func = np.sum if mode == "average" else np.max
                values = func(sub_image * mask, axis=(1, 2))

                if mode == "average":
                    aggregation[index] += values
                else:
                    aggregation[index] = np.maximum(aggregation[index], values)

    with ProgressBar():
        tasks = [
            dask.delayed(_average_chunk_inside_cells)(chunk, iy, ix)
            for iy, row in enumerate(image.chunk({"c": -1}).data.to_delayed()[0])
            for ix, chunk in enumerate(row)
        ]
        dask.compute(tasks)

    X = aggregation if mode in ["min", "max"] else aggregation / areas[:, None].clip(1)

    return AnnData(
        X,
        var=pd.DataFrame(index=validated_channel_names(image)),
        obs=pd.DataFrame(index=geo_df.index.astype(str)),
    )
