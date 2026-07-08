import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import spatialdata
import xarray as xr
from anndata import AnnData
from scipy.sparse import csr_matrix
from shapely import Point, Polygon, box
from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel, TableModel

import sopa
from sopa.aggregation.channels import _aggregate_channels_aligned
from sopa.aggregation.transcripts import _count_transcripts_aligned
from sopa.constants import SopaKeys


def test_aggregate_channels_aligned():
    image = np.random.randint(1, 10, size=(3, 8, 16))
    arr = da.from_array(image, chunks=(1, 8, 8))
    channel_names = ["DAPI", "CD3", "CD20"]
    xarr = xr.DataArray(arr, dims=["c", "y", "x"], coords={"c": channel_names})

    cell_size = 4
    cell_start = [(0, 0), (6, 2), (9, 3)]

    # One cell is on the first block, one is overlapping on both blocks, and one is on the last block
    cells = [box(x, y, x + cell_size - 1, y + cell_size - 1) for x, y in cell_start]
    gdf = gpd.GeoDataFrame(geometry=cells)

    adata_mean_intensities = _aggregate_channels_aligned(xarr, gdf, "average")
    adata_min_intensities = _aggregate_channels_aligned(xarr, gdf, "min")
    adata_max_intensities = _aggregate_channels_aligned(xarr, gdf, "max")

    true_mean_intensities = np.stack([
        image[:, y : y + cell_size, x : x + cell_size].mean(axis=(1, 2)) for x, y in cell_start
    ])
    true_min_intensities = np.stack([
        image[:, y : y + cell_size, x : x + cell_size].min(axis=(1, 2)) for x, y in cell_start
    ])
    true_max_intensities = np.stack([
        image[:, y : y + cell_size, x : x + cell_size].max(axis=(1, 2)) for x, y in cell_start
    ])

    assert (true_mean_intensities == adata_mean_intensities.X).all()
    assert (true_min_intensities == adata_min_intensities.X).all()
    assert (true_max_intensities == adata_max_intensities.X).all()


def test_count_transcripts():
    df_pandas = pd.DataFrame({
        "x": [1, 2, 3, 7, 11, 1, 3, 2, 2, 1],
        "y": [1, 1, 2, 8, 0, 2, 4, 3, 3, 1],
        "gene": ["a", "a", "b", "c", "a", "c", "gene_control", "b", "b", "blank"],
    })
    df_pandas["gene"] = df_pandas["gene"].astype(object)
    df_pandas.loc[0, "gene"] = np.nan

    points = dd.from_pandas(df_pandas, npartitions=2)
    polygons = [
        Polygon(((1, 2), (3, 2), (3, 4), (1, 4))),
        Polygon(((0, 0), (2, 0), (2, 2), (0, 2))),
        Polygon(((0, 0), (3, 0), (3, 3), (0, 3))),
    ]

    gdf = gpd.GeoDataFrame(geometry=polygons)

    adata = _count_transcripts_aligned(gdf, points, "gene")
    expected = np.array([[0, 3, 1], [1, 0, 1], [1, 3, 1]])

    assert set(adata.var_names) == {"a", "b", "c"}
    adata = adata[:, ["a", "b", "c"]].copy()
    assert (adata.X.toarray() == expected).all()

    adata = _count_transcripts_aligned(gdf, points, "gene", only_excluded=True)
    expected = np.array([[0, 1], [1, 0], [1, 0]])

    assert set(adata.var_names) == {"blank", "gene_control"}
    adata = adata[:, ["blank", "gene_control"]].copy()
    assert (adata.X.toarray() == expected).all()


def test_count_transcripts_with_low_quality():
    df_pandas = pd.DataFrame({
        "x": [1, 2, 3, 1],
        "y": [1, 1, 2, 2],
        "gene": ["a", "a", "b", "b"],
        SopaKeys.LOW_QUALITY_TRANSCRIPT_KEY: [True, False, False, False],
    })
    df_pandas["gene"] = df_pandas["gene"].astype(object)

    points = dd.from_pandas(df_pandas, npartitions=2)
    polygons = [box(-1, -1, 10, 10)]

    gdf = gpd.GeoDataFrame(geometry=polygons)

    adata = _count_transcripts_aligned(gdf, points, "gene")

    assert set(adata.var_names) == {"a", "b"}
    adata = adata[:, ["a", "b"]].copy()
    assert (adata.X.toarray() == np.array([[1, 2]])).all()


def test_aggregation_more_cells():
    n_genes, n_cells = 53, 123  # arbitrary numbers of genes and cells

    counts = np.random.poisson(1, (n_cells, n_genes))

    adata_original = AnnData(
        X=csr_matrix(counts),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )

    locs = np.arange(n_cells * 2).reshape(n_cells, 2)

    gdf = gpd.GeoDataFrame(geometry=[box(xmin - 0.5, ymin - 0.5, xmin + 0.5, ymin + 0.5) for xmin, ymin in locs])

    counts_per_cell = counts.sum(1)
    n_transcripts = counts.sum()

    transcript_locs = locs.repeat(counts_per_cell, axis=0) + np.random.uniform(-0.4, 0.4, (n_transcripts, 2))

    x, y = transcript_locs.T

    var_names = np.asarray(adata_original.var_names)
    gene_names = np.concatenate([var_names[None, :].repeat(counts[i]) for i in range(n_cells)])

    df_pandas = pd.DataFrame({
        "x": x,
        "y": y,
        "gene": gene_names,
    })
    df_pandas["gene"] = df_pandas["gene"].astype(object)

    points = dd.from_pandas(df_pandas, npartitions=3)

    adata = _count_transcripts_aligned(gdf, points, "gene")

    assert np.array_equal(adata[:, adata_original.var_names].X.toarray(), counts)


def test_aggregate_bins():
    bins = [
        box(0, 0, 1, 1),
        box(1, 0, 2, 1),
        box(0, 1, 1, 2),
        box(5, 5, 6, 6),
        box(6, 5, 7, 6),
        box(5, 6, 6, 7),
        box(6, 6, 7, 7),
    ]

    gdf_bins = gpd.GeoDataFrame(geometry=bins)
    gdf_bins.index = [f"bin{i}" for i in range(7)]
    gdf_bins = ShapesModel.parse(gdf_bins)

    cells = [
        Polygon([(-1, -1), (0.5, 0.5), (1, -1)]),  # touches 0
        box(-1, -1, 4, 4),  # touches 0, 1, 2
        Polygon([(4, 4), (4, 9), (9, 4)]),  # touches 3, 4, 5, 6
    ]

    gdf_cells = gpd.GeoDataFrame(geometry=cells)
    gdf_cells = ShapesModel.parse(gdf_cells)

    counts = np.array([
        [3, 2, 1],
        [0, 1, 0],
        [10, 10, 15],
        [0, 0, 12],
        [2, 2, 2],
        [0, 4, 0],
        [32, 1, 2],
    ])

    expected_counts = np.array([
        [3, 2, 1],
        [13, 13, 16],
        [34, 7, 16],
    ])

    adata = AnnData(counts)
    adata.var_names = ["gene1", "gene2", "gene3"]

    adata.obs["bin_id"] = gdf_bins.index
    adata.obs["bin_key"] = "bins_2um"

    adata = TableModel.parse(adata, region="bins_2um", instance_key="bin_id", region_key="bin_key")

    sdata = SpatialData(shapes={"bins_2um": gdf_bins, "cells": gdf_cells}, tables={"table": adata})

    adata_aggr = sopa.aggregation.aggregate_bins(sdata, "cells", "table")

    assert list(adata_aggr.var_names) == ["gene1", "gene2", "gene3"]

    assert (expected_counts == adata_aggr.X).all()

    sdata.tables["table"].X = csr_matrix(sdata.tables["table"].X)

    adata_aggr = sopa.aggregation.aggregate_bins(sdata, "cells", "table")

    assert isinstance(adata_aggr.obsm["bins_assignments"], csr_matrix)

    assert (adata_aggr.obsm["bins_assignments"].nonzero()[1] == [0, 0, 1, 2, 3, 4, 5, 6]).all()

    assert list(adata_aggr.var_names) == ["gene1", "gene2", "gene3"]

    assert (adata_aggr.X.toarray() == expected_counts).all()


@pytest.mark.parametrize("as_sparse", [True, False])
def test_aggregate_bins_no_overlap(as_sparse: bool):
    gdf_bins = gpd.GeoDataFrame(geometry=[box(i, j, i + 0.9, j + 0.9) for i in range(4) for j in range(4)])
    gdf_bins.index = [f"bin{i}" for i in range(len(gdf_bins))]
    gdf_bins = ShapesModel.parse(gdf_bins)

    counts = np.array([
        [
            (i < 2) and (j < 2),
            1,
            (i >= 2 and j >= 2),
            (i < 2 and j >= 2),
        ]
        for i in range(4)
        for j in range(4)
    ]).astype(int)

    if as_sparse:
        counts = csr_matrix(counts)

    adata = AnnData(counts)
    adata.var_names = ["gene1", "gene2", "gene3", "gene4"]

    adata.obs["bin_id"] = gdf_bins.index
    adata.obs["bin_key"] = "bins_2um"

    gdf_cells = gpd.GeoDataFrame(geometry=[Point(0.7, 1.5), Point(1.3, 3), Point(2.5, 3.2)])
    gdf_cells.geometry = gdf_cells.buffer(0.6)
    gdf_cells = ShapesModel.parse(gdf_cells)

    adata = TableModel.parse(adata, region="bins_2um", instance_key="bin_id", region_key="bin_key")

    sdata = SpatialData(shapes={"bins_2um": gdf_bins, "cells": gdf_cells}, tables={"table": adata})

    adata_aggr1 = sopa.aggregation.aggregate_bins(sdata, "cells", "table", no_overlap=False, expand_radius_ratio=1.5)
    assert isinstance(adata_aggr1.obsm["bins_assignments"], csr_matrix)

    adata_aggr2 = sopa.aggregation.aggregate_bins(sdata, "cells", "table", no_overlap=True, expand_radius_ratio=1.5)
    assert isinstance(adata_aggr2.obsm["bins_assignments"], csr_matrix)

    X1 = adata_aggr1.X.toarray() if as_sparse else adata_aggr1.X
    X2 = adata_aggr2.X.toarray() if as_sparse else adata_aggr2.X

    assert (X2 <= X1).all()

    assert (np.array([[4, 6, 0, 2], [0, 6, 2, 4], [0, 6, 4, 2]]) == X1).all()

    assert (np.array([[4, 4, 0, 0], [0, 4, 0, 4], [0, 4, 4, 0]]) == X2).all()


def test_overlay():
    np.random.seed(0)
    x, y = np.meshgrid(np.arange(6), np.arange(6))
    x = x.flatten()
    y = y.flatten()
    genes = np.random.choice(["gene_a", "gene_b", "gene_c"], x.shape[0])
    points = PointsModel.parse(pd.DataFrame({"x": x, "y": y, "gene": genes}), feature_key="gene")

    gdf = gpd.GeoDataFrame(
        geometry=[
            box(0, 0, 1, 1),
            box(1, 1, 2, 2),
            box(2, 2, 4, 4),
            box(1, 2.5, 1.5, 5),
        ]
    )
    gdf.geometry = gdf.geometry.buffer(0.1)
    gdf = ShapesModel.parse(gdf)

    gdf2 = gpd.GeoDataFrame(
        geometry=[
            box(0, 0, 1, 5),
            box(3, 3, 5, 5),
        ]
    )
    gdf2.geometry = gdf2.geometry.buffer(0.1)
    gdf2 = ShapesModel.parse(gdf2)

    sdata = spatialdata.SpatialData(shapes={"cellpose_boundaries": gdf, "selection": gdf2}, points={"points": points})

    sopa.aggregate(sdata, aggregate_channels=False)

    assert (
        sdata["table"].X.toarray()
        == np.array([
            [2, 1, 1],
            [1, 0, 3],
            [5, 2, 2],
            [0, 1, 2],
        ])
    ).all()

    sopa.overlay_segmentation(sdata, "selection", area_ratio_threshold=0.5)

    assert (
        sdata["table"].X.toarray()
        == np.array([
            [1, 0, 3],
            [5, 2, 2],
            [0, 1, 2],
            [2, 6, 4],
            [3, 5, 1],
        ])
    ).all()


def test_aggregate_without_dropping_cells():
    np.random.seed(0)
    x, y = np.meshgrid(np.arange(6), np.arange(6))
    x = x.flatten()
    y = y.flatten()
    genes = np.random.choice(["gene_a", "gene_b", "gene_c"], x.shape[0])
    points = PointsModel.parse(pd.DataFrame({"x": x, "y": y, "gene": genes}), feature_key="gene")

    gdf = gpd.GeoDataFrame(
        geometry=[
            box(0, 0, 1, 1),
            box(1, 1, 2, 2),
            box(2, 2, 4, 4),
            box(1, 2.5, 1.5, 5),
        ]
    )
    gdf.geometry = gdf.geometry.buffer(0.1)
    gdf = ShapesModel.parse(gdf)

    sdata = spatialdata.SpatialData(shapes={"cellpose_boundaries": gdf}, points={"points": points})

    sopa.aggregate(
        sdata,
        shapes_key="cellpose_boundaries",
        points_key="points",
        aggregate_genes=True,
        aggregate_channels=True,
        min_transcripts=4,
        drop_filtered_cells=False,
        key_added="table_no_drop",
    )

    table = sdata["table_no_drop"]

    assert table.n_obs == 4
    assert SopaKeys.PASSES_FILTERING in table.obs
    assert (~table.obs[SopaKeys.PASSES_FILTERING].to_numpy()).sum() >= 1


def test_sequential_aggregation():
    sdata = sopa.io.toy_dataset(as_output=True)

    sdata["cellpose_boundaries"].index = [f"cellpose_{i}" for i in range(len(sdata["cellpose_boundaries"]))]
    del sdata["table"]

    sopa.aggregate(sdata, min_intensity_ratio=0.9)
    assert sdata["table"].n_obs < 400

    n_obs = sdata["table"].n_obs

    sopa.aggregate(sdata, drop_filtered_cells=False, min_intensity_ratio=0.9)

    assert sdata["table"].n_obs == n_obs
    assert 0.5 < sdata["table"].obs[SopaKeys.PASSES_FILTERING].mean() < 0.999
