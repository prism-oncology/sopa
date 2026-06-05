import json
import logging
import zipfile
from math import ceil
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from zarr.storage import ZipStore

from ._constants import ExplorerConstants, FileNames
from .utils import explorer_file_path

log = logging.getLogger(__name__)

_BLOSC = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
_COMPRESSOR_META = {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}


def write_transcripts(
    path: Path,
    df: dd.DataFrame,
    gene: str = "gene",
    max_levels: int = 15,
    is_dir: bool = True,
    pixel_size: float = 0.2125,
):
    """Write a `transcripts.zarr.zip` file containing pyramidal transcript locations and genes densities.

    Args:
        path: Path to the Xenium Explorer directory where the transcript file will be written
        df: DataFrame representing the transcripts, with `"x"`, `"y"` column required, as well as the `gene` column (see the corresponding argument)
        gene: Column of `df` containing the genes names.
        max_levels: Maximum number of levels in the pyramid.
        is_dir: If `False`, then `path` is a path to a single file, not to the Xenium Explorer directory.
        pixel_size: Number of microns in a pixel. Invalid value can lead to inconsistent scales in the Explorer.
    """
    path = explorer_file_path(path, FileNames.POINTS, is_dir)

    df = df.compute()

    num_transcripts = len(df)
    grid_size = ExplorerConstants.GRID_SIZE / ExplorerConstants.PIXELS_TO_MICRONS * pixel_size
    df[gene] = df[gene].astype("category")

    xy: np.ndarray = (df[["x", "y"]] * pixel_size).values

    if xy.min() < 0:
        log.warning("Some transcripts are located outside of the image (pixels < 0)")
    log.info(f"Writing {len(df)} transcripts")

    xmax, ymax = xy.max(axis=0)

    gene_names = list(df[gene].cat.categories)
    num_genes = len(gene_names)

    codeword_gene_mapping = list(range(num_genes))

    valid = np.ones((num_transcripts, 1), dtype=np.uint8)
    arange = np.arange(num_transcripts, dtype=np.uint32)
    uuid = np.stack([arange, np.full(num_transcripts, 65535, dtype=np.uint32)], axis=1)
    transcript_id = np.stack([arange, np.full(num_transcripts, 65535, dtype=np.uint32)], axis=1)
    gene_identity = df[gene].cat.codes.values[:, None].astype(np.uint16)
    codeword_identity = np.stack([gene_identity[:, 0], np.full(num_transcripts, 65535, dtype=np.uint16)], axis=1)
    status = np.ones((num_transcripts, 1), dtype=np.uint8)
    quality_score = np.full((num_transcripts, 1), ExplorerConstants.QUALITY_SCORE, dtype=np.float32)

    ATTRS = {
        "codeword_count": num_genes,
        "codeword_gene_mapping": codeword_gene_mapping,
        "codeword_gene_names": gene_names,
        "gene_names": gene_names,
        "gene_index_map": dict(zip(gene_names, codeword_gene_mapping)),
        "number_genes": num_genes,
        "spatial_units": "micron",
        "coordinate_space": "refined-final_global_micron",
        "major_version": 4,
        "minor_version": 1,
        "name": "RnaDataset",
        "number_rnas": num_transcripts,
        "dataset_uuid": "unique-id-test",
        "data_format": 0,
    }

    GRIDS_ATTRS: dict[str, bool | list[Any | list[Any]]] = {
        "grid_key_names": ["grid_x_loc", "grid_y_loc"],
        "grid_zip": False,
        "grid_size": [grid_size],
        "grid_array_shapes": [],
        "grid_number_objects": [],
        "grid_keys": [],
    }

    subsampling_locs = {0: np.arange(num_transcripts)}

    with ZipStore(path, mode="w") as store:
        g = zarr.group(store=store, zarr_format=2, attributes=ATTRS)
        zf = store._zf  # bypass zarr 3's per-object API overhead; _zf is the underlying ZipFile

        _write_zgroup(zf, "grids")

        for level in range(max_levels):
            level_locs = subsampling_locs[level]
            log.info(f"   > Level {level}: {len(level_locs)} transcripts")

            tile_size = grid_size * 2**level
            level_xy = xy[level_locs]

            lvl_valid = valid[level_locs]
            lvl_status = status[level_locs]
            lvl_gene_identity = gene_identity[level_locs]
            lvl_codeword = codeword_identity[level_locs]
            lvl_quality = quality_score[level_locs]
            lvl_uuid = uuid[level_locs]
            lvl_transcript_id = transcript_id[level_locs]

            tx = np.floor(level_xy[:, 0] / tile_size).clip(0).astype(int)
            ty = np.floor(level_xy[:, 1] / tile_size).clip(0).astype(int)

            # sort=True preserves ascending (tx, ty) order, which GRIDS_ATTRS consumers expect.
            groups = pd.DataFrame({"tx": tx, "ty": ty}).groupby(["tx", "ty"], sort=True).indices

            n_tiles_x, n_tiles_y = ceil(xmax / tile_size), ceil(ymax / tile_size)

            GRIDS_ATTRS["grid_array_shapes"].append([])
            GRIDS_ATTRS["grid_number_objects"].append([])
            GRIDS_ATTRS["grid_keys"].append([])

            _write_zgroup(zf, f"grids/{level}")

            for (gtx, gty), loc in groups.items():
                str_index = f"{gtx},{gty}"
                n_pts = len(loc)
                location = np.concatenate([level_xy[loc], _z(n_pts)], axis=1).astype(np.float32)
                base = f"grids/{level}/{str_index}"

                _write_zgroup(zf, base)
                _write_zarray(zf, f"{base}/valid", lvl_valid[loc])
                _write_zarray(zf, f"{base}/status", lvl_status[loc])
                _write_zarray(zf, f"{base}/location", location)
                _write_zarray(zf, f"{base}/gene_identity", lvl_gene_identity[loc])
                _write_zarray(zf, f"{base}/quality_score", lvl_quality[loc])
                _write_zarray(zf, f"{base}/codeword_identity", lvl_codeword[loc])
                _write_zarray(zf, f"{base}/uuid", lvl_uuid[loc])
                _write_zarray(zf, f"{base}/id", lvl_transcript_id[loc])

                GRIDS_ATTRS["grid_array_shapes"][-1].append({})
                GRIDS_ATTRS["grid_keys"][-1].append(str_index)
                GRIDS_ATTRS["grid_number_objects"][-1].append(n_pts)

            if n_tiles_x * n_tiles_y == 1:
                GRIDS_ATTRS["number_levels"] = level + 1
                break

            if level + 1 < max_levels:
                subsampling_locs[level + 1] = _subsample_indices(level_locs)

        zf.writestr("grids/.zattrs", json.dumps(GRIDS_ATTRS).encode())
        _write_density(g, xy, gene_identity[:, 0], gene_names)


def _write_zgroup(zf: zipfile.ZipFile, path: str) -> None:
    zf.writestr(f"{path}/.zgroup", b'{"zarr_format": 2}')


def _write_zarray(zf: zipfile.ZipFile, path: str, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    meta = {
        "zarr_format": 2,
        "shape": list(arr.shape),
        "chunks": list(arr.shape),
        "dtype": arr.dtype.str,
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": ".",
        "compressor": _COMPRESSOR_META,
    }
    zf.writestr(f"{path}/.zarray", json.dumps(meta).encode())
    chunk_key = ".".join("0" for _ in arr.shape)
    zf.writestr(f"{path}/{chunk_key}", _BLOSC.encode(arr))


def _subsample_indices(indices: np.ndarray, factor: int = 4) -> np.ndarray:
    return np.random.choice(indices, len(indices) // factor, replace=False)


def _classify_gene_names(names: list[str], n_cols: int = 9) -> np.ndarray:
    """Build a boolean category matrix classifying gene/codeword names.

    Columns follow the native Xenium format:
        0: is_gene, 1: is_negative_control_probe, 2: is_negative_control_codeword,
        3: is_deprecated, 4: is_predicted_false_positive, 5: is_unassigned_codeword,
        6: is_blankcodeword, 7: is_anti_sense, 8: is_genomic_control
    """
    lower = pd.Series(names).str.lower()
    is_neg = lower.str.contains("negcontrol|negprobe", regex=True, na=False).to_numpy()
    is_blank = lower.str.contains("blank", na=False).to_numpy() & ~is_neg
    is_dep = lower.str.contains("systemcontrol|falsecode", regex=True, na=False).to_numpy() & ~is_neg & ~is_blank
    is_gene = ~(is_neg | is_blank | is_dep)

    cats = np.zeros((len(names), n_cols), dtype=bool)
    cats[is_gene, 0] = True
    cats[is_neg, 1] = True
    cats[is_dep, 3] = True
    cats[is_blank, 6] = True
    return cats


def _write_density(
    root: zarr.Group,
    xy: np.ndarray,
    gene_indices: np.ndarray,
    gene_names: list[str],
):
    """Write the gene density map"""
    from scipy.sparse import coo_matrix

    n_genes = len(gene_names)

    ### Density grid (10 µm bins)
    grid_size = ExplorerConstants.DENSITY_GRID_SIZE
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)

    origin_x = np.floor(x_min / grid_size) * grid_size
    origin_y = np.floor(y_min / grid_size) * grid_size

    n_cols = int(np.ceil((x_max - origin_x) / grid_size)) + 1
    n_rows = int(np.ceil((y_max - origin_y) / grid_size)) + 1

    col_idx = np.floor((xy[:, 0] - origin_x) / grid_size).astype(np.int32).clip(0, n_cols - 1)
    row_idx = np.floor((xy[:, 1] - origin_y) / grid_size).astype(np.int32).clip(0, n_rows - 1)

    # CSR: rows = gene_index * n_rows + grid_row, cols = grid_col
    csr_row = gene_indices.astype(np.int64) * n_rows + row_idx.astype(np.int64)
    density_csr = coo_matrix(
        (np.ones(len(csr_row), dtype=np.uint16), (csr_row, col_idx)),
        shape=(n_genes * n_rows, n_cols),
    ).tocsr()
    density_csr.data = density_csr.data.astype(np.uint16)
    density_csr.indices = density_csr.indices.astype(np.uint16)
    density_csr.indptr = density_csr.indptr.astype(np.uint32)

    density_attrs = {
        "grid_size": [grid_size, grid_size],
        "rows": n_rows,
        "cols": n_cols,
        "origin": {"x": float(origin_x), "y": float(origin_y)},
    }

    density_group = root.create_group("density")

    gene_density = density_group.create_group("gene", attributes={**density_attrs, "gene_names": gene_names})
    gene_density.create_array("data", data=density_csr.data, chunks=(len(density_csr.data),))
    gene_density.create_array("indices", data=density_csr.indices, chunks=(len(density_csr.indices),))
    gene_density.create_array("indptr", data=density_csr.indptr, chunks=(len(density_csr.indptr),))

    ### Gene / codeword categories
    gene_cat = _classify_gene_names(gene_names)
    root.create_array("gene_category", data=gene_cat, chunks=(n_genes, 1))
    root.create_array("codeword_category", data=gene_cat, chunks=(n_genes, 1))

    ### Metrics density (20 µm bins)
    spacing = ExplorerConstants.METRICS_DENSITY_SPACING
    mx_count = int(np.ceil((x_max - origin_x) / spacing)) + 1
    my_count = int(np.ceil((y_max - origin_y) / spacing)) + 1

    m_col = np.floor((xy[:, 0] - origin_x) / spacing).astype(np.int32).clip(0, mx_count - 1)
    m_row = np.floor((xy[:, 1] - origin_y) / spacing).astype(np.int32).clip(0, my_count - 1)

    # Channels follow the native Xenium layout:
    #   0 = total, 1 = decoded, 2 = negative_control_probes, 3 = negative_control_codewords. Channel 3 stays zero when there is no codeword-level negatives. Decoded == total for the same reason.
    metrics = np.zeros((my_count, mx_count, 4), dtype=np.float32)
    np.add.at(metrics[:, :, 0], (m_row, m_col), 1.0)
    metrics[:, :, 1] = metrics[:, :, 0]

    is_neg = gene_cat[gene_indices, 1]  # is_negative_control_probe
    if is_neg.any():
        np.add.at(metrics[:, :, 2], (m_row[is_neg], m_col[is_neg]), 1.0)

    root.create_array("metrics_density", data=metrics, chunks=metrics.shape)

    root.attrs.update({
        "metrics_density_x_count": mx_count,
        "metrics_density_x_origin": float(origin_x),
        "metrics_density_x_spacing": spacing,
        "metrics_density_y_count": my_count,
        "metrics_density_y_origin": float(origin_y),
        "metrics_density_y_spacing": spacing,
    })

    log.info(f"   > Density grid: {n_rows}x{n_cols} ({grid_size}µm bins), {density_csr.nnz} non-zero entries")


def _z(n: int) -> np.ndarray:
    """Create a zero array with a small non-zero value at the first element to avoid reading issues."""
    z = np.zeros((n, 1))
    z[0] = 1e-4
    return z
