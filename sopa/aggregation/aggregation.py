import logging

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix
from spatialdata import SpatialData

from ..constants import SopaAttrs, SopaKeys
from ..utils import add_spatial_element, get_boundaries, get_spatial_element, get_spatial_image
from . import aggregate_bins, count_transcripts
from . import aggregate_channels as _aggregate_channels
from .table import parse_table

log = logging.getLogger(__name__)


def aggregate(
    sdata: SpatialData,
    aggregate_genes: bool | None = None,
    aggregate_channels: bool = True,
    image_key: str | None = None,
    points_key: str | None = None,
    gene_column: str | None = None,
    shapes_key: str | None = None,
    bins_key: str | None = None,
    expand_radius_ratio: float | None = None,
    min_transcripts: int = 0,
    min_intensity_ratio: float = 0.1,
    no_overlap: bool = False,
    key_added: str | None = "table",
    drop_filtered_cells: bool = True,
):
    """Aggregate gene counts and/or channel intensities over a `SpatialData` object to create an `AnnData` table (saved in `sdata["table"]`).

    !!! info
        The main arguments are `sdata`, `aggregate_genes`, and `aggregate_channels`. The rest of the arguments are optional and will be inferred from the data if not provided.

        - If channels are aggregated and not genes, then `sdata['table'].X` will contain the mean channel intensities per cell.
        - If genes are aggregated and not channels, then `sdata['table'].X` will contain the gene counts per cell.
        - If both genes and channels are aggregated, then `sdata['table'].X` will contain the gene counts per cell and `sdata['table'].obsm['intensities']` will contain the mean channel intensities per cell.

    Args:
        sdata: A `SpatialData` object
        aggregate_genes: Whether to aggregate gene counts. If None, it will be inferred from the data.
        aggregate_channels: Whether to aggregate channel intensities inside cells.
        image_key: Key of `sdata` with the image channels to be averaged. By default, uses the segmentation image.
        points_key: Key of `sdata` with the points dataframe representing the transcripts. Inferred by default.
        gene_column: Key of `sdata[points_key]` with the gene names. Inferred by default.
        shapes_key: Key of `sdata` with the shapes corresponding to the cells boundaries. Inferred by default.
        bins_key: Key of `sdata` with the table corresponding to the bin-by-gene table of gene counts (e.g., for Visium HD data). Inferred by default.
        expand_radius_ratio: Ratio to expand the cells polygons for channels averaging. For instance, a ratio of 0.5 expands the shape radius by 50%. If `None` (default), use 1 if we aggregate bins data, and 0 otherwise.
        min_transcripts: Min number of transcripts to keep a cell.
        min_intensity_ratio: Min ratio of the 90th quantile of the mean channel intensity to keep a cell.
        no_overlap: If `True`, the (expanded) cells will not overlap for channels and bins aggregation. Note that if only affects the aggregation, but the shapes boundaries remain the same. To actually update the shapes, run [`sopa.shapes.remove_overlap`](../misc/#sopa.shapes.remove_overlap) before aggregation (in that case, `no_overlap` will not be needed).
        key_added: Key to save the table in `sdata.tables`. If `None`, it will be `f"{shapes_key}_table"`.
        drop_filtered_cells: If `True`, filtered cells are removed from the returned table. If `False`, all cells are kept and a `passes_filtering` column is added to `table.obs`.
    """
    assert points_key is None or bins_key is None, "Provide either `points_key` or `bins_key`, not both."

    if points_key is None:
        bins_key = bins_key or sdata.attrs.get(SopaAttrs.BINS_TABLE)

    if (bins_key is None) and (aggregate_genes or (aggregate_genes is None and sdata.points)):
        assert sdata.points, (
            "No points in the SpatialData object. You must have points, or set the `bins_key` argument (for VisiumHD-like data)."
        )

        points_key, _ = get_spatial_element(
            sdata.points, key=points_key or sdata.attrs.get(SopaAttrs.TRANSCRIPTS), return_key=True
        )

    aggr = Aggregator(
        sdata,
        image_key=image_key,
        shapes_key=shapes_key,
        bins_key=bins_key,
        points_key=points_key,
        drop_filtered_cells=drop_filtered_cells,
    )

    if key_added is None:
        key_added = f"{aggr.shapes_key}_{SopaKeys.TABLE}"
        log.info(f"key_added is None, saving the table as '{key_added}' by default.")

    aggr.compute_table(
        aggregate_genes=aggregate_genes,
        aggregate_channels=aggregate_channels,
        expand_radius_ratio=expand_radius_ratio,
        min_transcripts=min_transcripts,
        min_intensity_ratio=min_intensity_ratio,
        gene_column=gene_column,
        no_overlap=no_overlap,
        key_added=key_added,
    )


class Aggregator:
    """Perform transcript count and channel averaging over a `SpatialData` object"""

    table: AnnData | None

    def __init__(
        self,
        sdata: SpatialData,
        image_key: str | None = None,
        shapes_key: str | None = None,
        bins_key: str | None = None,
        points_key: str | None = None,
        drop_filtered_cells: bool = True,
    ):
        """
        Args:
            sdata: A `SpatialData` object
            image_key: Key of `sdata` with the image to be averaged. If only one image, this does not have to be provided
            shapes_key: Key of `sdata` with the shapes corresponding to the cells boundaries
            bins_key: Key of `sdata` with the table corresponding to the bins table of gene counts (e.g., for Visium HD data)
            points_key: Key of `sdata` with the points dataframe representing the transcripts
            drop_filtered_cells: If `True`, filtered cells are removed from the returned table. If `False`, all cells are kept and a `passes_filtering` column is added to `table.obs`.
        """
        self.sdata = sdata
        self.bins_key = bins_key
        self.points_key = points_key
        self.shapes_key, self.geo_df = get_boundaries(sdata, return_key=True, key=shapes_key)
        self.table = None
        self.drop_filtered_cells = drop_filtered_cells

        if not sdata.images:
            self.image_key, self.image = "None", None
        else:
            self.image_key, self.image = get_spatial_image(sdata, image_key, return_key=True)

    def already_has_valid_table(self, key_added: str) -> bool:
        if (key_added not in self.sdata.tables) or (self.sdata[key_added].n_obs != len(self.geo_df)):
            return False

        log.info("Found existing table, transcripts are not count again.")
        self.table = self.sdata[key_added]

        return True

    def update_passes_filtering(self, where_filter: np.ndarray, reason: str):
        log.info(f"{where_filter.sum()} cell(s) not passing filtering due to {reason}")

        if SopaKeys.PASSES_FILTERING not in self.table.obs:
            self.table.obs[SopaKeys.PASSES_FILTERING] = True

        self.table.obs[SopaKeys.PASSES_FILTERING] &= ~where_filter

    def filter_cells(self):
        if SopaKeys.PASSES_FILTERING not in self.table.obs:
            return

        if not self.table.obs[SopaKeys.PASSES_FILTERING].any():
            del self.table.obs[SopaKeys.PASSES_FILTERING]
            raise ValueError("All cells were filtered. Try adjusting the filtering parameters.")

        if not self.table.obs[SopaKeys.PASSES_FILTERING].all():
            self.table = self.table[self.table.obs[SopaKeys.PASSES_FILTERING]].copy()

            if self.geo_df is not None:
                self.geo_df = self.geo_df.loc[self.table.obs_names].copy()

        del self.table.obs[SopaKeys.PASSES_FILTERING]

    def compute_table(
        self,
        aggregate_genes: bool | None = None,
        aggregate_channels: bool = True,
        gene_column: str | None = None,
        expand_radius_ratio: float | None = None,
        min_transcripts: int = 0,
        min_intensity_ratio: float = 0,
        no_overlap: bool = False,
        key_added: str = SopaKeys.TABLE,
    ):
        if aggregate_genes is None:
            aggregate_genes = (
                (self.table is not None and isinstance(self.table.X, csr_matrix))
                or gene_column is not None
                or self.bins_key is not None
                or self.points_key is not None
            )

        if aggregate_channels and self.image is None:
            log.warning("No image found to aggregate channels. Use `aggregate_channels=False`")
            aggregate_channels = False

        assert aggregate_genes or aggregate_channels, (
            "At least one of `aggregate_genes` or `aggregate_channels` must be True"
        )

        if aggregate_genes:
            if self.bins_key is not None:
                self.table = aggregate_bins(
                    self.sdata,
                    self.shapes_key,
                    self.bins_key,
                    expand_radius_ratio=1 if expand_radius_ratio is None else expand_radius_ratio,
                    no_overlap=no_overlap,
                )
            elif not self.already_has_valid_table(key_added):
                self.table = count_transcripts(
                    self.sdata, gene_column, shapes_key=self.shapes_key, points_key=self.points_key
                )

            if min_transcripts > 0:
                where_filter = np.asarray(self.table.X.sum(axis=1) < min_transcripts).flatten()
                self.update_passes_filtering(where_filter, f"transcript count < {min_transcripts}")

        if aggregate_channels:
            adata_intensities = _aggregate_channels(
                self.sdata,
                image_key=self.image_key,
                shapes_key=self.shapes_key,
                expand_radius_ratio=expand_radius_ratio or 0,
                no_overlap=no_overlap,
            )

            if aggregate_genes:
                adata_intensities.obs_names = self.table.obs_names
                self.table.obsm[SopaKeys.INTENSITIES_OBSM] = adata_intensities.to_df()
            else:
                self.table = adata_intensities

            if min_intensity_ratio > 0:
                means = adata_intensities.X.mean(axis=1)
                intensity_threshold = min_intensity_ratio * np.quantile(means, 0.9)
                where_filter = means < intensity_threshold
                self.update_passes_filtering(where_filter, f"mean channel intensity < {intensity_threshold:.2f}")

        self.table.uns[SopaKeys.UNS_KEY] = {
            SopaKeys.UNS_HAS_TRANSCRIPTS: aggregate_genes,
            SopaKeys.UNS_HAS_INTENSITIES: aggregate_channels,
        }

        self.table = parse_table(self.table, self.geo_df, self.shapes_key, self.image_key)

        if self.drop_filtered_cells:
            self.filter_cells()

        add_spatial_element(self.sdata, self.shapes_key, self.geo_df)
        add_spatial_element(self.sdata, key_added, self.table)
