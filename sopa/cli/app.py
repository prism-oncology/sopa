import ast
import logging

import typer

from .annotate import app_annotate
from .download import app_download
from .explorer import app_explorer
from .patchify import app_patchify
from .resolve import app_resolve
from .segmentation import app_segmentation
from .utils import SDATA_HELPER

log = logging.getLogger(__name__)

app = typer.Typer()
app.add_typer(
    app_annotate,
    name="annotate",
    help="Perform cell-type annotation (based on transcripts and/or channel intensities)",
)
app.add_typer(
    app_explorer,
    name="explorer",
    help="Convertion to the Xenium Explorer's inputs, and image alignment",
)
app.add_typer(
    app_segmentation,
    name="segmentation",
    help="Perform cell segmentation on patches. NB: for `baysor`, use directly the `baysor` command line.",
)
app.add_typer(app_resolve, name="resolve", help="Resolve the segmentation conflicts over patches overlaps")
app.add_typer(
    app_download,
    name="download",
    help="Download the model weights used by the segmentation methods, so that they are cached before running the segmentation",
)
app.add_typer(
    app_patchify,
    name="patchify",
    help="Create patches with overlaps. Afterwards, segmentation will be run on each patch",
)


@app.command()
def convert(
    data_path: str = typer.Argument(
        help="Path to one data sample (most of the time, this corresponds to a directory with images files and eventually a transcript file)"
    ),
    technology: str = typer.Option(
        None,
        help="Name of the technology used to collected the data (`xenium`/`merfish`/`cosmx`/`phenocycler`/`macsima`/`hyperion`)",
    ),
    sdata_path: str = typer.Option(
        None,
        help="Optional path to write the SpatialData object. If not provided, will write to the `{data_path}.zarr` directory",
    ),
    config_path: str = typer.Option(
        None,
        help="Path to the snakemake config. This can be useful in order not to provide the `--technology` and the `--kwargs` arguments",
    ),
    overwrite: bool = typer.Option(
        False, help="Whether to overwrite the existing SpatialData object if already existing"
    ),
    kwargs: str = typer.Option(
        default={},
        callback=ast.literal_eval,
        help="Dictionary provided to the reader function as kwargs",
    ),
):
    """Read any technology data as a SpatialData object and save it as a `.zarr` directory.

    Either `--technology` or `--config-path` has to be provided."""
    import shutil
    from pathlib import Path

    from spatialdata import SpatialData

    from sopa import io
    from sopa.constants import SopaFiles, SopaKeys

    sdata_path: Path = Path(data_path).with_suffix(".zarr") if sdata_path is None else Path(sdata_path)

    if sdata_path.exists():
        if overwrite:
            cache_dir = sdata_path.resolve() / SopaFiles.SOPA_CACHE_DIR
            if cache_dir.exists() and cache_dir.is_dir():
                shutil.rmtree(cache_dir)
            if not any(sdata_path.iterdir()):
                sdata_path.rmdir()  # remove empty directory
        else:
            assert not any(sdata_path.iterdir()), (
                f"Zarr directory {sdata_path} already exists. Sopa will not continue to avoid overwritting files. Use the `--overwrite` flag to overwrite it."
            )
            sdata_path.rmdir()  # remove empty directory

    assert technology is not None or config_path is not None, "Provide the argument `--technology` or `--config-path`"

    if config_path is not None:
        assert not kwargs, "Provide either a path to a config, or some kwargs, but not both"
        with open(config_path) as f:
            import yaml

            config = yaml.safe_load(f)

        technology = config["read"]["technology"]
        kwargs = config["read"]["kwargs"]

    assert hasattr(io, technology), (
        f"Technology {technology} unknown. Currently available: xenium, merscope, visium_hd, cosmx, phenocycler, hyperion, macsima"
    )

    sdata: SpatialData = getattr(io, technology)(data_path, **kwargs)

    io.standardize.sanity_check(sdata)

    assert SopaKeys.TABLE not in sdata.tables, (
        f"sdata.tables['{SopaKeys.TABLE}'] exists. Delete it you want to use sopa, to avoid conflicts with future table generation"
    )

    log.info(f"Writing the following spatialdata object to {sdata_path}:\n{sdata}")

    sdata.write(sdata_path, overwrite=overwrite)


@app.command()
def aggregate(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    aggregate_genes: bool = typer.Option(None, help="Whether to aggregate the genes (counts) inside each cell"),
    aggregate_channels: bool = typer.Option(
        False, help="Whether to aggregate the channels (intensity) inside each cell"
    ),
    expand_radius_ratio: float = typer.Option(
        default=None,
        help="Cells polygons will be expanded by `expand_radius_ratio * mean_radius` for channels averaging **only**. This help better aggregate boundary stainings",
    ),
    min_transcripts: int = typer.Option(0, help="Cells with less transcript than this integer will be filtered"),
    min_intensity_ratio: float = typer.Option(
        0,
        help="Cells whose mean channel intensity is less than `min_intensity_ratio * quantile_90` will be filtered",
    ),
    image_key: str = typer.Option(
        None,
        help="Optional image key of the SpatialData object. By default, considers the only one image. It can be useful if another image is added later on",
    ),
    gene_column: str = typer.Option(
        None, help="Optional column of the transcript dataframe to be used as gene names. Inferred by default."
    ),
    method_name: str = typer.Option(
        None, help="If segmentation was performed with a generic method, this is the name of the method used."
    ),
):
    """Create an `anndata` table containing the transcript count and/or the channel intensities per cell"""
    import sopa
    from sopa.io.standardize import read_zarr_standardized

    if gene_column:
        assert aggregate_genes is not False
        aggregate_genes = True

    sdata = read_zarr_standardized(sdata_path)

    sopa.aggregate(
        sdata,
        aggregate_genes=aggregate_genes,
        aggregate_channels=aggregate_channels,
        image_key=image_key,
        shapes_key=method_name,
        gene_column=gene_column,
        min_transcripts=min_transcripts,
        expand_radius_ratio=expand_radius_ratio,
        min_intensity_ratio=min_intensity_ratio,
    )


@app.command()
def scanpy_preprocess(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    table_key: str = typer.Option("table", help="Key of the table in the `SpatialData` object to be preprocessed"),
    resolution: float = typer.Option(1.0, help="Resolution parameter for the leiden clustering"),
    check_counts: bool = typer.Option(True, help="Whether to check that adata.X contains counts"),
    hvg: bool = typer.Option(
        False, help="Whether to compute highly variable genes before computing the UMAP and clustering"
    ),
):
    """Optional scanpy table preprocessing (log1p, UMAP, leiden clustering) after aggregation/annotation."""
    import scanpy as sc
    from anndata import AnnData

    import sopa
    from sopa.io.standardize import read_zarr_standardized

    sdata = read_zarr_standardized(sdata_path)

    adata: AnnData = sdata.tables[table_key]

    if check_counts:
        assert adata.X.max() > 10, (
            f"adata.X looks already log-normalized (max={adata.X.max()}). If you are sure it contains counts, use `--no-check-counts`"
        )

    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    if hvg:
        sc.pp.highly_variable_genes(adata)

        if adata.var["highly_variable"].sum() <= 50:
            log.warning("Less than 50 HVG found. They will not be used for the UMAP and leiden clustering.")
            highly_variable = adata.var["highly_variable"]
            del adata.var["highly_variable"]  # else sc.pp.neighbors will crash

    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph")

    if hvg and "highly_variable" not in adata.var:
        adata.var["highly_variable"] = highly_variable  # put it back

    sopa.utils.add_spatial_element(sdata, table_key, adata)

    (sopa.utils.get_cache_dir(sdata) / "scanpy_preprocess").touch()


@app.command()
def report(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    path: str = typer.Argument(help="Path to the HTML report"),
    table_key: str = typer.Option(
        "table", help="Key of the table in the `SpatialData` object to be used for the report"
    ),
):
    """Create a HTML report of the pipeline run and some quality controls"""
    from sopa.io.report import write_report
    from sopa.io.standardize import read_zarr_standardized

    sdata = read_zarr_standardized(sdata_path)

    write_report(path, sdata, table_key=table_key)


def version_callback(value: bool):
    if value:
        from sopa import __version__

        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def common(
    ctx: typer.Context,
    version: bool = typer.Option(None, "--version", callback=version_callback, help="Show the Sopa version and exit."),
):
    pass
