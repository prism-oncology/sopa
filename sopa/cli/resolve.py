import typer

from .utils import SDATA_HELPER

app_resolve = typer.Typer()


@app_resolve.command()
def cellpose(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    cache_dir_name: list[str] = typer.Option(
        [],
        help="Name of the directories containing the cellpose segmentation on patches (or multiple directories if using multi-step segmentation). By default, uses the `cellpose_boundaries` directory",
    ),
):
    """Resolve patches conflicts after cellpose segmentation"""
    from sopa.constants import SopaKeys

    from .utils import _default_boundary_dir

    if not len(cache_dir_name):
        cache_dir_name = [SopaKeys.CELLPOSE_BOUNDARIES]

    patch_dir = [_default_boundary_dir(sdata_path, name) for name in cache_dir_name]

    _resolve_generic(sdata_path, patch_dir, SopaKeys.CELLPOSE_BOUNDARIES)


@app_resolve.command()
def stardist(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    cache_dir_name: list[str] = typer.Option(
        [],
        help="Name of the directories containing the stardist segmentation on patches (or multiple directories if using multi-step segmentation). By default, uses the `stardist_boundaries` directory",
    ),
):
    """Resolve patches conflicts after stardist segmentation"""
    from sopa.constants import SopaKeys

    from .utils import _default_boundary_dir

    if not len(cache_dir_name):
        cache_dir_name = [SopaKeys.STARDIST_BOUNDARIES]

    patch_dir = [_default_boundary_dir(sdata_path, name) for name in cache_dir_name]

    _resolve_generic(sdata_path, patch_dir, SopaKeys.STARDIST_BOUNDARIES)


@app_resolve.command()
def instanseg(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    cache_dir_name: list[str] = typer.Option(
        [],
        help="Name of the directories containing the instanseg segmentation on patches (or multiple directories if using multi-step segmentation). By default, uses the `instanseg_boundaries` directory",
    ),
):
    """Resolve patches conflicts after instanseg segmentation"""
    from sopa.constants import SopaKeys

    from .utils import _default_boundary_dir

    if not len(cache_dir_name):
        cache_dir_name = [SopaKeys.INSTANSEG_BOUNDARIES]

    patch_dir = [_default_boundary_dir(sdata_path, name) for name in cache_dir_name]

    _resolve_generic(sdata_path, patch_dir, SopaKeys.INSTANSEG_BOUNDARIES)


@app_resolve.command()
def generic(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    method_name: str = typer.Option(
        help="Name of the method used during segmentation. This is also the key correspnding to the boundaries in `sdata.shapes`"
    ),
    cache_dir_name: list[str] = typer.Option(
        [],
        help="Name of the directories containing the generic segmentation on patches (or multiple directories if using multi-step segmentation). By default, uses the `<method_name>` directory",
    ),
):
    """Resolve patches conflicts after generic segmentation"""
    from .utils import _default_boundary_dir

    if not len(cache_dir_name):
        cache_dir_name = [method_name]

    patch_dir = [_default_boundary_dir(sdata_path, name) for name in cache_dir_name]

    _resolve_generic(sdata_path, patch_dir, method_name)


def _resolve_generic(sdata_path: str, patch_dirs: list[str], key_added: str):
    from sopa.io.standardize import read_zarr_standardized
    from sopa.segmentation import StainingSegmentation, solve_conflicts
    from sopa.utils import get_spatial_image

    sdata = read_zarr_standardized(sdata_path)

    assert len(patch_dirs) > 0, "No patch directory was provided, cannot load cells"

    image_key, _ = get_spatial_image(sdata, return_key=True)

    cells = StainingSegmentation.read_patches_cells(patch_dirs)
    cells = solve_conflicts(cells)

    StainingSegmentation.add_shapes(sdata, cells, image_key, key_added)


@app_resolve.command()
def baysor(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    gene_column: str = typer.Option(None, help="Column of the transcripts dataframe containing the genes names"),
    min_area: float = typer.Option(0, help="Cells with an area less than this value (in microns^2) will be filtered"),
):
    """Resolve patches conflicts after baysor segmentation."""
    import sopa
    from sopa.constants import SopaKeys
    from sopa.io.standardize import read_zarr_standardized
    from sopa.segmentation.methods.utils import resolve

    sdata = read_zarr_standardized(sdata_path)

    patches_dirs = sopa.utils.get_transcripts_patches_dirs(sdata)

    resolve(
        sdata,
        patches_dirs,
        gene_column,
        min_area=min_area,
        key_added=SopaKeys.BAYSOR_BOUNDARIES,
    )


@app_resolve.command()
def comseg(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    gene_column: str = typer.Option(None, help="Column of the transcripts dataframe containing the genes names"),
    min_area: float = typer.Option(0, help="Cells with an area less than this value (in microns^2) will be filtered"),
):
    """Resolve patches conflicts after comseg segmentation."""
    import sopa
    from sopa.constants import SopaKeys
    from sopa.io.standardize import read_zarr_standardized
    from sopa.segmentation.methods.utils import resolve

    sdata = read_zarr_standardized(sdata_path)

    patches_dirs = sopa.utils.get_transcripts_patches_dirs(sdata)

    resolve(
        sdata,
        patches_dirs,
        gene_column,
        min_area=min_area,
        key_added=SopaKeys.COMSEG_BOUNDARIES,
    )
