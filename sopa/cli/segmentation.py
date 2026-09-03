import ast
from collections.abc import Iterable
from typing import Any

import typer

from .utils import SDATA_HELPER, _log_whether_to_resolve

app_segmentation = typer.Typer()


@app_segmentation.command()
def cellpose(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    diameter: int = typer.Option(help="Cellpose diameter parameter"),
    channels: list[str] = typer.Option(
        help="Names of the channels used for Cellpose. If one channel, then provide just a nucleus channel. If two channels, this is the cytoplasm and then the nucleus channel"
    ),
    flow_threshold: float = typer.Option(2, help="Cellpose `flow_threshold` parameter"),
    cellprob_threshold: float = typer.Option(-6, help="Cellpose `cellprob_threshold` parameter"),
    model_type: str = typer.Option("cyto3", help="Name of the cellpose model if using `cellpose<4.0.0`."),
    pretrained_model: str = typer.Option(
        None,
        help="Name of pretrained model (e.g., `cpsam`, if using `cellpose>=4.0.0`), or path to the pretrained model to be loaded, or `None`.",
    ),
    gpu: bool = typer.Option(False, help="Whether to use the GPU (recommended for cellpose>=4.0.0)"),
    min_area: int = typer.Option(0, help="Minimum area (in pixels^2) for a cell to be considered as valid"),
    clip_limit: float = typer.Option(
        0.2,
        help="Parameter for skimage.exposure.equalize_adapthist (applied before running cellpose)",
    ),
    clahe_kernel_size: int = typer.Option(
        None,
        help="Parameter for skimage.exposure.equalize_adapthist (applied before running cellpose)",
    ),
    gaussian_sigma: float = typer.Option(
        1, help="Parameter for scipy gaussian_filter (applied before running cellpose)"
    ),
    patch_index: int = typer.Option(
        default=None,
        help="Index of the patch on which cellpose should be run. NB: the number of patches is `len(sdata['image_patches'])`",
    ),
    cache_dir_name: str = typer.Option(
        default=None,
        help="Name of the temporary cellpose directory inside which we will store each individual patch segmentation. By default, uses the `cellpose_boundaries` directory",
    ),
    method_kwargs: str = typer.Option(
        {},
        callback=ast.literal_eval,
        help="Kwargs for the cellpose method builder. This should be a dictionnary, in inline string format.",
    ),
):
    """Perform cellpose segmentation. This can be done on all patches directly, or on one individual patch.

    Usage:

        - [On one patch] Use this mode to run patches in parallel. Provide `--patch-index` to run one patch, and execute all patches in a parallel manner (you need to define your own parallelization, else, use the Snakemake pipeline).

        - [On all patches at once] For small images, you can run the segmentation method sequentially (`--patch-index` is not needed)
    """
    from sopa.constants import SopaKeys

    channels = channels if isinstance(channels, list) else [channels]

    _run_staining_segmentation(
        sdata_path,
        "cellpose_patch",
        SopaKeys.CELLPOSE_BOUNDARIES,
        channels,
        min_area,
        clip_limit,
        clahe_kernel_size,
        gaussian_sigma,
        patch_index,
        cache_dir_name,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        model_type=model_type,
        pretrained_model=pretrained_model,
        gpu=gpu,
        **method_kwargs,
    )


@app_segmentation.command()
def stardist(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    model_type: str = typer.Option("2D_versatile_he", help="Name of the stardist model"),
    local_model: str | None = typer.Option(
        None,
        help="Path to the local stardist model to be loaded. If not provided, the pretrained model specified by `model_type` will be used.",
    ),
    prob_thresh: float = typer.Option(0.2, help="Stardist `prob_thresh` parameter."),
    nms_thresh: float = typer.Option(0.6, help="Stardist `nms_thresh` parameter."),
    channels: list[str] = typer.Option(None, help="Names of the channels used for segmentation."),
    min_area: int = typer.Option(0, help="Minimum area (in pixels^2) for a cell to be considered as valid"),
    clip_limit: float = typer.Option(
        0.2,
        help="Parameter for skimage.exposure.equalize_adapthist (applied before running the segmentation method)",
    ),
    clahe_kernel_size: int = typer.Option(
        None,
        help="Parameter for skimage.exposure.equalize_adapthist (applied before running stardist)",
    ),
    gaussian_sigma: float = typer.Option(
        1,
        help="Parameter for scipy gaussian_filter (applied before running the segmentation method)",
    ),
    patch_index: int = typer.Option(
        default=None,
        help="Index of the patch on which the segmentation method should be run. NB: the number of patches is `len(sdata['image_patches'])`",
    ),
    cache_dir_name: str = typer.Option(
        default=None,
        help="Name of the temporary the segmentation method directory inside which we will store each individual patch segmentation. By default, uses the `stardist_boundaries` directory",
    ),
    method_kwargs: str = typer.Option(
        {},
        callback=ast.literal_eval,
        help="Kwargs for the method. This should be a dictionnary, in inline string format.",
    ),
):
    """Perform Stardist segmentation. This can be done on all patches directly, or on one individual patch."""
    from sopa.constants import SopaKeys

    _run_staining_segmentation(
        sdata_path,
        "stardist_patch",
        SopaKeys.STARDIST_BOUNDARIES,
        channels,
        min_area,
        clip_limit,
        clahe_kernel_size,
        gaussian_sigma,
        patch_index,
        cache_dir_name,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        model_type=model_type,
        local_model=local_model,
        **method_kwargs,
    )


@app_segmentation.command()
def instanseg(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    model_type: str = typer.Option(
        "fluorescence_nuclei_and_cells", help="Name of the pretrained InstanSeg model, or path to a local model."
    ),
    pixel_size: float = typer.Option(
        None,
        help="Resolution of the image, in microns per pixel. If not provided, the image is not rescaled, and no cell may be segmented at all if they are not at the scale expected by the model.",
    ),
    target: str = typer.Option(
        "cells",
        help="Whether to output 'cells' or 'nuclei' boundaries (for models supporting both).",
    ),
    device: str = typer.Option(None, help="Device used for inference (e.g., 'cuda', 'mps', or 'cpu')."),
    channels: list[str] = typer.Option(None, help="Names of the channels used for segmentation."),
    min_area: int = typer.Option(0, help="Minimum area (in pixels^2) for a cell to be considered as valid"),
    clip_limit: float = typer.Option(
        0,
        help="Parameter for skimage.exposure.equalize_adapthist (applied before running the segmentation method)",
    ),
    clahe_kernel_size: int = typer.Option(
        None,
        help="Parameter for skimage.exposure.equalize_adapthist (applied before running instanseg)",
    ),
    gaussian_sigma: float = typer.Option(
        0,
        help="Parameter for scipy gaussian_filter (applied before running the segmentation method)",
    ),
    patch_index: int = typer.Option(
        default=None,
        help="Index of the patch on which the segmentation method should be run. NB: the number of patches is `len(sdata['image_patches'])`",
    ),
    cache_dir_name: str = typer.Option(
        default=None,
        help="Name of the temporary the segmentation method directory inside which we will store each individual patch segmentation. By default, uses the `instanseg_boundaries` directory",
    ),
    method_kwargs: str = typer.Option(
        {},
        callback=ast.literal_eval,
        help="Kwargs for the method. This should be a dictionnary, in inline string format.",
    ),
):
    """Perform InstanSeg segmentation. This can be done on all patches directly, or on one individual patch."""
    from sopa.constants import SopaKeys

    _run_staining_segmentation(
        sdata_path,
        "instanseg_patch",
        SopaKeys.INSTANSEG_BOUNDARIES,
        channels,
        min_area,
        clip_limit,
        clahe_kernel_size,
        gaussian_sigma,
        patch_index,
        cache_dir_name,
        model_type=model_type,
        pixel_size=pixel_size,
        target=target,
        device=device,
        **method_kwargs,
    )


@app_segmentation.command()
def generic_staining(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    method_name: str = typer.Option(
        help="Name of the segmentation method builder to use. The corresponding function (`sopa.segmentation.methods.<method_name>`) will be used, and the kwargs below will be used to instantiate the method."
    ),
    method_kwargs: str = typer.Option(
        {},
        callback=ast.literal_eval,
        help="Kwargs for the method. This should be a dictionnary, in inline string format.",
    ),
    channels: list[str] = typer.Option(None, help="Names of the channels used for segmentation."),
    min_area: int = typer.Option(0, help="Minimum area (in pixels^2) for a cell to be considered as valid"),
    clip_limit: float = typer.Option(
        0.2,
        help="Parameter for skimage.exposure.equalize_adapthist (applied before running the segmentation method)",
    ),
    clahe_kernel_size: int = typer.Option(
        None,
        help="Parameter for skimage.exposure.equalize_adapthist (applied before running segmentation)",
    ),
    gaussian_sigma: float = typer.Option(
        1,
        help="Parameter for scipy gaussian_filter (applied before running the segmentation method)",
    ),
    patch_index: int = typer.Option(
        default=None,
        help="Index of the patch on which the segmentation method should be run. NB: the number of patches is `len(sdata['image_patches'])`",
    ),
    cache_dir_name: str = typer.Option(
        default=None,
        help="Name of the temporary the segmentation method directory inside which we will store each individual patch segmentation. By default, uses the `<method_name>` directory",
    ),
):
    """Perform generic staining-based segmentation. This can be done on all patches directly, or on one individual patch.

    Usage:
        First, define a new segmentation method, and write it under `sopa.segmentation.methods`. It should correspond to a function that is a "callable builder", i.e. kwargs will be provided to this function, and it will return a callable that will be applied on patches.

        As for Cellpose, two modes ara available:

        - [On one patch] Use this mode to run patches in parallel. Provide `--patch-index` to run one patch, and execute all patches in a parallel manner (you need to define your own parallelization, else, use the Snakemake pipeline).

        - [On all patches at once] For small images, you can run the segmentation method sequentially (`--patch-index` is not needed)
    """
    from sopa.segmentation import methods

    assert hasattr(methods, method_name), (
        f"'{method_name}' is not a valid method builder under `sopa.segmentation.methods`"
    )

    _run_staining_segmentation(
        sdata_path,
        method_name,
        method_name,
        channels,
        min_area,
        clip_limit,
        clahe_kernel_size,
        gaussian_sigma,
        patch_index,
        cache_dir_name,
        **method_kwargs,
    )


def _run_staining_segmentation(
    sdata_path: str,
    method_name: str,
    key_added: str,
    channels: list[str] | None,
    min_area: float,
    clip_limit: float,
    clahe_kernel_size: int | Iterable[int] | None,
    gaussian_sigma: float,
    patch_index: int | None,
    cache_dir_name: str | None,
    **method_kwargs: Any,
):
    from sopa.io.standardize import read_zarr_standardized
    from sopa.segmentation import StainingSegmentation, custom_staining_based, methods

    from .utils import _default_boundary_dir

    sdata = read_zarr_standardized(sdata_path)

    if channels is not None:
        method_kwargs["channels"] = channels

    method = getattr(methods, method_name)(**method_kwargs)

    delete_cache = cache_dir_name is None
    if cache_dir_name is None:
        cache_dir_name = key_added

    if patch_index is None:
        custom_staining_based(
            sdata,
            method=method,
            channels=channels,
            min_area=min_area,
            clip_limit=clip_limit,
            clahe_kernel_size=clahe_kernel_size,
            gaussian_sigma=gaussian_sigma,
            key_added=key_added,
            cache_dir_name=cache_dir_name,
            delete_cache=delete_cache,
        )
        _log_whether_to_resolve(patch_index, delete_cache=delete_cache)
        return

    segmentation = StainingSegmentation(
        sdata,
        method,
        channels,
        min_area=min_area,
        clip_limit=clip_limit,
        clahe_kernel_size=clahe_kernel_size,
        gaussian_sigma=gaussian_sigma,
    )

    patch_dir = _default_boundary_dir(sdata_path, cache_dir_name)

    segmentation.write_patch_cells(patch_dir, patch_index)


@app_segmentation.command()
def comseg(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    config: str = typer.Option(
        {},
        callback=ast.literal_eval,
        help="""Comseg config. This should be a dictionnary in inline string format, wrapped inside quotes, e.g. '{"example": 2}'""",
    ),
    patch_index: int = typer.Option(
        default=None,
        help="Index of the patch on which the segmentation method should be run.",
    ),
    min_area: float = typer.Option(default=0, help="Minimum area (in micron^2) for a cell to be considered as valid"),
):
    """Perform ComSeg segmentation. This can be done on all patches directly, or on one individual patch."""
    from sopa.io.standardize import read_zarr_standardized
    from sopa.segmentation.methods import comseg

    sdata = read_zarr_standardized(sdata_path)

    comseg(sdata, config=config, min_area=min_area, patch_index=patch_index)

    _log_whether_to_resolve(patch_index)


@app_segmentation.command()
def baysor(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    config: str = typer.Option(
        {},
        callback=ast.literal_eval,
        help="""Baysor config. This should be a dictionnary in inline string format, wrapped inside single quotes, e.g. '{"example": 2}'""",
    ),
    patch_index: int = typer.Option(
        default=None,
        help="Index of the patch on which the segmentation method should be run. By default, run on all patches.",
    ),
    min_area: float = typer.Option(default=0, help="Minimum area (in micron^2) for a cell to be considered as valid"),
    scale: float = typer.Option(default=None, help="Baysor scale parameter (for config inference)"),
):
    """Perform Baysor segmentation. This can be done on all patches directly, or on one individual patch."""
    import sys
    from subprocess import CalledProcessError

    from sopa.io.standardize import read_zarr_standardized
    from sopa.segmentation.methods import baysor

    sdata = read_zarr_standardized(sdata_path)

    try:
        baysor(sdata, config=config, min_area=min_area, patch_index=patch_index, scale=scale)
    except CalledProcessError as e:
        sys.exit(e.returncode)

    _log_whether_to_resolve(patch_index)


@app_segmentation.command()
def proseg(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    command_line_suffix: str = typer.Option(
        "",
        help="String suffix to add to the proseg command line. This can be used to add extra parameters to the proseg command line.",
    ),
    infer_presets: bool = typer.Option(
        True,
        help="Whether to infer the proseg presets based on the columns of the transcripts dataframe.",
    ),
    prior_shapes_key: str | None = typer.Option(
        None,
        help="**Only for Visium HD data.** Key of `sdata` containing the prior cell boundaries. If `'auto'`, use the latest performed segmentation (e.g., stardist or the 10X Genomics segmentation).",
    ),
    bins_key: str = typer.Option(
        None,
        help="**Only for Visium HD data.** Key of `sdata` with the table corresponding to the bin-by-gene table of gene counts (e.g., for Visium HD data). Inferred by default.",
    ),
):
    """Perform Proseg segmentation. This needs to be done on a single
    patch as proseg is fast enough and doesn't require parallelization."""
    import sys
    from subprocess import CalledProcessError

    from sopa.io.standardize import read_zarr_standardized
    from sopa.segmentation.methods import proseg

    sdata = read_zarr_standardized(sdata_path)

    try:
        proseg(
            sdata,
            command_line_suffix=command_line_suffix,
            infer_presets=infer_presets,
            prior_shapes_key=prior_shapes_key,
            bins_key=bins_key,
        )
    except CalledProcessError as e:
        sys.exit(e.returncode)

    _log_whether_to_resolve(None)


@app_segmentation.command()
def tissue(
    sdata_path: str = typer.Argument(help=SDATA_HELPER),
    image_key: str = typer.Option(
        default=None,
        help="Name of the image key to use for tissue segmentation.",
    ),
    level: int = typer.Option(
        default=-1,
        help="Level of the multiscale image to use for tissue segmentation.",
    ),
    mode: str = typer.Option(
        default=None,
        help="Mode for the tissue segmentation: 'staining' or 'saturation' (for H&E images).",
    ),
    kwargs: str = typer.Option(
        {},
        callback=ast.literal_eval,
        help="Kwargs for `sopa.segmentation.tissue`. This should be a dictionnary, in inline string format.",
    ),
):
    """Perform tissue segmentation. This can be done only on objects with H&E staining."""
    import sopa
    from sopa.io.standardize import read_zarr_standardized

    sdata = read_zarr_standardized(sdata_path)

    sopa.segmentation.tissue(sdata, image_key=image_key, level=level, mode=mode, **kwargs)
