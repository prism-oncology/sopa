from functools import partial
from typing import Callable

import numpy as np
from spatialdata import SpatialData

from ...constants import SopaKeys
from ._custom import custom_staining_based


def instanseg(
    sdata: SpatialData,
    model_type: str = "fluorescence_nuclei_and_cells",
    channels: list[str] | str | None = None,
    pixel_size: float | None = None,
    target: str = "cells",
    device: str | None = None,
    image_key: str | None = None,
    min_area: int = 0,
    delete_cache: bool = True,
    recover: bool = False,
    clip_limit: float = 0,
    clahe_kernel_size: int | list[int] | None = None,
    gaussian_sigma: float = 0,
    key_added: str = SopaKeys.INSTANSEG_BOUNDARIES,
    **instanseg_eval_kwargs: int,
):
    """Run [InstanSeg](https://github.com/instanseg/instanseg) segmentation on a SpatialData object, and add a GeoDataFrame containing the cell boundaries.

    !!! warning "InstanSeg installation"
        Make sure to install the instanseg extra (`pip install 'sopa[instanseg]'`) for this method to work.

    !!! info "Pixel size parameter"
        InstanSeg models are trained at a specific resolution (in microns per pixel), e.g. `0.5` for `fluorescence_nuclei_and_cells`. Provide `pixel_size` (the resolution of your image) so InstanSeg can internally rescale the image for accurate segmentation. This is not optional in practice: if the cells are far from the size the model expects, the segmentation returns no cell at all.

    Args:
        sdata: A `SpatialData` object
        model_type: Name of the pretrained InstanSeg model (e.g., `"fluorescence_nuclei_and_cells"` or `"brightfield_nuclei"`), or path to a local model.
        channels: Name of the channel(s) to be used for segmentation (or list of channel names).
        pixel_size: Resolution of the image, in microns per pixel. If `None`, the image is not rescaled, and the segmentation may return no cell at all if the objects are not already at the scale expected by the model.
        target: For models that support both nuclei and whole-cell segmentation, whether to output `"cells"` or `"nuclei"` boundaries. Ignored for models supporting only one of the two.
        device: Device used for inference (e.g., `"cuda"`, `"mps"`, or `"cpu"`). By default, it is chosen automatically.
        image_key: Name of the image in `sdata` to be used for segmentation.
        min_area: Minimum area of a cell to be considered.
        delete_cache: Whether to delete the cache after segmentation.
        recover: If `True`, recover the cache from a failed segmentation, and continue.
        clip_limit: Parameter for skimage.exposure.equalize_adapthist (applied before running instanseg)
        clahe_kernel_size: Parameter for skimage.exposure.equalize_adapthist (applied before running instanseg)
        gaussian_sigma: Parameter for scipy gaussian_filter (applied before running instanseg)
        key_added: Name of the shapes element to be added to `sdata`.
        **instanseg_eval_kwargs: Kwargs to be provided to `model.eval_small_image` (where `model` is an `instanseg.InstanSeg` object)
    """
    method = instanseg_patch(
        model_type=model_type,
        pixel_size=pixel_size,
        target=target,
        device=device,
        **instanseg_eval_kwargs,
    )

    custom_staining_based(
        sdata,
        method,
        channels,
        image_key=image_key,
        min_area=min_area,
        delete_cache=delete_cache,
        recover=recover,
        clip_limit=clip_limit,
        clahe_kernel_size=clahe_kernel_size,
        gaussian_sigma=gaussian_sigma,
        cache_dir_name=key_added,
        key_added=key_added,
    )


def instanseg_patch(
    model_type: str = "fluorescence_nuclei_and_cells",
    pixel_size: float | None = None,
    target: str = "cells",
    device: str | None = None,
    channels: list[str] | str | None = None,  # for the CLI to work, as "channels" will be provided
    **instanseg_eval_kwargs: int,
) -> Callable:
    """Creation of a callable that runs InstanSeg segmentation on a patch

    Args:
        model_type: Name of the pretrained InstanSeg model, or path to a local model
        pixel_size: Resolution of the image, in microns per pixel
        target: Whether to output `"cells"` or `"nuclei"` boundaries (for models supporting both)
        device: Device used for inference (e.g., `"cuda"`, `"mps"`, or `"cpu"`)
        **instanseg_eval_kwargs: Kwargs to be provided to `model.eval_small_image` (where `model` is an `instanseg.InstanSeg` object)

    Returns:
        A `callable` whose input is an image of shape `(C, Y, X)` and output is a cell mask of shape `(Y, X)`. Each mask value `>0` represent a unique cell ID
    """
    try:
        import instanseg  # noqa: F401
    except ImportError:
        raise ImportError("To use instanseg, you need its corresponding sopa extra: `pip install 'sopa[instanseg]'`.")

    def _(
        patch: np.ndarray,
        model_type: str,
        pixel_size: float | None,
        target: str,
        device: str | None,
        **instanseg_eval_kwargs: int,
    ):
        model = load_instanseg_model(model_type=model_type, device=device)

        instances = model.eval_small_image(
            patch,
            pixel_size=pixel_size,
            target=target,
            return_image_tensor=False,
            **instanseg_eval_kwargs,
        )

        return instances[0, 0].numpy().astype(np.int32)

    return partial(
        _,
        model_type=model_type,
        pixel_size=pixel_size,
        target=target,
        device=device,
        **instanseg_eval_kwargs,
    )


def load_instanseg_model(model_type: str = "fluorescence_nuclei_and_cells", device: str | None = None):
    """Instantiate an InstanSeg model. If the pretrained model is not already on disk, it is downloaded.

    Args:
        model_type: Name of the pretrained InstanSeg model, or path to a local model.
        device: Device used for inference (e.g., `"cuda"`, `"mps"`, or `"cpu"`). By default, it is chosen automatically.

    Returns:
        An `instanseg.InstanSeg` object.
    """
    try:
        from instanseg import InstanSeg
    except ImportError:
        raise ImportError("To use instanseg, you need its corresponding sopa extra: `pip install 'sopa[instanseg]'`.")

    return InstanSeg(model_type, device=device, verbosity=0)
