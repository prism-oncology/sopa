import os
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
from spatialdata import SpatialData

from ...constants import SopaKeys
from ._custom import custom_staining_based


def stardist(
    sdata: SpatialData,
    model_type: str | None = "2D_versatile_he",
    local_model: str | None = None,
    image_key: str | None = None,
    channels: list[str] | str | None = None,
    min_area: int = 0,
    delete_cache: bool = True,
    recover: bool = False,
    prob_thresh: float = 0.2,
    nms_thresh: float = 0.6,
    clip_limit: float = 0,
    clahe_kernel_size: int | list[int] | None = None,
    gaussian_sigma: float = 0,
    key_added: str = SopaKeys.STARDIST_BOUNDARIES,
    **stardist_eval_kwargs: int,
):
    """Run [Stardist](https://github.com/stardist/stardist) segmentation on a SpatialData object, and add a GeoDataFrame containing the cell boundaries.

    !!! warning "Stardist installation"
        Make sure to install the stardist extra (`pip install 'sopa[stardist]'`) for this method to work.

    Args:
        sdata: A `SpatialData` object
        model_type: Stardist model name.
        local_model: Path to a local StarDist model. If `None`, the pretrained model specified by `model_type` will be used.
        image_key: Name of the image in `sdata` to be used for segmentation.
        channels: One or a list of channel names used for segmentation. None assumes RGB image.
        min_area: Minimum area of a cell to be considered.
        delete_cache: Whether to delete the cache after segmentation.
        recover: If `True`, recover the cache from a failed segmentation, and continue.
        prob_thresh: Stardist `prob_thresh` parameter.
        nms_thresh: Stardist `nms_thresh` parameter.
        clip_limit: Parameter for skimage.exposure.equalize_adapthist (applied before running stardist)
        clahe_kernel_size: Parameter for skimage.exposure.equalize_adapthist (applied before running stardist)
        gaussian_sigma: Parameter for scipy gaussian_filter (applied before running stardist)
        key_added: Name of the shapes element to be added to `sdata`.
        **stardist_eval_kwargs: Kwargs to be provided to `model.predict_instances` (where `model` is a `stardist.models.StarDist2D` object)
    """
    method = stardist_patch(
        model_type=model_type,
        local_model=local_model,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        **stardist_eval_kwargs,
    )

    custom_staining_based(
        sdata,
        method,
        channels=channels,
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


def stardist_patch(
    model_type: str | None = "2D_versatile_he",
    local_model: str | None = None,
    prob_thresh: float = 0.2,
    nms_thresh: float = 0.6,
    channels: list[str] | str | None = None,  # for the CLI to work, as "channels" will be provided
    **stardist_eval_kwargs: int,
) -> Callable:
    try:
        from csbdeep.utils import normalize
    except ImportError:
        raise ImportError("To use stardist, you need its corresponding sopa extra: `pip install 'sopa[stardist]'`.")

    def _(
        patch: np.ndarray,
        model_type: str | None,
        local_model: str | None,
        prob_thresh: float,
        nms_thresh: float,
        **stardist_eval_kwargs,
    ):
        with SuppressPrintsAndWarnings():
            model = load_stardist_model(model_type=model_type, local_model=local_model)

            patch = normalize(patch.transpose(1, 2, 0), clip=True)
            mask, _ = model.predict_instances(
                patch, prob_thresh=prob_thresh, nms_thresh=nms_thresh, **stardist_eval_kwargs
            )

            return mask

    return partial(
        _,
        model_type=model_type,
        local_model=local_model,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        **stardist_eval_kwargs,
    )


def load_stardist_model(model_type: str | None = "2D_versatile_he", local_model: str | None = None):
    """Instantiate a StarDist model. If the pretrained model is not already on disk, it is downloaded.

    Args:
        model_type: Stardist model name.
        local_model: Path to a local StarDist model. If `None`, the pretrained model specified by `model_type` will be used.

    Returns:
        A `stardist.models.StarDist2D` object.
    """
    try:
        from stardist.models import StarDist2D
    except ImportError:
        raise ImportError("To use stardist, you need its corresponding sopa extra: `pip install 'sopa[stardist]'`.")

    with SuppressPrintsAndWarnings():
        if local_model is None:
            return StarDist2D.from_pretrained(model_type)

        model_path = Path(local_model).absolute()
        if not model_path.is_dir():
            raise FileNotFoundError(f"StarDist model directory not found: {model_path}")

        return StarDist2D(None, name=model_path.name, basedir=str(model_path.parent))


class SuppressPrintsAndWarnings:
    def __enter__(self):
        # Suppress stdout (print statements)
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        # Suppress warnings
        self._original_warnings_filter = warnings.filters[:]
        warnings.simplefilter("ignore")

    def __exit__(self, *args, **kwargs):
        # Restore stdout
        sys.stdout.close()
        sys.stdout = self._original_stdout
        # Restore warnings filter
        warnings.filters = self._original_warnings_filter
