import shutil
from collections.abc import Callable

from spatialdata import SpatialData

from ...constants import SopaKeys
from ...utils import get_cache_dir, set_boundaries_attrs
from .. import StainingSegmentation, solve_conflicts


def custom_staining_based(
    sdata: SpatialData,
    method: Callable,
    channels: list[str] | str | None,
    image_key: str | None = None,
    min_area: float = 0,
    delete_cache: bool = True,
    recover: bool = False,
    clip_limit: float = 0.2,
    clahe_kernel_size: int | list[int] | None = None,
    gaussian_sigma: float = 1,
    cache_dir_name: str = SopaKeys.CUSTOM_BOUNDARIES,
    key_added: str = SopaKeys.CUSTOM_BOUNDARIES,
    min_patch_size: int = 10,
):
    """Run a generic staining-based segmentation model, and add a GeoDataFrame containing the cell boundaries.

    Args:
        sdata: A `SpatialData` object.
        method: A segmentation `callable` whose input is an image of shape `(C, Y, X)` and output is a cell mask of shape `(Y, X)`. Each mask value `>0` represent a unique cell ID. The `C` channels is determined by the `channels` argument.
        channels: Name of the channels to be used for segmentation (or list of channel names).
        image_key: Name of the image in `sdata` to be used for segmentation.
        min_area: Minimum area of a cell to be considered.
        delete_cache: Whether to delete the cache after segmentation.
        recover: If `True`, recover the cache from a failed segmentation, and continue.
        clip_limit: Parameter for skimage.exposure.equalize_adapthist (applied before running segmentation)
        clahe_kernel_size: Parameter for skimage.exposure.equalize_adapthist (applied before running segmentation)
        gaussian_sigma: Parameter for scipy gaussian_filter (applied before running segmentation)
        cache_dir_name: Name of the cache directory.
        key_added: Name of the key to be added to `sdata.shapes`.
        min_patch_size: Minimum patch size (in pixels) for both width and height. Patches smaller than this will be skipped to avoid segmentation errors.
    """
    temp_dir = get_cache_dir(sdata) / cache_dir_name

    segmentation = StainingSegmentation(
        sdata,
        method,
        channels,
        min_area=min_area,
        image_key=image_key,
        clip_limit=clip_limit,
        clahe_kernel_size=clahe_kernel_size,
        gaussian_sigma=gaussian_sigma,
        min_patch_size=min_patch_size,
    )
    segmentation.write_patches_cells(temp_dir, recover=recover)

    cells = StainingSegmentation.read_patches_cells(temp_dir)
    cells = solve_conflicts(cells)

    StainingSegmentation.add_shapes(sdata, cells, image_key=segmentation.image_key, key_added=key_added)

    set_boundaries_attrs(sdata, key_added)

    if delete_cache:
        shutil.rmtree(temp_dir)
