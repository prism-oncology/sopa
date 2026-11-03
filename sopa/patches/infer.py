import logging
from collections.abc import Callable

import tqdm
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import TableModel

from ..constants import SopaKeys
from ..utils import add_spatial_element
from . import TileLoader

log = logging.getLogger(__name__)


def compute_embeddings(
    sdata: SpatialData,
    model: Callable | str,
    patch_width: int,
    patch_overlap: int = 0,
    level: int | None = 0,
    magnification: int | None = None,
    image_key: str | None = None,
    batch_size: int = 32,
    device: str | None = None,
    data_parallel: bool | list[int] = False,
    roi_key: str | None = SopaKeys.ROI,
    key_added: str | None = None,
    **kwargs: int,
) -> str:
    """It creates patches, runs a computer vision model on each patch, and store the embeddings of each all patches as an [`AnnData` object](https://anndata.readthedocs.io/en/stable/). This is mostly useful for WSI images.

    !!! info
        The `AnnData` object will be saved into the `SpatialData` object with the key `"{model_name}_embeddings"` (see the `model_name` argument below), except if `key_added` is provided.
        The shapes of the patches will be saved with the key `"embeddings_patches"`.

    !!! warning
        In addition to the WSI extra (`pip install 'sopa[wsi]'`) and depending on the model used, you might need to install additional dependencies. Also, CONCH requires to be logged in Hugging Face and having approved their License.

    Args:
        sdata: A `SpatialData` object
        model: A supported model name (`resnet50`, `histo_ssl`, `dinov2`, `hoptimus0`, or `conch`), or a callable that takes as an input a tensor of size `(batch_size, channels, x, y)` and returns a vector for each tile `(batch_size, emb_dim)`.
        patch_width: Width of the patches in pixels.
        patch_overlap: Width of the overlap between the patches in pixels.
        level: Image level on which the processing is performed. Either `level` or `magnification` should be provided.
        magnification: The target magnification on which the processing is performed. If `magnification` is provided, the `level` argument will be automatically computed.
        image_key: Optional image key of the image. By default, uses the only image (if only one) or the image used for cell or tissue segmentation.
        batch_size: Mini-batch size used during inference.
        device: Device used for the computer vision model.
        data_parallel: If `True`, the model will be run in data parallel mode. If a list of GPUs is provided, the model will be run in data parallel mode on the specified GPUs.
        roi_key: Optional name of the shapes that needs to touch the patches. Patches that do not touch any shape will be ignored. If `None`, all patches will be used. By default, uses the tissue segmentation if available.
        key_added: Optional name of the spatial element that will be added (storing the embeddings).
        **kwargs: Additional keyword arguments passed to the `Patches2D` constructor.

    Returns:
        The name of the `AnnData` table that was added to the `SpatialData` object.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "For patch embedding, you need `torch` (and perhaps `torchvision`). Consider installing the sopa WSI extra: `pip install 'sopa[wsi]'`."
        )
    from . import models

    if isinstance(model, str):
        assert model in models.available_models, (
            f"'{model}' is not a valid model name. Valid names are: {', '.join(list(models.available_models.keys()))}"
        )
        model_name, model = model, models.available_models[model]()
    else:
        model_name = model.__class__.__name__

    if device is not None:
        model.to(device)

    if data_parallel:
        ids = data_parallel if isinstance(data_parallel, list) else list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=ids)

    tile_loader = TileLoader(sdata, patch_width, image_key, level, magnification, patch_overlap, roi_key)

    log.info(f"Processing {len(tile_loader)} patches extracted from level {tile_loader.level}")

    predictions = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(tile_loader), batch_size)):
            batch = tile_loader[i : i + batch_size]
            embedding: torch.Tensor = model(batch.to(device))
            assert len(embedding.shape) == 2, "The model must have the signature (B, C, Y, X) -> (B, C)"

            predictions.append(embedding.cpu())

    predictions = torch.cat(predictions)
    if len(predictions.shape) == 1:
        predictions = torch.unsqueeze(predictions, 1)

    patches = tile_loader.patches
    patches.add_shapes(key_added=SopaKeys.EMBEDDINGS_PATCHES)

    gdf = sdata[SopaKeys.EMBEDDINGS_PATCHES]

    adata = AnnData(predictions.numpy())
    adata.obs["region"] = SopaKeys.EMBEDDINGS_PATCHES
    adata.obs["instance"] = gdf.index.values
    adata = TableModel.parse(
        adata,
        region=SopaKeys.EMBEDDINGS_PATCHES,
        region_key="region",
        instance_key="instance",
    )
    adata.obsm["spatial"] = patches.centroids()
    adata.uns["embedding_config"] = {
        "patch_width": patch_width,
        "patch_overlap": patch_overlap,
        "magnification": magnification,
        "level": tile_loader.level,
        "level_downsample": tile_loader.level_downsample,
        "tile_resize_factor": tile_loader.tile_resize_factor,
        "model_name": model_name,
    }

    key_added = key_added or f"{model_name}_embeddings"
    add_spatial_element(sdata, key_added, adata)

    return key_added
