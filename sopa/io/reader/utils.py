import logging
from pathlib import Path
from typing import Callable

import dask.array as da
import numpy as np
import pandas as pd
import tifffile as tf
import xarray as xr
from dask_image.imread import imread
from spatialdata import SpatialData
from spatialdata.models import Image2DModel
from xarray import DataArray

from ...constants import SopaAttrs

log = logging.getLogger(__name__)


def _default_image_kwargs(
    image_models_kwargs: dict | None = None, imread_kwargs: dict | None = None
) -> tuple[dict, dict]:
    image_models_kwargs = {} if image_models_kwargs is None else image_models_kwargs
    imread_kwargs = {} if imread_kwargs is None else imread_kwargs

    if "chunks" not in image_models_kwargs:
        image_models_kwargs["chunks"] = (1, 1024, 1024)

    if "scale_factors" not in image_models_kwargs:
        image_models_kwargs["scale_factors"] = [2, 2, 2, 2]

    return image_models_kwargs, imread_kwargs


def _deduplicate_names(names: pd.Series | np.ndarray | list[str]) -> np.ndarray:
    if not isinstance(names, pd.Series):
        names = pd.Series(names)
    names = names.astype(str)

    duplicates = names.duplicated()
    names[duplicates] += " (" + names.groupby(by=names).cumcount().astype(str)[duplicates] + ")"

    return names.values


def _get_ome_channel_names(files):
    return _deduplicate_names([_ome_channels_names(file)[0] for file in files])


def _general_tif_directory_reader(
    path: str,
    files_to_channels: Callable = _get_ome_channel_names,
    suffix: str = ".tif",
    image_models_kwargs: dict | None = None,
    imread_kwargs: dict | None = None,
):
    image_models_kwargs, imread_kwargs = _default_image_kwargs(image_models_kwargs, imread_kwargs)

    files = [file for file in Path(path).iterdir() if file.suffix == suffix]

    names = files_to_channels(files)
    image = da.concatenate(
        [imread(file, **imread_kwargs) for file in files],
        axis=0,
    )
    image = image.rechunk(chunks=image_models_kwargs["chunks"])

    log.info(f"Found channel names {names}")

    image_name = Path(path).absolute().stem
    image = Image2DModel.parse(
        image,
        dims=("c", "y", "x"),
        c_coords=names,
        **image_models_kwargs,
    )

    return SpatialData(images={image_name: image}, attrs={SopaAttrs.CELL_SEGMENTATION: image_name})


def _to_uint8(image: xr.DataArray) -> xr.DataArray:
    channels_max = image.max(dim=("y", "x")).data.compute()[:, None, None]

    return ((image / channels_max).clip(0, 1) * 255).astype(np.uint8)


def _image_int_dtype(image: xr.DataArray) -> xr.DataArray:
    image = image.transpose("c", "y", "x")

    if np.issubdtype(image.dtype, np.integer):
        return image

    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"Invalid image type {image.dtype}")

    return _to_uint8(image)


def _ome_channels_names(path: Path | str):
    import xml.etree.ElementTree as ET

    tiff = tf.TiffFile(path)
    omexml_string = tiff.pages[0].description

    root = ET.fromstring(omexml_string)
    namespaces = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    channels = root.findall("ome:Image[1]/ome:Pixels/ome:Channel", namespaces)
    return [c.attrib["Name"] if "Name" in c.attrib else c.attrib["ID"] for c in channels]


def ome_tif(path: str | Path, as_image: bool = False) -> DataArray | SpatialData:
    """Read an `.ome.tif` image. This image should be a 2D image (with possibly multiple channels).
    Typically, this function can be used to open Xenium IF images.

    Args:
        path: Path to the `.ome.tif` image
        as_image: If `True`, will return a `DataArray` object

    Returns:
        A `DataArray` or a `SpatialData` object
    """
    image_models_kwargs, _ = _default_image_kwargs()
    image_name = Path(path).absolute().name.split(".")[0]
    image: da.Array = imread(path)

    if image.ndim == 4:
        assert image.shape[0] == 1, "4D images not supported"
        image = da.moveaxis(image[0], 2, 0)
        log.info(f"Transformed 4D image into a 3D image of shape (c, y, x) = {image.shape}")
    elif image.ndim != 3:
        raise ValueError(f"Number of dimensions not supported: {image.ndim}")

    image = image.rechunk(chunks=image_models_kwargs["chunks"])

    try:
        channel_names = _ome_channels_names(path)
    except:
        channel_names = []
    if len(channel_names) != len(image):
        channel_names = [str(i) for i in range(len(image))]
        log.warning(f"Channel names couldn't be read. Using {channel_names} instead.")

    image = DataArray(image, dims=["c", "y", "x"], name=image_name, coords={"c": channel_names})
    image = _image_int_dtype(image)

    if as_image:
        return image

    image = Image2DModel.parse(
        image,
        c_coords=channel_names,
        **image_models_kwargs,
    )

    return SpatialData(images={image_name: image}, attrs={SopaAttrs.CELL_SEGMENTATION: image_name})
