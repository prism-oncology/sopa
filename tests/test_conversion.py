import numpy as np
from spatialdata.models import Image2DModel

from sopa.io.reader.utils import _to_uint8


def test_to_uint8() -> None:
    x = 1 + np.random.rand(10, 10, 3)
    x[:, :, 0] *= 2

    image = Image2DModel.parse(x, dims=["y", "x", "c"])
    converted_image = _to_uint8(image)

    assert (converted_image.max(dim=("y", "x")).data.compute() == 255).all()

    ratio = (converted_image.sel(c=0) / image.sel(c=0)).data.compute()

    assert abs(ratio.mean() - 255 / 4) < 10
    assert ratio.std() < 5

    ratio = (converted_image.sel(c=1) / image.sel(c=1)).data.compute()

    assert abs(ratio.mean() - 255 / 2) < 10
    assert ratio.std() < 5
