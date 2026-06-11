import pandas as pd
import zarr
from zarr.storage import ZipStore

import sopa
from sopa.io.explorer.utils import int_cell_id, is_valid_explorer_id, str_cell_id


def test_is_valid_explorer_id():
    assert is_valid_explorer_id("aaaaaaaa-1")
    assert is_valid_explorer_id("aaaaabcd")
    assert not is_valid_explorer_id("aaaaaaaa-")
    assert not is_valid_explorer_id("aaaaaaa1")
    assert not is_valid_explorer_id("aaaaaaaa-11")
    assert not is_valid_explorer_id("aaabb")
    assert not is_valid_explorer_id(13)


def test_convert_ids():
    indices = pd.Index(range(17))
    ids = str_cell_id(indices)
    assert len(ids.unique()) == 17
    assert (int_cell_id(ids) == indices).all()

    assert (
        ids
        == [
            "aaaaaaaa-1",
            "aaaaaaab-1",
            "aaaaaaac-1",
            "aaaaaaad-1",
            "aaaaaaae-1",
            "aaaaaaaf-1",
            "aaaaaaag-1",
            "aaaaaaah-1",
            "aaaaaaai-1",
            "aaaaaaaj-1",
            "aaaaaaak-1",
            "aaaaaaal-1",
            "aaaaaaam-1",
            "aaaaaaan-1",
            "aaaaaaao-1",
            "aaaaaaap-1",
            "aaaaaaba-1",
        ]
    ).all()


def test_explorer_write():
    sdata = sopa.io.toy_dataset(as_output=True)

    sopa.io.explorer.write("tests/out.explorer", sdata)

    with ZipStore("tests/out.explorer/transcripts.zarr.zip") as store:
        z = zarr.open(store, mode="r")

        LOCS = ["0,0", "0,1", "1,0", "1,1"]

        assert set(z["grids"]["0"].group_keys()) == set(LOCS)

        expected_count = sdata["transcripts"].shape[0].compute()
        assert sum(z["grids"]["0"][locs]["location"].shape[0] for locs in LOCS) == expected_count
