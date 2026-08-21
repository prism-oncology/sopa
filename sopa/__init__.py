import contextlib
import importlib.metadata
import logging
import os
import sys
from ._logging import configure_logger

__version__ = importlib.metadata.version("sopa")

log = logging.getLogger("sopa")
configure_logger(log)


def _configure_anndata():
    """Write string columns in the legacy (non-nullable) format.

    Since pandas 3, anndata defaults to the `nullable-string-array` encoding, which
    external readers (e.g. proseg) can't parse.
    """
    import anndata

    if "ANNDATA_ALLOW_WRITE_NULLABLE_STRINGS" in os.environ:
        return

    with contextlib.suppress(AttributeError, ValueError):
        anndata.settings.allow_write_nullable_strings = False


if not any(f"--{option}" in sys.argv for option in ["version", "help"]):  # no import for cli helpers
    from ._settings import settings

    _configure_anndata()

    from . import utils
    from . import shapes
    from . import io
    from . import spatial
    from . import segmentation
    from .aggregation import aggregate, overlay_segmentation
    from .patches import make_transcript_patches, make_image_patches, compute_embeddings, cluster_embeddings
    from .utils import get_spatial_image, get_spatial_element, to_intrinsic, get_boundaries
