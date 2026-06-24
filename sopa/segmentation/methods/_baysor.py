import logging
from functools import partial
from pathlib import Path

from spatialdata import SpatialData

from ... import settings
from ...constants import SopaFiles, SopaKeys
from ...utils import (
    delete_transcripts_patches_dirs,
    get_feature_key,
    get_transcripts_patches_dirs,
    set_boundaries_attrs,
)
from .utils import _check_transcript_patches, _get_executable_path, resolve

log = logging.getLogger(__name__)


def baysor(
    sdata: SpatialData,
    config: dict | str | None = None,
    min_area: int = 0,
    delete_cache: bool = True,
    recover: bool = False,
    force: bool = False,
    scale: float | None = None,
    key_added: str = SopaKeys.BAYSOR_BOUNDARIES,
    patch_index: int | None = None,
):
    """Run [Baysor](https://kharchenkolab.github.io/Baysor/dev/) segmentation on a SpatialData object, and add a GeoDataFrame containing the cell boundaries.

    !!! warning "Baysor installation"
        Make sure to install [Baysor](https://kharchenkolab.github.io/Baysor/dev/installation/), and either have the executable at `~/.julia/bin/baysor`, or create an alias called
        `baysor` that points to the binary executable. Also, you'll need to install
        sopa with the baysor extra: `pip install 'sopa[baysor]'` (basically, this installs `toml` and `loompy`).

    !!! info "Inferred config"
        If the `config` argument is not provided, the configuration is inferred.
        If [sopa.make_transcript_patches][] was run with a `prior_shapes_key`, the configuration is inferred based on the prior segmentation.
        Otherwise, the configuration is inferred based on the `scale` parameter (you'll need to provide it).

    Args:
        sdata: A `SpatialData` object.
        config: Optional configuration dictionary or path to a TOML file containing a valid Baysor config. By default, a configuration is inferred based on the cell area of the prior segmentation, or based on the `scale` parameter.
        min_area: Minimal area (in microns^2) of a cell to be considered.
        delete_cache: Whether to delete the cache after segmentation.
        recover: If `True`, recover the cache from a failed segmentation, and continue.
        force: If `True`, ignore failed patches and continue with the successful ones.
        scale: The typical cell radius in microns. If `config` is not provided, the configuration is inferred based on this parameter. Else, it will overwrite the existing scale in the config.
        key_added: Name of the shapes element to be added to `sdata.shapes`.
        patch_index: Index of the patch to segment (we do not recommend to set this argument). By default, segment all patches.
    """
    _check_transcript_patches(sdata)

    prior_shapes_key = (
        SopaKeys.SOPA_PRIOR if SopaKeys.PRIOR_SHAPES_KEY in sdata.shapes[SopaKeys.TRANSCRIPTS_PATCHES] else None
    )

    if config is None or not len(config):
        config = _get_default_config(sdata, prior_shapes_key, scale)
    elif scale is not None:
        assert "segmentation" in config, "The provided config should contain a 'segmentation' key"
        config["segmentation"]["scale"] = scale

    assert "data" in config, "The provided config should contain a 'data' key"

    if "gene" not in config["data"]:
        config["data"]["gene"] = str(get_feature_key(sdata, raise_error=True))

    if "scale" in config["segmentation"] and config["segmentation"]["scale"] >= 20:
        log.warning(
            f"The provided scale ({config['segmentation']['scale']}) seems large. Make sure the scale is in microns, and not in pixels. If you are using CosMX data, note that the transcripts coordinates are now in microns."
        )

    baysor_command = _get_baysor_command(prior_shapes_key)

    baysor_patch = BaysorPatch(baysor_command, config, force=force, capture_output=patch_index is None)

    if patch_index is not None:
        patch_dir = get_transcripts_patches_dirs(sdata, patch_index)
        baysor_patch(patch_dir)
        return

    patches_dirs = get_transcripts_patches_dirs(sdata)

    remaining_patches_dirs = (
        [patch_dir for patch_dir in patches_dirs if not (patch_dir / "segmentation_counts.loom").exists()]
        if recover
        else patches_dirs
    )

    settings._run_with_backend([partial(baysor_patch, patch_dir) for patch_dir in remaining_patches_dirs])

    if force:
        patches_dirs = [patch_dir for patch_dir in patches_dirs if (patch_dir / "segmentation_counts.loom").exists()]
        assert patches_dirs, "Baysor failed on all patches"

    resolve(sdata, patches_dirs, gene_column=config["data"]["gene"], min_area=min_area, key_added=key_added)

    set_boundaries_attrs(sdata, key_added)

    if delete_cache:
        delete_transcripts_patches_dirs(sdata)


class BaysorPatch:
    def __init__(
        self,
        baysor_command: str,
        config: dict | str,
        force: bool = False,
        capture_output: bool = True,
    ):
        self.baysor_command = baysor_command
        self.config = config
        self.force = force
        self.capture_output = capture_output

    def __call__(self, patch_dir: Path):
        _copy_segmentation_config(patch_dir / SopaFiles.TOML_CONFIG_FILE, self.config)

        import subprocess

        result = subprocess.run(self.baysor_command, cwd=patch_dir, shell=True, capture_output=self.capture_output)

        if result.returncode != 0 or not (patch_dir / "segmentation_counts.loom").exists():
            if self.force:
                log.warning(f"Baysor error on patch {patch_dir.resolve()} with command `{self.baysor_command}`")
                return
            raise subprocess.CalledProcessError(
                returncode=result.returncode,
                cmd=self.baysor_command,
                output=result.stdout,
                stderr=result.stderr,
            )


def _get_baysor_command(prior_shapes_key: str | None) -> str:
    baysor_executable_path = _get_executable_path("baysor", ".julia")

    use_polygons_format_argument = _use_polygons_format_argument(baysor_executable_path)
    polygon_format = (
        "--polygon-format GeometryCollection" if use_polygons_format_argument else "--save-polygons GeoJSON"
    )  # depends on the version of baysor

    prior_suffix = f" :{prior_shapes_key}" if prior_shapes_key else ""  # use a prior segmentation

    return f"{baysor_executable_path} run {polygon_format} -c config.toml transcripts.csv -o ." + prior_suffix


def _use_polygons_format_argument(baysor_executable_path: str) -> bool:
    import subprocess

    from packaging.version import InvalidVersion, Version

    result = subprocess.run(f"{baysor_executable_path} run --version", shell=True, capture_output=True, text=True)

    try:
        return Version(result.stdout) >= Version("0.7.0")
    except InvalidVersion:
        log.warning("Could not parse the version of baysor. Assumes baysor >= 0.7.0.")
        return True


def _get_default_config(sdata: SpatialData, prior_shapes_key: str | None, scale: float | None) -> dict:
    assert prior_shapes_key is not None or scale is not None, (
        "The config can't be inferred. Choose among the following solutions:\n"
        "   - Provide the `scale` argument (typical cell radius in microns)\n"
        "   - Run `sopa.make_transcript_patches(...)` with `prior_shapes_key`\n"
        "   - Provide the `config` argument, containing a valid Baysor config."
    )

    config = {
        "data": {
            "x": "x",
            "y": "y",
            "gene": str(get_feature_key(sdata, raise_error=True)),
            "min_molecules_per_gene": 10,
            "min_molecules_per_cell": 20,
            "force_2d": True,
        },
        "segmentation": {"prior_segmentation_confidence": 0.8},
    }

    if scale is not None:
        config["segmentation"]["scale"] = scale

    log.info(f"The Baysor config was not provided, using the following by default:\n{config}")

    return config


def _copy_segmentation_config(path: Path | str, config: dict | str):
    """Copy the segmentation config to a file (`.json` or `.toml`).

    Args:
        path: Where the config will be saved
        config: Dictionnary config, or path to an existing config file (json or toml)
    """
    path = Path(path)

    if isinstance(config, str):
        import shutil

        shutil.copy(config, path)
        return

    assert path.suffix == ".toml"

    try:
        import toml
    except ImportError:
        raise ImportError(
            "To use baysor, you need its corresponding sopa extra: `pip install 'sopa[baysor]'`.\
            \nAlso, make sure to install the baysor executable (https://github.com/kharchenkolab/Baysor)."
        )

    with open(path, "w") as f:
        toml.dump(config, f)
        return
