from __future__ import annotations

from pathlib import Path

from .constants import STAINING_BASED_METHODS, TOY_READERS, TRANSCRIPT_BASED_METHODS


def validate_config(config: dict) -> dict:
    """Basic config sanity check before running Snakemake"""

    assert "read" in config and "technology" in config["read"], (
        "Invalid config. Provide a 'read' section in the config file, and specify the 'technology' key"
    )

    assert "segmentation" in config, "Invalid config. Provide a 'segmentation' section in the config file"

    backward_compatibility(config)

    check_segmentation_methods(config)

    check_data_paths(config)

    check_prior_shapes_key(config)

    if "baysor" in config["segmentation"]:
        check_executable_path(config, "baysor", ".julia")

    if "proseg" in config["segmentation"]:
        check_executable_path(config, "proseg", ".cargo")

        assert "prior_shapes_key" in config["segmentation"]["proseg"], (
            "Invalid config. Provide a 'prior_shapes_key' key in the 'proseg' section of the config file"
        )

        if "patch_width_microns" in config["patchify"]:
            assert config["patchify"]["patch_width_microns"] == -1, (
                "Invalid config. 'patch_width_microns' must be -1 for 'proseg' segmentation method"
            )
        config["patchify"]["patch_width_microns"] = -1

    if "fullres_image_file" in config:  # convenient way to pass the fullres image file path to the reader
        config["read"]["kwargs"] = config["read"].get("kwargs", {})
        config["read"]["kwargs"]["fullres_image_file"] = config["fullres_image_file"]

    return config


def check_segmentation_methods(config: dict):
    assert len(config["segmentation"]) > 0, (
        "Invalid config. Provide at least one segmentation method in the 'segmentation' section"
    )

    assert sum(method in config["segmentation"] for method in TRANSCRIPT_BASED_METHODS) <= 1, (
        f"Only one of the following methods can be used: {TRANSCRIPT_BASED_METHODS}"
    )

    assert sum(method in config["segmentation"] for method in STAINING_BASED_METHODS) <= 1, (
        f"Only one of the following methods can be used: {STAINING_BASED_METHODS}"
    )

    if "stardist" in config["segmentation"]:
        if config["read"]["technology"] == "visium_hd":
            assert "proseg" in config["segmentation"], (
                "Invalid config. 'stardist' can only be used with 'proseg' for 'visium_hd' technology"
            )
        else:
            assert not any(method in config["segmentation"] for method in TRANSCRIPT_BASED_METHODS), (
                "Invalid config. 'stardist' cannot be combined with transcript-based methods"
            )


def check_prior_shapes_key(config: dict):
    for method in TRANSCRIPT_BASED_METHODS:
        if method in config["segmentation"] and "cellpose" in config["segmentation"]:
            config["segmentation"][method]["prior_shapes_key"] = "cellpose_boundaries"


def check_data_paths(config: dict):
    config["is_toy_reader"] = config["read"]["technology"] in TOY_READERS

    if config["is_toy_reader"]:
        config["data_path"] = "."

    assert "data_path" in config or "sdata_path" in config, (
        "Invalid config. Provide '--config data_path=...' when running the pipeline"
    )

    if "data_path" in config and "sdata_path" not in config:
        config["sdata_path"] = Path(config["data_path"]).with_suffix(".zarr")
        print(
            f"SpatialData object path set to default: {config['sdata_path']}\nTo change this behavior, provide `--config sdata_path=...` when running the snakemake pipeline"
        )

    if "data_path" not in config:
        assert Path(config["sdata_path"]).exists(), (
            f"When `data_path` is not provided, the spatial data object must exist, but the directory doesn't exists: {config['sdata_path']}"
        )
        config["data_path"] = []


def check_executable_path(config: dict, name: str, default_dir: str):
    import shutil

    if shutil.which(name) is not None:
        return

    default_path = Path.home() / default_dir / "bin" / name
    if default_path.exists():
        return

    bin_path = Path.home() / ".local" / "bin" / name
    error_message = f"Please install {name} and ensure that either `{default_path}` executes {name}, or that `{name}` is an existing command (add it to your PATH, or create a symlink at {bin_path})."

    if "executables" in config and name in config["executables"]:
        raise ValueError(f"The config['executables']['{name}'] argument is deprecated. {error_message}.")
    raise KeyError(error_message)


def backward_compatibility(config: dict) -> None:
    """Ensure backward compatibility with old config files"""

    for method in TRANSCRIPT_BASED_METHODS:
        if method in config["segmentation"] and ("cell_key" in config["segmentation"][method]):
            print("Snakemake argument 'cell_key' is deprecated. Use 'prior_shapes_key' instead.")
            config["segmentation"][method]["prior_shapes_key"] = config["segmentation"][method]["cell_key"]
            del config["segmentation"][method]["cell_key"]

    if "aggregate" in config and "average_intensities" in config["aggregate"]:
        print("Snakemake aggregate argument 'average_intensities' is deprecated. Use 'aggregate_channels' instead.")
        config["aggregate"]["aggregate_channels"] = config["aggregate"]["average_intensities"]
        del config["aggregate"]["average_intensities"]
