import ast

import typer

app_cache = typer.Typer()


@app_cache.command()
def cellpose(
    model_type: str = typer.Option("cyto3", help="Name of the cellpose model if using `cellpose<4.0.0`."),
    pretrained_model: str = typer.Option(
        None,
        help="Name of pretrained model (e.g., `cpsam`, if using `cellpose>=4.0.0`), or path to the pretrained model to be loaded, or `None`.",
    ),
    model_dir: str = typer.Option(
        None,
        help="Directory where the model is stored. By default, uses the cellpose default directory (`~/.cellpose/models`, or `$CELLPOSE_LOCAL_MODELS_PATH`)",
    ),
    method_kwargs: str = typer.Option(
        {},
        callback=ast.literal_eval,
        help="Kwargs for the `cellpose.models.CellposeModel` object. This should be a dictionnary, in inline string format.",
    ),
):
    """Download the Cellpose model, so that they are cached before running the segmentation."""
    import logging
    import os
    from pathlib import Path

    if model_dir:
        model_dir_path = Path(model_dir).resolve()
        model_dir_path.mkdir(parents=True, exist_ok=True)
        os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(model_dir_path)

    from sopa.segmentation.methods._cellpose import _cellpose_version_check, load_cellpose_model

    log = logging.getLogger(__name__)

    _, pretrained_model = _cellpose_version_check(pretrained_model)

    load_cellpose_model(
        model_type=model_type,
        pretrained_model=pretrained_model,
        cellpose_model_kwargs=method_kwargs,
    )

    log.info(f"Cellpose model '{pretrained_model or model_type}' is cached and ready to be used.")
