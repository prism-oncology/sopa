<p align="center">
  <img src="https://raw.githubusercontent.com/prism-oncology/sopa/main/docs/assets/sopa.png" alt="sopa_logo" width="250"/>
</p>
<p align="center"><b><i>
	Spatial omics pipeline and analysis
</b></i></p>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/sopa.svg)](https://pypi.org/project/sopa)
[![Downloads](https://static.pepy.tech/badge/sopa)](https://pepy.tech/project/sopa)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://prism-oncology.github.io/sopa)
![Build](https://github.com/prism-oncology/sopa/workflows/ci/badge.svg)
[![License](https://img.shields.io/pypi/l/sopa.svg)](https://github.com/prism-oncology/sopa/blob/main/LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

Built on top of [SpatialData](https://github.com/scverse/spatialdata), Sopa enables processing and analyses of spatial omics data with single-cell resolution (spatial transcriptomics or multiplex imaging data) using a standard data structure and output. We currently support the following technologies: Xenium, Visium HD, MERSCOPE, CosMx, PhenoCycler, MACSima, Molecular Cartography, and others. Sopa was designed for generability and low memory consumption on large images (scales to `1TB+` images).

> [!NOTE]
> You may also be interested in [Novae](https://github.com/prism-oncology/novae), developed by the same authors, now published in [Nature Methods](https://www.nature.com/articles/s41592-025-02899-6) 🎉

## Documentation

Check [Sopa's documentation](https://prism-oncology.github.io/sopa) to get started. It contains installation explanations, CLI/API details, and tutorials.

## Overview

The following illustration describes the main steps of `sopa`:

<p align="center">
  <img src="https://raw.githubusercontent.com/prism-oncology/sopa/main/docs/assets/overview_white.png" alt="sopa_overview" width="100%"/>
</p>

## Installation

Sopa can be installed from `PyPI` on all OS, for any Python version `>=3.11`. On a new environment, run the following command:
```sh
pip install sopa
```

> [!NOTE]
> See this [installation section](https://prism-oncology.github.io/sopa/getting_started/) for more details about extras and other installations modes.

## Usage modes
Sopa comes in four different flavours, each corresponding to a different use case:
- `API`: use directly `sopa` as a Python package for complete flexibility and customization (example below).
- `nf-core/sopa`: run Sopa with Nextflow (see [this repo](https://github.com/nf-core/sopa) and the corresponding [usage guide](https://nf-co.re/sopa/usage)).
- `Snakemake pipeline`: choose a config, and run our pipeline on your spatial data in a couple of minutes (deprecated in favor of `nf-core/sopa`, see the public archive [here](https://github.com/prism-oncology/snakemake-sopa)).
- `CLI`: use our command-line-interface for prototyping quickly your own pipeline (advanced users).

## API usage

Below is a minimal example of API usage. For a complete API description, please refer to the [documentation](https://prism-oncology.github.io/sopa/tutorials/api_usage/).

```python
import sopa

sdata = sopa.io.xenium("path/to/data") # reading Xenium data

sopa.make_image_patches(sdata) # creating overlapping patches
sopa.segmentation.cellpose(sdata, "DAPI", diameter=30) # running cellpose segmentation
sopa.aggregate(sdata) # counting the transcripts inside the cells
```

## Cite us
Our article is published in [Nature Communications](https://www.nature.com/articles/s41467-024-48981-z). You can cite Sopa as below:

```txt
Blampey, Q., Mulder, K., Gardet, M. et al. Sopa: a technology-invariant pipeline for analyses of image-based spatial omics.
Nat Commun 15, 4981 (2024). https://doi.org/10.1038/s41467-024-48981-z
```
