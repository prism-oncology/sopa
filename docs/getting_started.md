## Installing Sopa

Sopa can be installed from `PyPI` on all OS, for any Python version `>=3.11`.

!!! note "Advice (optional)"

    We advise creating a new environment via a package manager.

    For instance, you can create a new `conda` environment:

    ```bash
    conda create --name sopa python=3.12
    conda activate sopa
    ```

Choose one of the following, depending on your needs:

=== "From PyPI"

    ```sh
    pip install sopa
    ```

=== "With conda"

    ```sh
    conda install bioconda::sopa
    ```

=== "Editable mode"

    ``` bash
    git clone https://github.com/prism-oncology/sopa.git
    cd sopa

    # no extra
    pip install  -e .

    # or, to install extras, among cellpose/baysor/stardist/instanseg/wsi:
    pip install -e '.[cellpose,baysor]'
    ```

=== "uv (dev mode)"

    ``` bash
    git clone https://github.com/prism-oncology/sopa.git
    cd sopa

    uv sync --all-extras --dev
    ```

!!! warning "Extra dependencies"
    Dependending on the segmentation tool that you use, you'll need extras, as detailed in the next section.

## Extra dependencies

By default, `sopa` only install the minimal dependencies to avoid a heavy installation. Depending on your usage, you can install some extras. The available extras are listed below.

=== "Cellpose"
    If you want to use [Cellpose](https://github.com/mouseland/cellpose) for cell segmentation, you can install the corresponding extra:

    ```sh
    pip install 'sopa[cellpose]'

    # you can also combine extras: pip install 'sopa[cellpose,baysor,wsi,stardist,instanseg]'
    ```

=== "Proseg"
    [Proseg](https://github.com/dcjones/proseg) has to be installed independently. This can be done with `conda`:

    ```sh
    conda install bioconda::rust-proseg
    ```

    Or [`cargo`](https://doc.rust-lang.org/cargo/getting-started/installation.html):

    ```sh
    cargo install proseg
    ```

=== "Baysor"
    To use [Baysor](https://kharchenkolab.github.io/Baysor/dev/), you'll first need to install Sopa with the `baysor` extra:

    ```sh
    pip install 'sopa[baysor]'
    ```

    Then, also install the `baysor` executable: this can be done via `conda` (recommended, see below) or as detailed in the [official Baysor documentation](https://kharchenkolab.github.io/Baysor/dev/installation/).

    ```sh
    conda install bioconda::baysor-cpp
    ```


=== "Stardist"
    If you want to use [Stardist](https://github.com/stardist/stardist) for cell segmentation, you can install the corresponding extra:

    ```sh
    pip install 'sopa[stardist]'
    ```
=== "InstanSeg"
    If you want to use [InstanSeg](https://github.com/instanseg/instanseg) for cell segmentation, you can install the corresponding extra:

    ```sh
    pip install 'sopa[instanseg]'
    ```
=== "Comseg"
    If you want to use [Comseg](https://github.com/fish-quant/ComSeg) for cell segmentation, you can install it via pip:

    ```sh
    pip install comseg
    ```
=== "WSI"
    If you need to work on whole slide images / H&E images, you can install the corresponding extras as below. You can also consider the `stardist` extra, if you want to run cell segmentation on the H&E image.

    ```sh
    pip install 'sopa[wsi]'
    ```

## Snakemake setup

If you plan to use Snakemake, in addition to the above `sopa` environment, you'll need to clone the `sopa` repository containing the Snakemake workflow:

```sh
git clone https://github.com/prism-oncology/sopa.git
cd sopa   # move inside the sopa repository
```

Also, make sure you have installed `snakemake>=8.0.0`. This does **not** necessarily have to be inside the `sopa` environment — for instance, you can create a new environment specific to snakemake:

```sh
# this will create a new environment called "snakemake"
conda create -c conda-forge -c bioconda -n snakemake snakemake
conda activate snakemake
```

Now, follow our [snakemake tutorial](../tutorials/snakemake) to run your first pipeline.

## Usage

Sopa comes in four different flavours, each corresponding to a different use case:

- `API`: use directly `sopa` as a Python package for full flexibility and customization (see a tutorial [here](../tutorials/api_usage)).
- `Snakemake pipeline`: choose a config, and run our pipeline on your spatial data in a few minutes. See our [snakemake tutorial](../tutorials/snakemake).
- `nf-core/sopa`: run Sopa with Nextflow (see [this repo](https://github.com/nf-core/sopa) and the corresponding [usage guide](https://nf-co.re/sopa/usage)). Great for Docker users.
- `CLI`: use our [command-line-interface](../tutorials/cli_usage) to prototype quickly your own pipeline (advanced users).
