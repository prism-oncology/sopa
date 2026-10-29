When installing `sopa` as written in our [getting-started guidelines](../../getting_started), a new command named `sopa` becomes available.

!!! warning
    The [API](https://prism-oncology.github.io/sopa/tutorials/api_usage/) or [nf-core/sopa pipeline](https://nf-co.re/sopa/dev/) is recommended to get started with Sopa. Using the CLI can be used if you want more flexibility, but you'll need to parallelize the segmentation yourself, as detailed below.

## CLI helper

Run `sopa --help` to get details about all the command line purposes. You can also use this helper on any subcommand, for instance, `sopa convert --help`.

<div class="termy">
```console
// Run the Sopa CLI helper
$ sopa --help
 Usage: sopa [OPTIONS] COMMAND [ARGS]...
╭─ Commands ─────────────────────────────────────────────────────╮
│ aggregate     Aggregate transcripts/channels inside cells      │
│ annotate      Perform cell-type annotation                     │
│ crop          Crop an image based on a user-defined polygon    │
│ explorer      Conversion to the Xenium Explorer's inputs       │
│ patchify      Create patches with overlaps                     │
│ read          Read any technology + write a SpatialData object │
│ report        Create a web-report with figures/QCs             │
│ resolve       Resolve the segmentation conflicts over patches  │
│ segmentation  Perform cell segmentation on patches             │
╰────────────────────────────────────────────────────────────────╯
// Example: run cellpose segmentation
$ sopa segmentation cellpose sdata.zarr
... [Logs] ...
```
</div>

## Save the `SpatialData` object

For this tutorial, we use a generated dataset. You can expect a total runtime of a few minutes.

The command below will generate and save it on disk (you can change the path `tuto.zarr` to save it somewhere else). If you want to load your own data: choose the right panel below. For more information, refer to this [FAQ](../../faq/#what-are-the-inputs-of-sopa) describing which data input you need, or run `sopa convert --help`.

=== "Tutorial"
    ```sh
    # it will generate a 'tuto.zarr' directory
    sopa convert . --sdata-path tuto.zarr --technology toy_dataset
    ```
=== "Xenium"
    ```sh
    # it will generate a '/path/to/sample/directory.zarr' directory
    sopa convert /path/to/sample/directory --technology xenium
    ```
=== "MERSCOPE"
    ```sh
    # it will generate a '/path/to/sample/directory.zarr' directory
    sopa convert /path/to/sample/directory --technology merscope
    ```
=== "CosMx"
    ```sh
    # it will generate a '/path/to/sample/directory.zarr' directory
    sopa convert /path/to/sample/directory --technology cosmx
    ```
=== "PhenoCycler"
    ```sh
    # it will generate a '/path/to/sample.zarr' directory
    sopa convert /path/to/sample.qptiff --technology phenocycler
    ```
=== "MACSima"
    ```sh
    # it will generate a '/path/to/sample/directory.zarr' directory
    sopa convert /path/to/sample/directory --technology macsima
    ```
=== "Other"
    There are also several other readers, such as `hyperion`, or `molecular_cartography`. You can also try generic readers, like `ome_tif`, or even `bioio` which supports many inputs formats. Note that, to use `bioio`, you'll need to `pip install bioio` and potentially other format-specific dependencies as described in their [documentation](https://bioio-devs.github.io/bioio/OVERVIEW.html#reader-installation).

    Replace `<TECHNOLOGY>` by the right name on the following command line:
    ```sh
    # it will generate a '/path/to/sample/directory.zarr' directory
    sopa convert /path/to/sample/directory --technology <TECHNOLOGY>
    ```


!!! info
    It has created a `.zarr` directory which stores a [`SpatialData` object](https://github.com/scverse/spatialdata) corresponding to your data sample. You can choose the location of the `.zarr` directory using the `--sdata-path` command line argument.

## Run segmentation

### Option 1: Cellpose

First, generate the bounding boxes of the patches on which Cellpose will be run. Here, the patches have a width and height of 1500 pixels and an overlap of 50 pixels. We advise bigger sizes for real datasets (see our default parameters in one of our [config files](https://github.com/prism-oncology/sopa/tree/main/workflow/config)). On the toy dataset, this will generate **4** patches.

```sh
sopa patchify image tuto.zarr --patch-width-pixel 1500 --patch-overlap-pixel 50
```

Now, we can run Cellpose on each individual patch. You can either run it directly on all patches (first option below), or on each patch individually (second option - ideal if you want to parallelize it yourself).

=== "Run all patches at once"
    The easiest way to run Cellpose is to use the command below, which directly run Cellpose on all patches and resolve the segmentation. You can add an additional channel, for instance, you can use `--channels DAPI --channels PolyT`.

    ```sh
    sopa segmentation cellpose tuto.zarr \
        --channels DAPI \
        --diameter 35 \
        --min-area 2000
    ```

    By default, this will run cellpose sequentially. To make it run it parallel, you can export the following env variable: `export SOPA_PARALLELIZATION_BACKEND=dask`.

=== "Run on each patch"
    Execute the following command line on all `patch-index` (i.e., `0`, `1`, `2`, and `3`) to run Cellpose using DAPI only (you can add an additional channel, for instance, `--channels DAPI --channels PolyT`):

    !!! tip
        Manually running the commands below can involve using many consecutive commands, so we recommend automatizing it. For instance, this can be done using Snakemake or Nextflow. This will help you parallelize it since you can run each task on separate jobs or using multithreading.

        To automatically get the number of patches, you can either open the `tuto.zarr/.sopa_cache/patches_file_image` file, or compute `len(sdata['image_patches'])` in Python.

    ```sh
    sopa segmentation cellpose tuto.zarr \
        --channels DAPI \
        --diameter 35 \
        --min-area 2000 \
        --patch-index 0

    sopa segmentation cellpose tuto.zarr \
        --channels DAPI \
        --diameter 35 \
        --min-area 2000 \
        --patch-index 1

    sopa segmentation cellpose tuto.zarr \
        --channels DAPI \
        --diameter 35 \
        --min-area 2000 \
        --patch-index 2

    sopa segmentation cellpose tuto.zarr \
        --channels DAPI \
        --diameter 35 \
        --min-area 2000 \
        --patch-index 3
    ```


    At this stage, you executed 4 times Cellpose (once per patch). Now, we need to resolve the conflict, i.e. where boundaries are overlapping due to segmentation on multiple patches:
    ```sh
    sopa resolve cellpose tuto.zarr
    ```

!!! Note
    In the above commands, the `--diameter` and `--min-area` parameters are specific to the data type we work on. For your own data, consider using the default parameters from one of our [config files](https://github.com/prism-oncology/sopa/tree/main/workflow/config). Here, `min-area` is in pixels^2.

### Option 2: Baysor

Baysor needs a config to be executed. You can find official config examples [here](https://github.com/kharchenkolab/Baysor/tree/main/configs).

!!! note
    You can also reuse the Baysor parameter we have defined for each machine, as in our [Snakemake config files](https://github.com/prism-oncology/sopa/tree/main/workflow/config). Note that, our Snakemake config is a `.yaml` file, but the Baysor config should still be a `.toml` file.

For this tutorial, we will use the config below. Save this in a `config.toml` file.
```toml
[data]
force_2d = true
min_molecules_per_cell = 10
x = "x"
y = "y"
z = "z"
gene = "genes"
min_molecules_per_gene = 0
min_molecules_per_segment = 3
confidence_nn_id = 6

[segmentation]
scale = 6         # typical cell radius in microns (IMPORTANT)
scale_std = "25%" # cell radius standard deviation
prior_segmentation_confidence = 0
estimate_scale_from_centers = false
n_clusters = 4
iters = 500
n_cells_init = 0
nuclei_genes = ""
cyto_genes = ""
```

Then, we generate the bounding boxes of the patches on which Baysor will be run. Here, the patches have a width and height of 200 microns. We advise bigger sizes for real datasets (see our default parameters in one of our [config files](https://github.com/prism-oncology/sopa/tree/main/workflow/config)). On the toy dataset, this will generate **4** patches.

```sh
sopa patchify transcripts tuto.zarr --patch-width-microns 200 --prior-shapes-key cellpose_boundaries
```

As for cellpose, you can either run Baysor directly on all patches, or on each patch individually. You can provide the config argument as an inline dictionnary, or as a path to a `.toml` file, as below.

=== "Run all patches at once"
    The easiest way to run Baysor is to use the command below, which directly run Baysor on all patches and resolve the segmentation.

    ```sh
    sopa segmentation baysor tuto.zarr --config '"config.toml"'
    ```

    By default, this will run baysor sequentially. To make it run it parallel, you can export the following env variable: `export SOPA_PARALLELIZATION_BACKEND=dask`.

=== "Run on each patch"
    Now, we can run Baysor on each individual patch. Execute the following command lines to run Baysor on each patch (i.e., `0`, `1`, `2`, and `3`).

    !!! tip
        Manually running the commands below can involve using many consecutive commands, so we recommend automatizing it. For instance, this can be done using Snakemake or Nextflow. This will help you parallelize it since you can run each task on separate jobs or using multithreading. You can also see how we do it in the [Sopa Snakemake pipeline](https://github.com/prism-oncology/sopa/blob/main/workflow/Snakefile).

        To automatically get the number of patches, you can open the `tuto.zarr/.sopa_cache/patches_file_transcripts` file. This lists the names of the directories inside `tuto.zarr/.sopa_cache/baysor` related to each patch. If you selected an ROI, the excluded patches are effectively not in the `patches_file_transcripts` file.

    ```sh
    sopa segmentation baysor tuto.zarr --config '"config.toml"' --patch-index 0

    sopa segmentation baysor tuto.zarr --config '"config.toml"' --patch-index 1

    sopa segmentation baysor tuto.zarr --config '"config.toml"' --patch-index 2

    sopa segmentation baysor tuto.zarr --config '"config.toml"' --patch-index 3
    ```

    At this stage, you executed 4 times Baysor (once per patch). Now, we need to resolve the conflict, i.e. where boundaries are overlapping due to segmentation on multiple patches:
    ```sh
    sopa resolve baysor tuto.zarr --gene-column genes
    ```


### Option 3: Custom staining-based

As for Cellpose, we generate the bounding boxes of the patches on which staining-based segmentation will be run.

```sh
sopa patchify image tuto.zarr --patch-width-pixel 1500 --patch-overlap-pixel 50
```

With the `sopa segmentation generic-staining` command, you can use a custom segmentation method, with the signature described [here](../custom_segmentation/) (or any function named `*_patch` from [this file](https://github.com/prism-oncology/sopa/blob/main/sopa/segmentation/methods/__init__.py)).

For instance, we can use `stardist_patch` directly from the CLI as below.

```sh
sopa segmentation generic-staining tuto.zarr --method-name stardist_patch
```

!!! warning
    The toy dataset is not H&E based, so stardist will fail on this example because of the number of channels. To create the right number of channels on the toy dataset, you can use the following command:
    ```sh
    sopa convert . --sdata-path tuto.zarr --technology toy_dataset --kwargs '{"c_coords": ["r", "g", "b"]}'
    ```

## Aggregation

This **mandatory** step turns the data into an `AnnData` object. We can count the transcript inside each cell, and/or average each channel intensity inside each cell boundary.

=== "Count transcripts + average intensities"
    ```sh
    sopa aggregate tuto.zarr --aggregate-genes --aggregate-channels --min-transcripts 10
    ```
=== "Count transcripts"
    ```sh
    sopa aggregate tuto.zarr --aggregate-genes --min-transcripts 10
    ```
=== "Average intensities"
    ```sh
    sopa aggregate tuto.zarr --aggregate-channels
    ```

## Annotation

If desired, cell-type annotation can be run. Currently, we support Tangram for transcript-based annotation and a simple scoring approach for channel-based annotation (called channel z-score).

=== "Channel Z-score annotation"
    For now, our fluorescence-based annotation is very simple. We provide a dictionary where a channel is associated with a population. Then, each cell is associated with the cell type whose corresponding channel is the brightest (according to a certain Z-score). In this tutorial example, we can annotate Tumoral cells, T cells, and B cells:
    ```sh
    sopa annotate fluorescence tuto.zarr --marker-cell-dict '{"CK": "Tumoral cell", "CD3": "T cell", "CD20": "B cell"}'
    ```
    !!! note "More complex annotation"
        If you have a large number of channels, it may be preferable to run clustering on your data, for instance, using [Leiden clustering](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html). Then, you can annotate each cluster manually by plotting a heatmap of all channels expressions per cluster.
=== "Tangram annotation"
    [Tangram](https://github.com/broadinstitute/Tangram) is a transcript-based annotation that uses an annotated single-cell reference. Let's suppose your reference `AnnData` object is stored in a file called `adata_reference.h5ad` (preferably, keep raw counts), and the cell type is in `adata.obs["cell_type"]`. Then, you can annotate your spatial data as follows:
    ```sh
    sopa annotate tangram tuto.zarr --sc-reference-path adata_reference.h5ad --cell-type-key cell_type
    ```


## Pipeline report

You can optionally create an HTML report of the pipeline run (in the example below, we save it under `report.html`). It contains some quality controls for your data.

```sh
sopa report tuto.zarr report.html
```

## Visualization (Xenium Explorer)
The Xenium Explorer is a software developed by 10X Genomics for visualizing spatial data, and it can be downloaded freely [here](https://www.10xgenomics.com/support/software/xenium-explorer/latest). Sopa allows the conversion to the Xenium Explorer. It will create some files under a new `tuto.explorer` directory:

```sh
sopa explorer write tuto.zarr
```

If you have downloaded the Xenium Explorer, you can now open the results in the explorer: `open tuto.explorer/experiment.xenium` (if using a Unix operating system), or double-click on the latter file.

!!! License
    _Xenium Explorer_ is a registered trademark of 10x Genomics. The Xenium Explorer is licensed for usage on Xenium data (more details [here](https://www.10xgenomics.com/legal/end-user-software-license-agreement)).

!!! info "Time efficiency"
    Creating the image needed by the Xenium Explorer can be time-consuming. Therefore, we recommend performing one run for the image generation (below) and another to save the transcripts/boundaries/observations.
    ```sh
    # this can be done directly after saving the raw data in a .zarr directory
    sopa explorer write tuto.zarr --mode "+i" --no-save-h5ad
    ```

    After running everything with Sopa, you can finally save all the other Xenium Explorer input (e.g. boundaries and cell categories):
    ```sh
    # this should be done after aggregation and an eventual annotation
    sopa explorer write tuto.zarr --mode "-i"
    ```
    For more details and customization, run the command line helper with `sopa explorer write --help`.

## Further analyses

- If you are familiar with the [`spatialdata` library](https://github.com/scverse/spatialdata), you can directly use the `tuto.zarr` directory, corresponding to a `SpatialData` object:
```python
import spatialdata

sdata = spatialdata.read_zarr("tuto.zarr")
```
- You can use [Squidpy](https://squidpy.readthedocs.io/en/latest/index.html) which operates on both the `SpatialData` object or the `AnnData` object, or use other tools of the `scverse` ecosystem such as [`Scanpy`](https://scanpy.readthedocs.io/en/stable/index.html).
- You can also use the file `tuto.explorer/adata.h5ad` if you prefer the `AnnData` object instead of the full `SpatialData` object.
