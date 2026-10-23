## What are the inputs of Sopa?

You need the raw inputs of your machine, that is:

- One or multiple image(s), usually corresponding to one or multiple `.tiff` file(s)

- Optionally, a file of transcript location, usually a `.csv` or `.parquet` file

In this documentation, `data_path` denotes the path to your raw data. Select the correct tab below to understand what is the right path to your raw data:

=== "Xenium"
    `data_path` is the path to the directory containing the following files: `morphology.ome.tif`, `experiment.xenium` and `transcripts.parquet`. In brief, you should have this file structure:

    ```txt
    .
    ├─ morphology_focus.ome.tif   # or a directory (for recent versions of the Xenium)
    ├─ experiment.xenium
    └─ transcripts.parquet
    ```

=== "MERSCOPE"
    `data_path` is the path to the "region" directory containing a `detected_transcripts.csv` file and an `images` directory. For instance, the directory may be called `region_0`. In brief, you should have this file structure:

    ```txt
    .
    ├─ detected_transcripts.csv
    └─ images
       ├─ mosaic_{stain}_z{z_layer}.tif
       └─ micron_to_mosaic_pixel_transform.csv
    ```
=== "CosMx"
    `data_path` is the path to the directory containing:

    - a transcript file `*_tx_file` (with columns `target`, `x_global_px`, `y_global_px`)
    - a FOV locations file `*_fov_positions_file` (with columns `FOV`, `X_mm`, `Y_mm`)
    - a `Morphology_ChannelID_Dictionary.txt` file containing channel names
    - a `Morphology2D` directory containing the images, end in `_F*.TIF`.

    These files must be exported as flat files in AtomX. That is: within a study, click on "Export" and then select files from the "Flat CSV Files" section (transcripts flat and FOV position flat). You should have this file structure:

    ```txt
    .
    ├─ <DATASET_ID>_tx_file.csv (or csv.gz)
    ├─ <DATASET_ID>_fov_positions_file.csv (or csv.gz)
    ├─ Morphology_ChannelID_Dictionary.txt
    └─ Morphology2D
       ├─ XXX_F001.TIF
       ├─ XXX_F002.TIF
       └─ ...
    ```
=== "MACSima"
    `data_path` is the path to the directory containing multiple `.ome.tif` files (one file per channel). In brief, you should have this file structure:

    ```txt
    .
    ├─ AAA.ome.tif
    ├─ BBB.ome.tif
    └─ CCC.ome.tif
    ```
=== "PhenoCycler"
    `data_path` is the path to one `.qptiff` file, or one `.tif` file (if exported from QuPath).
=== "Hyperion"
    `data_path` is path to the directory containing multiple `.ome.tiff` files (one file per channel). In brief, you should have this file structure:
    ```txt
    .
    ├─ AAA.ome.tiff
    ├─ BBB.ome.tiff
    └─ CCC.ome.tiff
    ```
=== "Others (CZI, ...)"
    Other file formats (ND2, CZI, LIF, or DV) are supported via the `bioio` reader. In that case, you'll need to add new dependencies: `pip install bioio` (and potentially some file-format specific dependencies, see [documentation](https://bioio-devs.github.io/bioio/OVERVIEW.html#reader-installation)).

    This reader is called `bioio`, i.e. you can use it via `sopa.io.bioio(data_path)`, where `data_path` is the path to your data file containing your image(s). For the Snakemake pipeline, provide `bioio` as a `technology` in the config file.

## How to disable the auto-save?

When using the API, and when your `SpatialData` object is saved on-disk, Sopa will automatically save any new spatial element on disk by default. To disable this behavior, use the following command from the API:

```python
sopa.settings.auto_save_on_disk = False
```

## How to parallelize segmentation?

Some steps of Sopa, notably the segmentation, can be accelerated via a parallelization backend. If you use the API, you can set the `"dask"` backend as below.

```python
# when using the API
sopa.settings.parallelization_backend = "dask"
```

Otherwise, if you don't use the API, you can also set the `SOPA_PARALLELIZATION_BACKEND` env variable, e.g.:
```sh
export SOPA_PARALLELIZATION_BACKEND=dask
```

!!! warning
    The `dask` backend is still experimental. You can add a comment to [this issue](https://github.com/prism-oncology/sopa/issues/145) to help us improve it.

You can also pass some kwargs to the [dask Client](https://distributed.dask.org/en/stable/api.html#client) as below. These kwargs are highly dependent of your cluster and the size of your patches. For "middle-size" patches, we recommend about 4GB of memory per worker for cellpose, and between 8GB and 16GB or memory for baysor.

```python
sopa.settings.dask_client_kwargs["n_workers"] = 4
```

For testing purposes, you can run the lines below, which will show you how many workers and memory you have by default:
```python
from dask.distributed import Client

client = Client()

n_workers = len(client.cluster.workers)
mem_worker0 = client.cluster.workers[0].memory_manager.memory_limit / 1024**3

print(f"{n_workers=}, {mem_worker0=:.3}GB")
```

## Which pipeline parameters should I use?

Some parameters such as the Cellpose diameter is crucial and depends highly on the resolution of your technology (pixel size).
As a guide, you can start from the parameters of the [config files](https://github.com/prism-oncology/sopa/tree/main/workflow/config) of your specific technology, and adjust them based on your knowledge of the tissue you work on.

Here are some parameters which are important to check: `diameter` (in pixels for cellpose), `min_area` (in pixels^2 for cellpose, in microns^2 for baysor), `scale` (in microns for baysor), `pixel_size` (for the Xenium Explorer conversion, to have the right scale during display).

## How to filter genes?

By default, we remove some genes names during segmentation and aggregation (for instance, `"blank"` or `"unassigned"` gene names). To change this behavior, you can update the gene pattern under `sopa.settings.gene_exclude_pattern` (this pattern is used by [`pandas.Series.str.match`](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.match.html)).

You can also decide not to remove some specific low quality transcripts for segmenation. To do that, create a (boolean) column called `"low_quality_transcript"` to your transcript dataframe. The rows whose value is `True` will not be used during segmentation. For instance, if you have Xenium data, you can filter the genes based on a QV value of 20.

```python
df = sdata["transcripts"]
df["low_quality_transcript"] = df.qv < 20
```

## How does Sopa know when using which elements?

Many functions of Sopa can run without any argument. For instance, `sopa.aggregate(sdata)` works for very different technologies, such as VisiumHD, MERSCOPE, or MACSima data.

Internally, when reading the raw data, Sopa saves some attributes inside `sdata.attrs`. These attributes are then used to know which spatial element corresponds to what. For instance, one H&E image may be tagged for tissue segmentation, while the DAPI image will be tagged to be used for cellpose.

Sopa handles this internally by default, and it is completely invisible to the user. But, to get a full control on what is done, you can always set some arguments to specify which element to be used. You can refer to the following arguments of the API: `image_key`, `points_key`, `shapes_key`, `bins_key`, and `table_key`.

## How to remove cells artifacts?

When segmenting a patch that is outside of the issue, Cellpose may "hallucinate" and generate some fake cells. To avoid that, you can run [`sopa.segmentation.tissue`](../api/segmentation/#sopa.segmentation.tissue) to ensure segmentation is always run inside the tissue.

Otherwise, if you have inside the tissue some small cells artefacts, `Sopa` offers three filtering approaches:

- Using a [min_area](../api/segmentation/#sopa.segmentation.cellpose) threshold to remove small cells (provide this argument to the segmentation methods).
- A [min_transcripts](/api/aggregation/#sopa.aggregate) threshold to remove cells with a low transcript count (provide this argument to the aggregation step).
- A [min_intensity_ratio](..//api/aggregation/#sopa.aggregate) value to remove cells with a low fluorescence intensity (provide this argument to the aggregation step).

## How to get more cells with Cellpose?

- The main Cellpose parameter to check is `diameter`, i.e. a typical cell diameter **in pixels**. Note that this is highly specific to the technology you're using since the micron-to-pixel ratio can differ. We advise you to start with the default parameter for your technology of interest (see the `diameter` parameter inside our config files [here](https://github.com/prism-oncology/sopa/tree/main/workflow/config)).
- Maybe `min_area` is too high, and all the cells are filtered because they are smaller than this area. Remind that, when using Cellpose, the areas correspond to pixels^2.
- This can be due to a low image quality. If the image is too pixelated, consider increasing `gaussian_sigma` (e.g., `2`) under the cellpose parameters of our config. If the image has a low contrast, consider increasing `clip_limit` (e.g., `0.3`). These parameters are detailed in [this example config](https://github.com/prism-oncology/sopa/blob/main/workflow/config/example_commented.yaml).
- Consider updating the official Cellpose parameters. In particular, try `cellprob_threshold=-6` and `flow_threshold=2`.

## How to use a custom Cellpose model?

You can use any existing [Cellpose model](https://cellpose.readthedocs.io/en/latest/models.html) with the `model_type` argument (via the API, CLI, or Snakemake pipeline). For the Snakemake pipeline, see [here](https://github.com/prism-oncology/sopa/blob/main/workflow/config/example_commented.yaml) how to set this argument.
If you have a custom pretrained model, use the `pretrained_model` argument instead of `model_type`, and give the path to your cellpose model.

## How to use the GPU for Cellpose?

You can provide `gpu=True` to `sopa.segmentation.cellpose`. This is recommended for `cellpose>=4.0.0`, which supports larger models and may be much faster on GPUs.

!!! Warning
    If you have many CPU cores and only one GPU, it may be faster to run in parallel on CPUs rather than sequentially using the GPU (mostly on `cellpose<4.0.0`). Also, if you are on MacOS, you may experience issues because the PyTorch MPS backend doesn't support all features yet.

```python
import sopa

sdata = sopa.io.toy_dataset()
sopa.make_image_patches(sdata)

sopa.segmentation.cellpose(sdata, channels="DAPI", diameter=30, gpu=True)
```

## How to provide other arguments to Cellpose?

When using the Snakemake pipeline, you can use `method_kwargs` to provide extra arguments to Cellpose. For instance, we use `resample=False` in the example below, which may significantly speed up the segmentation while not decreasing significantly the segmentation quality:

```yaml
segmentation:
  cellpose:
    diameter: 60
    channels: ["DAPI"]
    flow_threshold: 2
    cellprob_threshold: -6
    min_area: 2000
    method_kwargs:
      resample: False
```

## How to use a prior cell segmentation?

If you have MERSCOPE or Xenium data, you probably already have a cell segmentation. This can be used as a prior for Proseg or Baysor, instead of running Cellpose with Sopa. For that, you have an existing config file for the Snakemake pipeline for both [MERSCOPE](https://github.com/prism-oncology/sopa/blob/main/workflow/config/merscope/baysor_vizgen.yaml) and [Xenium](https://github.com/prism-oncology/sopa/blob/main/workflow/config/xenium/baysor_prior.yaml) data. If using the API/CLI, consider using the `prior_shapes_key` and the `unassigned_value` arguments when creating the patches for the transcripts. For MERSCOPE data, `prior_shapes_key="cell_id"` and `unassigned_value=-1`. For Xenium data, `prior_shapes_key="cell_id"` and `unassigned_value="UNASSIGNED"`. You can also decide to run Cellpose via Sopa, and then use it as a prior: in that case, simply pass `prior_shapes_key="cellpose_boundaries"` after running cellpose.

## How to optimize the segmentation parameters?

Selecting the right parameters for Cellpose/Baysor can significantly improve the output data quality. To choose these parameters, we recommend subsetting the data (`spatialdata.bounding_box_query`), saving the subset (`sdata.write("subset.zarr")`), running different segmentation on the subset (use `key_added` to save the segmentation with a specific name), and compare the results. Refer to [this tutorial](../tutorials/compare_segmentations) for an example.

## How to provide dictionnaries to CLI arguments?

Some CLI arguments are optionnal dictionnaries. For instance, `sopa convert` has a `--kwargs` option. In that case, a dictionnary can be provided as an inline string, for instance:

`--kwargs "{'backend': 'rioxarray'}"`

## How to fix an "out-of-memory" issue on MERSCOPE data?

If using MERSCOPE data, images can be huge. To improve RAM efficiency, you can install `rioxarray` (`pip install rioxarray`). Then, the `rioxarray` will be used by default by the reader (no change needed, it will be detected automatically).

## How to remove the logs?

You can change the level of `logging` for sopa, e.g. you can run the lines below to set the logging level to show only errors:

```python
import sopa

sopa.log.setLevel(sopa.logging.ERROR)
```

## How to ask for help?

If you have an issue that is not detailed in this FAQ, you can still open an issue on [Sopa's Github repository](https://github.com/prism-oncology/sopa/issues), and detail your issue with as much precision as possible for the maintainers to be able to reproduce it.

Make sure to have a quick look to the existing issues, maybe someone faced the same problem.
