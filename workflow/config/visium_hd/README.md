# Visium HD usage

For Visium HD, the full-resolution microscopy image (not the cytassist image) is required by Sopa as we’ll run cell segmentation on the H&E full-resolution slide. Therefore, you'll need to also pass `fullres_image_file=/path/to_fullres_image` to the config in the command line:

```sh
snakemake \
    --config data_path=/path/to/directory fullres_image_file=/path/to_fullres_image \
    --configfile=config/visium_hd/stardist.yaml \
    --workflow-profile profile/slurm  # or any profile you want
```

More details about the rest of the command [in the docs](https://prism-oncology.github.io/sopa/tutorials/snakemake/#run-snakemake).
