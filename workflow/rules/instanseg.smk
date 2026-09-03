rule patch_segmentation_instanseg:
    input:
        paths.smk_patches_file_image,
        paths.smk_patches,
    output:
        paths.temp_dir("instanseg") / "{index}.parquet",
    conda:
        "sopa"
    params:
        instanseg = args["segmentation"]["instanseg"].as_cli(),
        sdata_path = paths.sdata_path,
    shell:
        """
        sopa segmentation instanseg {params.sdata_path} --patch-index {wildcards.index} {params.instanseg}
        """


rule resolve_instanseg:
    input:
        get_input_resolve("image", "instanseg"),
    output:
        touch(paths.segmentation_done("instanseg")),
    conda:
        "sopa"
    params:
        sdata_path = paths.sdata_path,
    shell:
        """
        sopa resolve instanseg {params.sdata_path}
        """
