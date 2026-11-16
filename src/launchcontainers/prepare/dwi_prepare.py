"""
MIT License

Copyright (c) 2020-2023 Garikoitz Lerma-Usabiaga
Copyright (c) 2020-2022 Mengxing Liu
Copyright (c) 2022-2024 Leandro Lecca
Copyright (c) 2022-2023 Yongning Lei
Copyright (c) 2023 David Linhardt
Copyright (c) 2023 Iñigo Tellaetxe

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.
"""

from __future__ import annotations

import json
import os
import os.path as op

from launchcontainers import utils as do
from launchcontainers.log_setup import console
from launchcontainers.prepare import RTP2_prepare_input as prepare_input


def copy_rtp2_configs(container, extra_config_fpath, analysis_dir, force, option=None):
    """
    Copy an auxiliary container input file into the analysis directory.

    Parameters
    ----------
    container : str
        Active container name.
    extra_config_fpath : str
        Source path for the auxiliary file.
    analysis_dir : str
        Analysis directory that should receive the copied file.
    force : bool
        If ``True``, overwrite an existing destination file.
    option : str, optional
        Additional selector used by pipeline containers to distinguish between
        ``tractparams`` and ``fsmask`` inputs.

    Returns
    -------
    str
        Basename of the copied file.

    Raises
    ------
    ValueError
        If the file type is not supported for the selected container/option.
    """
    if os.path.isfile(extra_config_fpath):
        console.print(
            "\n" + f" We will pass  {extra_config_fpath} to {container} analysis dir",
            style="cyan",
        )
    else:
        console.print(f"\n{extra_config_fpath} does not exist", style="red")

    config_fname = os.path.basename(extra_config_fpath)
    file_suffix = os.path.splitext(config_fname)[1]

    src_fpath = extra_config_fpath
    dst_fpath = os.path.join(analysis_dir, config_fname)
    if container in ["anatrois", "freesurferator"]:
        if file_suffix in [".nii", ".gz", ".zip"]:
            do.copy_file(src_fpath, dst_fpath, force)
        else:
            raise ValueError("Unsupported file type.")

    if container in ["rtp2-pipeline", "rtp-pipeline"]:
        if option == "tractparams":
            if file_suffix in [".csv"]:
                do.copy_file(src_fpath, dst_fpath, force)
            else:
                raise ValueError("Unsupported file type.")
        if option == "fsmask":
            if file_suffix in [".nii", ".gz"]:
                do.copy_file(src_fpath, dst_fpath, force)
            else:
                raise ValueError("Unsupported file type.")
    return config_fname


def gen_config_dict_and_copy(parser_namespace, analysis_dir):
    """
    Build the container input mapping and copy auxiliary files when needed.

    The returned dictionary mirrors the ``inputs`` structure expected by the
    container ``config.json`` file and includes any optional files copied into
    the analysis directory, such as annotation files, ROI archives, tract
    parameter tables, masks, or qmap archives.

    Parameters
    ----------
    parser_namespace : argparse.Namespace
        Parsed CLI arguments for prepare mode.
    analysis_dir : str
        Prepared analysis directory.

    Returns
    -------
    dict
        Dictionary describing the container input files and the path to the
        copied container JSON under ``config_path``.
    """
    # read the yaml to get input info
    lc_config_fpath = parser_namespace.lc_config
    lc_config = do.read_yaml(lc_config_fpath)
    console.print("\n prepare_dwi_extra_configs reading lc config yaml", style="cyan")
    # read parameters from lc_config
    container = lc_config["general"]["container"]
    force = lc_config["general"]["force"]

    container_configs_fname = f"{container}.json"
    json_under_analysis_dir = op.join(analysis_dir, container_configs_fname)
    # set up dict to get extra config infomation
    config_json_dict = {}
    config_json_dict["config_path"] = json_under_analysis_dir

    # store the json content into dict and copy the extract configs to analysis folder
    if container in ["anatrois"]:
        config_json_dict[container] = {}
        pre_fs = lc_config["container_specific"][container]["pre_fs"]
        annotfile = lc_config["container_specific"][container]["annotfile"]
        mniroizip = lc_config["container_specific"][container]["mniroizip"]
        if pre_fs:
            config_fname = "existingFS.zip"
            config_json_dict[container]["pre_fs"] = f"pre_fs/{config_fname}"
        else:
            config_fname = "T1.nii.gz"
            config_json_dict[container]["anat"] = f"anat/{config_fname}"
        if annotfile:
            config_fname = copy_rtp2_configs(container, annotfile, analysis_dir, force)

            config_json_dict[container]["annotfile"] = f"annotfile/{config_fname}"
        if mniroizip:
            config_fname = copy_rtp2_configs(container, mniroizip, analysis_dir, force)
            config_json_dict[container]["mniroizip"] = f"mniroizip/{config_fname}"

    if container in ["freesurferator"]:
        config_json_dict[container] = {}
        pre_fs = lc_config["container_specific"][container]["pre_fs"]
        control_points = lc_config["container_specific"][container]["control_points"]
        annotfile = lc_config["container_specific"][container]["annotfile"]
        mniroizip = lc_config["container_specific"][container]["mniroizip"]
        if pre_fs:
            config_fname = "existingFS.zip"
            config_json_dict[container]["pre_fs"] = f"pre_fs/{config_fname}"
        else:
            config_fname = "T1.nii.gz"
            config_json_dict[container]["anat"] = f"anat/{config_fname}"
        if control_points:
            config_fname = "control.dat"
            config_json_dict[container]["control_points"] = (
                f"control_points/{config_fname}"
            )
        if annotfile:
            config_fname = copy_rtp2_configs(container, annotfile, analysis_dir, force)
            config_json_dict[container]["annotfile"] = f"annotfile/{config_fname}"
        if mniroizip:
            config_fname = copy_rtp2_configs(container, mniroizip, analysis_dir, force)
            config_json_dict[container]["mniroizip"] = f"mniroizip/{config_fname}"

    if container in ["rtppreproc"]:
        preproc_json_keys = ["ANAT", "BVAL", "BVEC", "DIFF", "FSMASK"]
        preproc_json_val = [
            "ANAT/T1.nii.gz",
            "BVAL/dwiF.bval",
            "BVEC/dwiF.bvec",
            "DIFF/dwiF.nii.gz",
            "FSMASK/brain.nii.gz",
        ]
        config_json_dict[container] = {
            key: value for key, value in zip(preproc_json_keys, preproc_json_val)
        }
        rpe = lc_config["container_specific"][container]["rpe"]
        if rpe:
            config_json_dict[container]["RBVC"] = "RBVC/dwiR.bvec"
            config_json_dict[container]["RBVL"] = "RBVL/dwiR.bval"
            config_json_dict[container]["RDIF"] = "RDIF/dwiR.nii.gz"

    if container in ["rtp2-preproc"]:
        preproc_json_keys = ["ANAT", "BVAL", "BVEC", "DIFF", "FSMASK"]
        preproc_json_val = [
            "ANAT/T1.nii.gz",
            "BVAL/dwiF.bval",
            "BVEC/dwiF.bvec",
            "DIFF/dwiF.nii.gz",
            "FSMASK/brain.nii.gz",
        ]
        config_json_dict[container] = {
            key: value for key, value in zip(preproc_json_keys, preproc_json_val)
        }

        rpe = lc_config["container_specific"][container]["rpe"]
        use_qmap = lc_config["container_specific"][container]["use_qmap"]
        if rpe:
            config_json_dict[container]["RBVC"] = "RBVC/dwiR.bvec"
            config_json_dict[container]["RBVL"] = "RBVL/dwiR.bval"
            config_json_dict[container]["RDIF"] = "RDIF/dwiR.nii.gz"
        if use_qmap:
            config_fname = "qmap.zip"
            config_json_dict[container]["qmap"] = f"qmap/{config_fname}"

    if container in ["rtp-pipeline"]:
        pipeline_json_keys = ["anatomical", "bval", "bvec", "dwi", "fs"]
        pipeline_json_val = [
            "anatomical/T1.nii.gz",
            "bval/dwi.bval",
            "bvec/dwi.bvec",
            "dwi/dwi.nii.gz",
            "fs/fs.zip",
        ]
        config_json_dict[container] = {
            key: value for key, value in zip(pipeline_json_keys, pipeline_json_val)
        }

        tractparams = lc_config["container_specific"][container]["tractparams"]
        if tractparams:
            config_fname = copy_rtp2_configs(
                container,
                tractparams,
                analysis_dir,
                force,
                "tractparams",
            )
            config_json_dict[container]["tractparams"] = f"tractparams/{config_fname}"

    if container in ["rtp2-pipeline"]:
        pipeline_json_keys = ["anatomical", "bval", "bvec", "dwi", "fs"]
        pipeline_json_val = [
            "anatomical/T1.nii.gz",
            "bval/dwi.bval",
            "bvec/dwi.bvec",
            "dwi/dwi.nii.gz",
            "fs/fs.zip",
        ]
        config_json_dict[container] = {
            key: value for key, value in zip(pipeline_json_keys, pipeline_json_val)
        }
        tractparams = lc_config["container_specific"][container]["tractparams"]
        fsmask = lc_config["container_specific"][container]["fsmask"]
        use_qmap = lc_config["container_specific"][container]["use_qmap"]
        if tractparams:
            config_fname = copy_rtp2_configs(
                container,
                tractparams,
                analysis_dir,
                force,
                "tractparams",
            )
            config_json_dict[container]["tractparams"] = f"tractparams/{config_fname}"
        if fsmask:
            config_fname = copy_rtp2_configs(
                container, fsmask, analysis_dir, force, "fsmask"
            )
            config_json_dict[container]["fsmask"] = f"fsmask/{config_fname}"
        if use_qmap:
            config_fname = "qmap.zip"
            config_json_dict[container]["qmap"] = f"qmap/{config_fname}"

    return config_json_dict


def gen_config_json_extra_field(container, config_json_dict):
    """
    Convert an internal config mapping into the container ``inputs`` format.

    Parameters
    ----------
    container : str
        Active container name.
    config_json_dict : dict
        Internal mapping returned by :func:`gen_config_dict_and_copy`.

    Returns
    -------
    dict
        ``inputs`` block ready to be injected into the container JSON file.
    """

    config_dict = config_json_dict[container]

    extra_field_config_json = {}
    for key in config_dict.keys():
        extra_field_config_json[key] = {
            "location": {
                "path": op.join("/flywheel/v0/input", config_dict[key]),
                "name": op.basename(config_dict[key]),
            },
            "base": "file",
        }
    # this is check for anatrois and freesurferator
    if (
        "anat" in extra_field_config_json.keys()
        and "pre_fs" in extra_field_config_json.keys()
    ):
        del extra_field_config_json["anat"]

    return extra_field_config_json


def write_json(extra_field_config_json, json_path, force):
    """
    Insert or update the ``inputs`` field of a container JSON file.

    Parameters
    ----------
    extra_field_config_json : dict
        Prepared ``inputs`` block to write.
    json_path : str or path-like
        Path to the container JSON file to modify.
    force : bool
        If ``True``, overwrite an existing ``inputs`` block.

    Returns
    -------
    bool
        Always returns ``True`` after writing the JSON file back to disk.
    """
    # 1) Load safely
    with open(json_path) as infile:
        config = json.load(infile)

    # 2) Decide whether to set/overwrite
    if "inputs" not in config:
        console.print(f"'inputs' field missing; adding to {json_path}", style="cyan")
        config["inputs"] = extra_field_config_json

    elif force:
        console.print(
            f"'inputs' already exists in {json_path}; overwriting because --force=True",
            style="cyan",
        )
        config["inputs"] = extra_field_config_json

    else:
        console.print(
            f"'{json_path}' already has an 'inputs' field; use --force to overwrite",
            style="yellow",
        )
        # No change to config

    # 3) Write back (only really changed if we set above)
    with open(json_path, "w") as outfile:
        json.dump(config, outfile, indent=4)

    return True


def copy_and_edit_config_json(parser_namespace, analysis_dir):
    """
    Prepare the container JSON file stored under the analysis directory.

    Parameters
    ----------
    parser_namespace : argparse.Namespace
        Parsed CLI arguments for prepare mode.
    analysis_dir : str
        Prepared analysis directory.

    Returns
    -------
    dict
        Internal mapping of copied inputs and the rewritten JSON path.
    """

    # get the config json dict and copy the extra configs
    config_json_dict = gen_config_dict_and_copy(parser_namespace, analysis_dir)

    # read the yaml to get input info
    lc_config_fpath = parser_namespace.lc_config
    lc_config = lc_config = do.read_yaml(lc_config_fpath)
    console.print("\n copy_and_edit_config_json reading lc config yaml", style="cyan")
    # read parameters from lc_config
    container = lc_config["general"]["container"]
    force = lc_config["general"]["force"]

    # get the path of the json under analysis dir
    json_under_analysis_dir = config_json_dict["config_path"]

    # get the extra field for config json
    extra_field_config_json = gen_config_json_extra_field(container, config_json_dict)

    json_under_analysis_dir = config_json_dict["config_path"]

    if write_json(extra_field_config_json, json_under_analysis_dir, force):
        console.print(f"Successfully write json for {container}", style="cyan")

    return config_json_dict


def main(parser_namespace, analysis_dir, df_subses, layout):
    """
    Prepare analysis-level and session-level inputs for DWI containers.

    Parameters
    ----------
    parser_namespace : argparse.Namespace
        Parsed CLI arguments for prepare mode.
    analysis_dir : str
        Prepared analysis directory.
    df_subses : list[tuple[str, str]]
        Filtered subject/session pairs to process.
    layout : bids.BIDSLayout
        BIDS layout built from the configured raw dataset.

    Returns
    -------
    bool
        ``True`` when the container JSON has been updated and all requested
        session inputs have been prepared.
    """
    # read the yaml to get input info
    lc_config_fpath = parser_namespace.lc_config
    lc_config = lc_config = do.read_yaml(lc_config_fpath)
    console.print("\n prepare_dwi_input_folder reading lc config yaml", style="cyan")

    container = lc_config["general"]["container"]
    force = lc_config["general"]["force"]
    version = lc_config["container_specific"][container]["version"]

    # read parameters from lc_config
    container = lc_config["general"]["container"]
    force = lc_config["general"]["force"]
    version = lc_config["container_specific"][container]["version"]

    console.print(
        "#####################################################\n"
        + "Preparing for DWI pipeline RTP2",
        style="cyan",
    )

    # copy and edit config json and extra config files
    config_json_dict = copy_and_edit_config_json(parser_namespace, analysis_dir)

    if config_json_dict:
        console.print(
            "#####################################################\n"
            + "Successfully copy extra confings and rewrite the json\n",
            style="cyan",
        )
    else:
        console.print(
            "\n"
            + "#####################################################\n"
            + "Prepare json not finished. Please check\n",
            style="red",
        )
        raise Exception(
            "Sorry the Json file seems not being written correctly, \
                it may cause container dysfunction",
        )

    console.print(
        "\n"
        + "#####################################################\n"
        + f"DWI Create the symlinks of all the input files RTP2-{container}\n",
        style="cyan",
    )

    # anatrois/freesurferator retest sessions are symlinks to the src session,
    # so every src session must be staged before the sessions that link to it.
    use_src_session = None
    if container in ["anatrois", "freesurferator"]:
        use_src_session = lc_config["container_specific"][container].get(
            "use_src_session"
        )
        if use_src_session is not None:
            use_src_session = str(use_src_session).removeprefix("ses-")
            if use_src_session.isdigit():
                use_src_session = use_src_session.zfill(2)
            df_subses = sorted(df_subses, key=lambda p: p[1] != use_src_session)

    failed = []
    for sub, ses in df_subses:
        try:
            _prepare_one_session(
                parser_namespace,
                analysis_dir,
                lc_config,
                config_json_dict,
                layout,
                container,
                version,
                force,
                use_src_session,
                sub,
                ses,
            )
        except Exception as e:
            console.print(
                f"FAILED sub-{sub}_ses-{ses}: {type(e).__name__}: {e}", style="red"
            )
            failed.append((sub, ses))

    if failed:
        console.print(
            f"{len(failed)} of {len(df_subses)} session(s) failed to prepare: "
            + ", ".join(f"sub-{s}_ses-{e}" for s, e in failed),
            style="red",
        )
        return False
    return True


def _prepare_one_session(
    parser_namespace,
    analysis_dir,
    lc_config,
    config_json_dict,
    layout,
    container,
    version,
    force,
    use_src_session,
    sub,
    ses,
):
    """Stage the inputs of one sub/ses; raise on any failure."""
    console.print(
        f"The current ses is: sub-{sub}_ses-{ses}_{container}_{version}",
        style="bold cyan",
    )

    tmpdir = op.join(
        analysis_dir,
        "sub-" + sub,
        "ses-" + ses,
        "output",
        "tmp",
    )
    # Tiger: for now, the log dir for container is under output folder,
    # mainly bc RTP will wrote RTP.txt to output/log
    # don't change this
    container_logdir = op.join(
        analysis_dir,
        "sub-" + sub,
        "ses-" + ses,
        "output",
        "log",
    )
    # A retest session of anatrois/freesurferator is a symlink to the src
    # session (made in prepare_input.anatrois); it gets no dirs or configs
    # of its own. Everything else gets output/tmp and output/log.
    is_retest = use_src_session is not None and ses != use_src_session
    if not is_retest:
        os.makedirs(tmpdir, exist_ok=True)
        os.makedirs(container_logdir, exist_ok=True)
        do.copy_file(
            parser_namespace.lc_config,
            op.join(container_logdir, "lc_config.yaml"),
            force,
        )
        do.copy_file(
            config_json_dict["config_path"],
            op.join(container_logdir, "config.json"),
            force,
        )

    if container in ["rtppreproc", "rtp2-preproc"]:
        prepare_input.rtppreproc(
            config_json_dict,
            analysis_dir,
            lc_config,
            sub,
            ses,
            layout,
        )
    elif container in ["rtp-pipeline", "rtp2-pipeline"]:
        prepare_input.rtppipeline(
            config_json_dict,
            analysis_dir,
            lc_config,
            sub,
            ses,
        )
    elif container in ["anatrois", "freesurferator"]:
        prepare_input.anatrois(
            config_json_dict,
            analysis_dir,
            lc_config,
            sub,
            ses,
            layout,
        )
    else:
        raise ValueError(
            f"{container} is not supported, check for typos or "
            "contact admin for singularity images"
        )
