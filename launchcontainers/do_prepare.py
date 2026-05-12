# """
# MIT License
# Copyright (c) 2020-2025 Garikoitz Lerma-Usabiaga
# Copyright (c) 2020-2022 Mengxing Liu
# Copyright (c) 2022-2023 Leandro Lecca
# Copyright (c) 2022-2025 Yongning Lei
# Copyright (c) 2023 David Linhardt
# Copyright (c) 2023 Iñigo Tellaetxe
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to
# whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# """
from __future__ import annotations

import os
import os.path as op

from bids import BIDSLayout

from launchcontainers import utils as do
from launchcontainers.log_setup import console
from launchcontainers.prepare import dwi_prepare as dwi_prepare
from launchcontainers.prepare.glm_prepare import run_glm_prepare

_GLM_PIPELINES = {"l1_surface"}
_DWI_PIPELINES = {
    "anatrois",
    "rtppreproc",
    "rtp-pipeline",
    "freesurferator",
    "rtp2-preproc",
    "rtp2-pipeline",
}


def _create_analysis_dir(lc_config: dict) -> str:
    """
    Create the derivative container/analysis directory for DWI pipelines.

    Parameters
    ----------
    lc_config : dict
        Parsed launchcontainers YAML configuration.

    Returns
    -------
    str
        Absolute path to the analysis directory.
    """
    basedir = lc_config["general"]["basedir"]
    bidsdir_name = lc_config["general"]["bidsdir_name"]
    deriv_layout = lc_config["general"]["deriv_layout"]
    container = lc_config["general"]["container"]
    analysis_name = lc_config["general"]["analysis_name"]

    if container == "l1_surface":
        analysis_dir = op.join(
            basedir, bidsdir_name, "derivatives", "l1_surface", f"analysis-{analysis_name}"
        )
    elif deriv_layout == "legacy":
        version = lc_config["container_specific"][container]["version"]
        container_folder = op.join(
            basedir, bidsdir_name, "derivatives", f"{container}_{version}"
        )
        analysis_dir = op.join(container_folder, f"analysis-{analysis_name}")
    else:
        version = lc_config["container_specific"][container]["version"]
        analysis_dir = op.join(
            basedir,
            bidsdir_name,
            "derivatives",
            f"{container}-{version}_{analysis_name}",
        )

    os.makedirs(analysis_dir, exist_ok=True)
    console.print(
        f"Container layout is {deriv_layout}, analysis dir: {analysis_dir}",
        style="blue",
    )
    return analysis_dir


def _prepare_analysis_dir(parse_namespace, analysis_dir: str, lc_config: dict):
    """Copy lc_config, subseslist, and (optionally) container-specific config into analysis_dir."""
    container = lc_config["general"]["container"]
    force = lc_config["general"]["force"]

    do.copy_file(
        parse_namespace.lc_config, op.join(analysis_dir, "lc_config.yaml"), force
    )
    do.copy_file(
        parse_namespace.sub_ses_list, op.join(analysis_dir, "subseslist.txt"), force
    )
    if parse_namespace.container_specific_config is not None:
        do.copy_file(
            parse_namespace.container_specific_config,
            op.join(analysis_dir, f"{container}.json"),
            force,
        )
    console.print(
        f"\n The analysis folder: {analysis_dir} successfully created, all configs copied",
        style="green",
    )


def _chmod777(path: str) -> None:
    """Recursively set permissions to 777 on *path* so other users can access it."""
    console.print(f"Setting permissions 777 on {path}", style="blue")
    for dirpath, dirnames, filenames in os.walk(path):
        os.chmod(dirpath, 0o777)
        for fname in filenames:
            os.chmod(op.join(dirpath, fname), 0o777)


def main(parse_namespace) -> tuple[bool, str | None]:
    """
    Run the full prepare workflow.

    For both DWI and GLM pipelines, creates an analysis directory under
    ``BIDS/derivatives/fMRI-GLM-{version}/analysis-{analysis_name}`` (legacy)
    or ``fMRI-GLM-{version}_{analysis_name}`` (new layout), copies configs into
    it, and delegates to the appropriate prepare helpers.

    If the ``container_specific.<container>`` section is absent for GLM, an
    example config is written and the function returns early.

    Parameters
    ----------
    parse_namespace : argparse.Namespace
        Parsed CLI arguments (``lc_config``, ``sub_ses_list``,
        ``container_specific_config``).

    Returns
    -------
    tuple[bool, str | None]
        ``(success, analysis_dir)`` where *analysis_dir* is ``None`` only
        when the GLM config section is missing.
    """
    lc_config = do.read_yaml(parse_namespace.lc_config)
    container = lc_config["general"]["container"]
    basedir = lc_config["general"]["basedir"]
    bidsdir_name = lc_config["general"]["bidsdir_name"]

    sub_ses_list_path = parse_namespace.sub_ses_list
    df_subses = do.parse_subses_list(sub_ses_list_path)

    if container in _DWI_PIPELINES:
        analysis_dir = _create_analysis_dir(lc_config)
        _prepare_analysis_dir(parse_namespace, analysis_dir, lc_config)

        console.print("Reading the BIDS layout...", style="blue")
        layout = BIDSLayout(os.path.join(basedir, bidsdir_name), validate=False)
        console.print("Finished reading the BIDS layout.", style="green")

        console.print(f"{container}: running RTP2 prepare", style="dim")
        success = dwi_prepare.main(parse_namespace, analysis_dir, df_subses, layout)
        console.print(
            f"\n #####\n \U0001f37a Analysis dir is \n{analysis_dir}\n",
            style="bold red",
        )
        _chmod777(analysis_dir)
        return success, analysis_dir

    elif container in _GLM_PIPELINES:
        # Early exit: if container_specific section is empty, write example config
        if not lc_config.get("container_specific", {}).get(container):
            console.print(
                f"\n### No container_specific.{container} section found — writing example config.",
                style="yellow",
            )
            from launchcontainers.prepare.glm_prepare import GLMPrepare

            GLMPrepare.write_example_config()
            return False, None

        analysis_dir = _create_analysis_dir(lc_config)
        _prepare_analysis_dir(parse_namespace, analysis_dir, lc_config)

        console.print("Reading the BIDS layout...", style="blue")
        layout = BIDSLayout(os.path.join(basedir, bidsdir_name), validate=False)
        console.print("Finished reading the BIDS layout.", style="green")

        console.print(f"{container}: running GLM prepare", style="dim")
        success = run_glm_prepare(lc_config, df_subses, layout)
        console.print(
            f"\n #####\n \U0001f37a Analysis dir is \n{analysis_dir}\n",
            style="bold red",
        )
        _chmod777(analysis_dir)
        return success, analysis_dir

    else:
        console.print(f"{container} is not in the list", style="red")
        return False, None
