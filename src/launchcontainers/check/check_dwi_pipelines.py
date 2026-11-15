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
import zipfile
from datetime import datetime

from launchcontainers.log_setup import console, log_info
from launchcontainers.utils import resolve_deriv_dir


def check_tractparam(lc_config, sub, ses, tractparam_df):
    """
    Verify that all ROIs referenced in ``tractparams`` exist in ``fs.zip``.

    Parameters
    ----------
    lc_config : dict
        Parsed launchcontainers YAML configuration.
    sub : str
        Subject identifier without the ``sub-`` prefix.
    ses : str
        Session identifier without the ``ses-`` prefix.
    tractparam_df : pandas.DataFrame
        Tract parameter table used by ``rtp-pipeline`` or ``rtp2-pipeline``.

    Returns
    -------
    bool
        ``True`` if every required ROI file is present in the anatomical
        derivative ``fs.zip`` archive.

    Raises
    ------
    FileNotFoundError
        If one or more required ROI files are missing.
    """
    # Define the list of required ROIs
    console.print(
        "\n" + "#####################################################\n",
        style="bold red",
    )
    roi_list = []
    # Iterate over some defined roisand check if they are required or not in the config.yaml
    for col in ["roi1", "roi2", "roi3", "roi4", "roiexc1", "roiexc2"]:
        for val in tractparam_df[col][~tractparam_df[col].isna()]:
            if "_AND_" in val:
                multi_roi = val.split("_AND_")
                roi_list.extend(multi_roi)
            else:
                if val != "NO":
                    roi_list.append(val)

    required_rois = set(roi_list)

    # Define the zip file
    container = lc_config["general"]["container"]
    precontainer_anat = lc_config["container_specific"][container]["precontainer_anat"]
    anat_analysis_name = lc_config["container_specific"][container][
        "anat_analysis_name"
    ]

    # Define where the fs.zip file is
    fs_zip = op.join(
        resolve_deriv_dir(lc_config, precontainer_anat, anat_analysis_name),
        "sub-" + sub,
        "ses-" + ses,
        "output",
        "fs.zip",
    )

    # Extract .gz files from zip file and check if they are all present
    with zipfile.ZipFile(fs_zip, "r") as zip:
        zip_gz_files = set(zip.namelist())

    # See which ROIs are present in the fs.zip file
    required_gz_files = {f"fs/ROIs/{file}.nii.gz" for file in required_rois}
    console.print(
        "\n"
        + f"---The following are the ROIs in fs.zip file: \n {zip_gz_files} \n"
        + f"---there are {len(zip_gz_files)} .nii.gz files in fs.zip from anatrois output\n"
        + f"---There are {len(required_gz_files)} ROIs that are required to run RTP-PIPELINE\n",
        style="cyan",
    )
    if required_gz_files.issubset(zip_gz_files):
        console.print(
            "\n" + "---checked! All required .gz files are present in the fs.zip \n",
            style="green",
        )
    else:
        missing_files = required_gz_files - zip_gz_files
        console.print(
            "\n"
            + "*****Error: \n"
            + f"there are {len(missing_files)} missed in fs.zip \n"
            + f"The following .gz files are missing in the zip file:\n {missing_files}",
            style="red",
        )
        raise FileNotFoundError("Required .gz file are missing")

    ROIs_are_there = required_gz_files.issubset(zip_gz_files)
    console.print(
        "\n" + "#####################################################\n",
        style="bold red",
    )
    return ROIs_are_there


def check_dwi_analysis_folder(parse_namespace, container):
    """
    Validate the analysis-level configuration files for a prepared run.

    Parameters
    ----------
    parse_namespace : argparse.Namespace
        Parsed CLI arguments for run mode.
    container : str
        Container name used to resolve the expected JSON filename.

    Returns
    -------
    bool
        ``True`` if the expected analysis-level configuration files exist.

    Raises
    ------
    FileNotFoundError
        If any required analysis-level configuration file is missing.
    """
    # analysis dir path
    analysis_dir = parse_namespace.workdir

    # for the most general 3 things: -lcc -ssl -cc
    ana_dir_lcc = op.join(analysis_dir, "lc_config.yaml")
    ana_dir_ssl = op.join(analysis_dir, "subseslist.txt")
    container_configs_fname = f"{container}.json"
    ana_dir_cc = op.join(analysis_dir, container_configs_fname)

    copies = [
        ana_dir_lcc,
        ana_dir_ssl,
        ana_dir_cc,
    ]

    general_config_present = all(op.isfile(copy_path) for copy_path in copies)

    if general_config_present:
        console.print(
            f"\n### Analysis folder {analysis_dir} is having all the general configs\n"
            + "Pass to next step",
            style="bold red",
        )
    else:
        console.print(
            "\n Did NOT detect back up configs in the analysis folder, \
                Please check then continue the run mode",
            style="red",
        )
        raise FileNotFoundError("Not all the 3 configs is under analysis dir, aborting")

    # 1) Load safely
    with open(ana_dir_cc) as infile:
        config = json.load(infile)

    # 2) check if inputs in the json
    if "inputs" not in config:
        console.print(
            f"ERROR: 'inputs' field is missing in {ana_dir_cc}\n"
            "Run 'lc prepare' first to populate the inputs field.",
            style="bold red",
        )
        raise ValueError(
            f"'inputs' field missing in {ana_dir_cc}. Run 'lc prepare' first."
        )

    return general_config_present


def backup_old_rtp2pipeline_log(
    parse_namespace,
    df_subses,
):
    """
    Rename pre-existing RTP log files before launching a new run.

    Parameters
    ----------
    parse_namespace : argparse.Namespace
        Parsed CLI arguments for run mode.
    df_subses : list[tuple[str, str]]
        Subject/session pairs whose output directories should be inspected.
    """
    # read LC config yml from analysis dir
    analysis_dir = parse_namespace.workdir
    for sub, ses in df_subses:
        subses_dir_output = op.join(
            analysis_dir,
            "sub-" + sub,
            "ses-" + ses,
            "output",
        )

        # check if there is old RTP
        old_rtp_log = os.path.join(
            subses_dir_output,
            "log",
            "RTP_log.txt",
        )
        if os.path.exists(old_rtp_log):
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            new_name = old_rtp_log.replace("_log", f"_log_backup_at_{now}")
            os.rename(old_rtp_log, new_name)
        else:
            log_info("\n no previous RTP, will run ")

    return
