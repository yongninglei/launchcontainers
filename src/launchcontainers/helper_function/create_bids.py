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
from pathlib import Path

from heudiconv import bids as hb

from launchcontainers import cli as lc_parser
from launchcontainers import utils as do
from launchcontainers.log_setup import console


def main():
    """
    Create a fake BIDS tree and matching derivative skeleton for testing.

    The helper reads a small configuration YAML plus a subject/session list and
    creates empty files and folders that mimic the inputs expected by the rest
    of the package.
    """
    parser_namespace, parse_dict = lc_parser.get_parser()

    # Check if download_configs argument is provided

    console.print(
        "You are creating a fake BIDS folder structure based on "
        "your input basedir, and subseslist",
    )
    # Your main function logic here
    # e.g., launch_container(args.other_arg)
    # read yaml and setup the bids folder

    newcontainer_config_path = parser_namespace.creat_bids_config
    newcontainer_config = do.read_yaml(newcontainer_config_path)

    # Get general information from the config.yaml file
    basedir = newcontainer_config["general"]["basedir"]
    bidsdir_name = newcontainer_config["general"]["bidsdir_name"]
    container = newcontainer_config["general"]["container"]
    version = newcontainer_config["general"]["version"]
    analysis_name = newcontainer_config["general"]["analysis_name"]
    file_name = newcontainer_config["general"]["file_name"]
    log_dir = newcontainer_config["general"]["log_dir"]
    log_filename = newcontainer_config["general"]["log_filename"]

    # get stuff from subseslist for future jobs scheduling
    sub_ses_list_path = parser_namespace.sub_ses_list
    sub_ses_list = do.parse_subses_list(sub_ses_list_path)

    bids_dir = op.join(
        basedir,
        bidsdir_name,
    )
    # create the BIDS dir
    os.makedirs(bids_dir, exist_ok=True)
    # create the empty things under BIDS dir
    hb.populate_bids_templates(bids_dir)

    if version:
        analysis_dir = op.join(
            basedir,
            bidsdir_name,
            "derivatives",
            f"{container}-v_{version}_analysis_{analysis_name}",
        )
    else:
        analysis_dir = op.join(
            basedir,
            bidsdir_name,
            "derivatives",
            f"{container}-analysis_{analysis_name}",
        )
    os.makedirs(analysis_dir, exist_ok=True)
    hb.populate_bids_templates(analysis_dir)
    if log_dir == "analysis_dir":
        log_dir = analysis_dir
    os.makedirs(log_dir, exist_ok=True)
    console_file = open(op.join(log_dir, f"{log_filename}.txt"), "w")
    console.file = console_file

    # figure out a way to copy the bids componement to the corresponding bids folder
    for sub, ses in sub_ses_list:
        bids_dir_subses = op.join(
            bids_dir,
            f"sub-{sub}",
            f"ses-{ses}",
        )

        os.makedirs(bids_dir_subses, exist_ok=True)
        os.makedirs(op.join(bids_dir_subses, "anat"), exist_ok=True)
        fake_T1w = op.join(bids_dir_subses, "anat", f"sub-{sub}_ses-{ses}_T1w.nii.gz")
        if not Path(fake_T1w).is_file():
            Path(fake_T1w).touch()
        os.makedirs(op.join(bids_dir_subses, "func"), exist_ok=True)
        fake_bold = op.join(bids_dir_subses, "func", f"sub-{sub}_ses-{ses}_bold.nii.gz")
        fake_bold_json = op.join(
            bids_dir_subses, "func", f"sub-{sub}_ses-{ses}_bold.json"
        )
        if not Path(fake_bold).is_file():
            Path(fake_bold).touch()
        if not Path(fake_bold_json).is_file():
            Path(fake_bold_json).touch()
        os.makedirs(op.join(bids_dir_subses, "dwi"), exist_ok=True)
        fake_dwi = op.join(
            bids_dir_subses, "dwi", f"sub-{sub}_ses-{ses}_dir-AP_dwi.nii.gz"
        )
        fake_dwi_json = op.join(
            bids_dir_subses, "dwi", f"sub-{sub}_ses-{ses}_dir-AP_dwi.json"
        )
        fake_dwi_bvec = op.join(
            bids_dir_subses, "dwi", f"sub-{sub}_ses-{ses}_dir-AP_dwi.bvec"
        )
        fake_dwi_bval = op.join(
            bids_dir_subses, "dwi", f"sub-{sub}_ses-{ses}_dir-AP_dwi.bval"
        )
        if not Path(fake_dwi).is_file():
            Path(fake_dwi).touch()
        if not Path(fake_dwi_json).is_file():
            Path(fake_dwi_json).touch()
        if not Path(fake_dwi_bvec).is_file():
            Path(fake_dwi_bvec).touch()
        if not Path(fake_dwi_bval).is_file():
            Path(fake_dwi_bval).touch()

        session_dir = op.join(
            analysis_dir,
            f"sub-{sub}",
            f"ses-{ses}",
        )

        os.makedirs(session_dir, exist_ok=True)
        if container != "Processed_nifti":
            input_dir = op.join(session_dir, "input")
            outpt_dir = op.join(session_dir, "output")

            if not op.exists(input_dir):
                os.makedirs(input_dir)
            else:
                console.print(
                    f"Input folder for sub-{sub}/ses-{ses} is there",
                    style="cyan",
                )
            if not op.exists(outpt_dir):
                os.makedirs(outpt_dir)
            else:
                console.print(
                    f"Output folder for sub-{sub}/ses-{ses} is there",
                    style="cyan",
                )
            if file_name:
                fake_file = op.join(outpt_dir, file_name)
                if not Path(fake_file).is_file():
                    Path(fake_file).touch()
                else:
                    console.print(
                        f"The file for sub-{sub}/ses-{ses}/output is there",
                        style="cyan",
                    )


# #%%
if __name__ == "__main__":
    main()
