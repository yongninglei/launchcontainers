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

import os.path as op
from datetime import datetime

from launchcontainers import utils as do
from launchcontainers.log_setup import console


def qc():
    """
    Placeholder QC entrypoint.

    The package currently routes QC work through :func:`main`; this function is
    kept only as a stub for future expansion.
    """
    return


def main(workdir: str, log_dir: str = "./logs", debug: bool = False):
    """
    Run the package-level QC workflow for a prepared analysis directory.

    Parameters
    ----------
    workdir : str
        Working directory for QC.
    log_dir : str, default="./logs"
        Directory to write logs.
    debug : bool, default=False
        Debug mode.
    """
    # 1. setup run mode logger
    # read the yaml to get input info
    analysis_dir = workdir
    logging_dir = log_dir
    console.print(f" The logging dir is {logging_dir}")
    # get the dir and fpath for launchcontainer logger
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    console_file = open(op.join(logging_dir, f"qc_console_{timestamp}.txt"), "w")
    console.file = console_file

    # read LC config yml from analysis dir
    lc_config_fpath = op.join(analysis_dir, "lc_config.yaml")
    lc_config = do.read_yaml(lc_config_fpath)
    console.print("\n do_qc.main() reading lc config yaml", style="cyan")

    # Get general information from the config.yaml file
    # bidsdir_name = lc_config['general']['bidsdir_name']
    container = lc_config["general"]["container"]
    # if container is freesurgerator or anatrois, there is nothing to check
    if container in ["freesurferator", "anatrois"]:
        console.print(
            "\n There is not yet any QC for freesurferator or anatrois, finishing",
            style="cyan",
        )

    # if it is rtppreproc, or rtp2-preproc, check if the output log has the field success,
    if container in ["freesurferator", "anatrois"]:
        console.print(
            "\n There is not yet any QC for freesurferator or anatrois, finishing",
            style="cyan",
        )
    #
    #
    # if not,
    # generate a new subseslist based on the already have subseslist,
    # mark the ones that are not success
    # create new folder QC_{timestamp}_failed
    # keep running it, until you observed a QC_success

    # if it is rtp2-pipeline or rtppipeline, check if the output log has the field success, if not,
    # check which tract are not finished, generate new tractparams and generate new subseslist
    # create new folder QC_{timestamp}_failed
    # keep running it, until observed a QC_success

    return
