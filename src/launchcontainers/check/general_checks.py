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

import os
import subprocess as sp

from launchcontainers.log_setup import console
from launchcontainers.utils import resolve_deriv_dir


def cli_show_folder_struc(analysis_dir, sub, ses):
    """
    Show an example prepared input tree for one subject/session.

    Parameters
    ----------
    analysis_dir : str
        Prepared analysis directory.
    sub : str
        Subject identifier without the ``sub-`` prefix.
    ses : str
        Session identifier without the ``ses-`` prefix.
    """

    # Build the path to that subject/session folder
    path = os.path.join(analysis_dir, f"sub-{sub}", f"ses-{ses}", "input")

    # Call “tree -C <path>” and let it print directly to your terminal
    sp.run(["tree", "-C", "-L", "3", path], check=True)


def print_option_for_review(
    num_of_jobs,
    lc_config,
    container,
    bidsdir_name,
):
    """
    Log the most important launch settings for manual review.

    Parameters
    ----------
    num_of_jobs : int
        Number of subject/session rows that will be launched.
    lc_config : dict
        Parsed launchcontainers YAML configuration.
    container : str
        Container name selected for the run.
    bidsdir_name : str
        Name of the BIDS dataset directory under the configured base path.

    Raises
    ------
    FileNotFoundError
        If the configured container image is not present in ``containerdir``.
    """

    basedir = lc_config["general"]["basedir"]
    host = lc_config["general"]["host"]
    bids_dname = os.path.join(basedir, bidsdir_name)
    containerdir = lc_config["general"]["containerdir"]
    version = lc_config["container_specific"][container]["version"]
    analysis_name = lc_config["general"]["analysis_name"]
    all_containers = os.listdir(containerdir)
    # add a check to see if container is there
    container_sif_name = f"{container}_{version}.sif"
    container_in_place = container_sif_name in all_containers
    if not container_in_place:
        raise FileNotFoundError(
            f"No such file : {container_sif_name} \n under {containerdir} "
        )
    # output the options here for the user to review:
    console.print(
        "\n"
        + "#####################################################\n"
        + f"SubsesList is read, there are * {num_of_jobs} * jobs \n "
        + f"Host is {host} \n"
        + f"Basedir is: {basedir} \n"
        + f"Container is:  {container_sif_name}\n"
        + f"singularity image dir is {containerdir} \n"
        + f"analysis name is: {analysis_name} \n"
        + "##################################################### \n",
        style="bold red",
    )

    if container in ["freesurferator", "anatrois"]:
        src_dir = bids_dname
        console.print(f"\n### The source dir is: {src_dir}", style="bold red")

    if container in ["rtppreproc", "rtp2-preproc"]:
        precontainer_anat = lc_config["container_specific"][container][
            "precontainer_anat"
        ]
        anat_analysis_name = lc_config["container_specific"][container][
            "anat_analysis_name"
        ]

        pre_anatrois_dir = resolve_deriv_dir(
            lc_config, precontainer_anat, anat_analysis_name
        )

        console.print(
            f"\n ### The source FSMASK and T1w dir: {pre_anatrois_dir}",
            style="bold red",
        )
    if container in ["rtp-pipeline", "rtp2-pipeline"]:
        # rtppipeline specefic variables
        precontainer_anat = lc_config["container_specific"][container][
            "precontainer_anat"
        ]
        anat_analysis_name = lc_config["container_specific"][container][
            "anat_analysis_name"
        ]
        precontainer_preproc = lc_config["container_specific"][container][
            "precontainer_preproc"
        ]
        preproc_analysis_name = lc_config["container_specific"][container][
            "preproc_analysis_name"
        ]
        # define the pre containers
        pre_anatrois_dir = resolve_deriv_dir(
            lc_config, precontainer_anat, anat_analysis_name
        )

        pre_preproc_dir = resolve_deriv_dir(
            lc_config, precontainer_preproc, preproc_analysis_name
        )

        console.print(
            f"\n### The source FSMASK and ROI dir is: {pre_anatrois_dir} \n"
            + f"The source DWI preprocessing dir is: {pre_preproc_dir} \n",
            style="bold red",
        )
    return
