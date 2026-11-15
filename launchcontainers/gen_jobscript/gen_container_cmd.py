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
from datetime import datetime

from launchcontainers.log_setup import console


def gen_cmd_prefix(lc_config):
    """
    Generate the common Apptainer command prefix for a configured host.

    The prefix includes optional module loading and the shared bind mounts
    required to run launchcontainers analyses inside an Apptainer/Singularity
    container.

    Parameters
    ----------
    lc_config : dict
        Parsed launchcontainers YAML configuration.

    Returns
    -------
    str
        Shell command prefix used by container-specific launch commands.
    """
    host = lc_config["general"]["host"]
    jobqueue_config = lc_config["host_options"][host]
    use_module = jobqueue_config["use_module"]
    mount_options = jobqueue_config["mount_options"]
    env_cmd = f"module load {jobqueue_config['apptainer']} &&" if use_module else ""
    # define the mount cmd
    path_mount_cmd = ""
    for mount in mount_options:
        path_mount_cmd += f"--bind {mount}:{mount} "
    # this prefix will give you:
    # first module load apptainer
    # then unset python path
    # -e means cleanenv, no bash env variable is passed to the singularity shell
    # --no-home means the $HOME directory are not bind int the singularity shell
    # --pwd /flywheel/v0 means point the working directory to /flywheel/v0
    # The prefix command we will use for all container for now, it is generic
    cmd_prefix = (
        f"{env_cmd} unset PYTHONPATH; apptainer run "
        f"-e --no-home --containall --pwd /flywheel/v0 {path_mount_cmd}"
    )

    return cmd_prefix


def gen_RTP2_cmd(
    lc_config,
    sub,
    ses,
    analysis_dir,
):
    """
    Build the launch command for one subject/session container run.

    The generated command binds the prepared input and output directories into
    the Flywheel-style container layout expected by the supported DWI-related
    containers and redirects stdout/stderr into timestamped log files.

    Parameters
    ----------
    lc_config : dict
        Parsed launchcontainers YAML configuration.
    sub : str
        Subject identifier without the ``sub-`` prefix.
    ses : str
        Session identifier without the ``ses-`` prefix.
    analysis_dir : str
        Prepared analysis directory containing per-session input/output trees.

    Returns
    -------
    str
        Full shell command used to run the container for one session.

    Raises
    ------
    ValueError
        If no command template can be selected for the configured container.
    """

    # Relevant directories
    container = lc_config["general"]["container"]
    containerdir = lc_config["general"]["containerdir"]
    version = lc_config["container_specific"][container]["version"]

    # Location of the Singularity Image File (.sif)
    container_name = os.path.join(containerdir, f"{container}_{version}.sif")

    # Define the directory and the file name to output the log of each subject
    container_logdir = os.path.join(
        analysis_dir,
        "sub-" + sub,
        "ses-" + ses,
        "output",
        "log",
    )
    # get timestamp for output log
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfilename = f"{container_logdir}/{container}-sub-{sub}_ses-{ses}_{timestamp}"
    deriv_subses_dir = os.path.join(analysis_dir, f"sub-{sub}", f"ses-{ses}")
    # get the cmd prefix
    cmd_prefix = gen_cmd_prefix(lc_config)
    cmd = None
    if container in ["anatrois", "rtppreproc", "rtp-pipeline"]:
        cmd = (
            f"{cmd_prefix} "
            f"--bind {deriv_subses_dir}/input:/flywheel/v0/input:ro "
            f"--bind {deriv_subses_dir}/output:/flywheel/v0/output "
            f"--bind {deriv_subses_dir}/output/log/config.json:/flywheel/v0/config.json "
            f"{container_name} 1> {logfilename}.log 2> {logfilename}.err  "
        )

    if container == "freesurferator":
        cmd = (
            f"{cmd_prefix} "
            f"--bind {deriv_subses_dir}/input:/flywheel/v0/input:ro "
            f"--bind {deriv_subses_dir}/output:/flywheel/v0/output "
            f"--bind {deriv_subses_dir}/work:/flywheel/v0/work "
            f"--bind {deriv_subses_dir}/output/log/config.json:/flywheel/v0/config.json "
            f"--env PATH=/opt/freesurfer/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/freesurfer/fsfast/bin:/opt/freesurfer/tktools:/opt/freesurfer/mni/bin:/sbin:/bin:/opt/ants/bin "
            f"--env LANG=C.UTF-8 "
            f"--env GPG_KEY=E3FF2839C048B25C084DEBE9B26995E310250568 "
            f"--env PYTHON_VERSION=3.9.15 "
            f"--env PYTHON_PIP_VERSION=22.0.4 "
            f"--env PYTHON_SETUPTOOLS_VERSION=58.1.0 "
            f"--env PYTHON_GET_PIP_URL=https://github.com/pypa/get-pip/raw/66030fa03382b4914d4c4d0896961a0bdeeeb274/public/get-pip.py "
            f"--env PYTHON_GET_PIP_SHA256=1e501cf004eac1b7eb1f97266d28f995ae835d30250bec7f8850562703067dc6 "
            f"--env FLYWHEEL=/flywheel/v0 "
            f"--env ANTSPATH=/opt/ants/bin/ "
            f"--env FREESURFER_HOME=/opt/freesurfer "
            f"--env FREESURFER=/opt/freesurfer "
            f"--env DISPLAY=:50.0 "
            f"--env FS_LICENSE=/flywheel/v0/work/license.txt "
            f"--env OS=Linux "
            f"--env FS_OVERRIDE=0 "
            f"--env FSF_OUTPUT_FORMAT=nii.gz "
            f"--env MNI_DIR=/opt/freesurfer/mni "
            f"--env LOCAL_DIR=/opt/freesurfer/local "
            f"--env FSFAST_HOME=/opt/freesurfer/fsfast "
            f"--env MINC_BIN_DIR=/opt/freesurfer/mni/bin "
            f"--env MINC_LIB_DIR=/opt/freesurfer/mni/lib "
            f"--env MNI_DATAPATH=/opt/freesurfer/mni/data "
            f"--env FMRI_ANALYSIS_DIR=/opt/freesurfer/fsfast "
            f"--env PERL5LIB=/opt/freesurfer/mni/lib/perl5/5.8.5 "
            f"--env MNI_PERL5LIB=/opt/freesurfer/mni/lib/perl5/5.8.5 "
            f"--env XAPPLRESDIR=/opt/freesurfer/MCRv97/X11/app-defaults "
            f"--env MCR_CACHE_ROOT=/flywheel/v0/output "
            f"--env MCR_CACHE_DIR=/flywheel/v0/output/.mcrCache9.7 "
            f"--env FSL_OUTPUT_FORMAT=nii.gz "
            f"--env ANTS_VERSION=v2.4.2 "
            f"--env QT_QPA_PLATFORM=xcb "
            f"--env PWD=/flywheel/v0 "
            f"{container_name} "
            f"-c python run.py 1> {logfilename}.log 2> {logfilename}.err  "
        )

    if container == "rtp2-preproc":
        cmd = (
            f"{cmd_prefix} "
            f"--bind {deriv_subses_dir}/input:/flywheel/v0/input:ro "
            f"--bind {deriv_subses_dir}/output:/flywheel/v0/output "
            f"--bind {deriv_subses_dir}/output/log/config.json:/flywheel/v0/config.json "
            f"--env FLYWHEEL=/flywheel/v0 "
            f"--env LD_LIBRARY_PATH=/opt/fsl/lib:  "
            f"--env FSLWISH=/opt/fsl/bin/fslwish  "
            f"--env FSLTCLSH=/opt/fsl/bin/fsltclsh  "
            f"--env FSLMULTIFILEQUIT=TRUE "
            f"--env FSLOUTPUTTYPE=NIFTI_GZ  "
            f"--env FSLDIR=/opt/fsl  "
            f"--env FREESURFER_HOME=/opt/freesurfer  "
            f"--env ARTHOME=/opt/art  "
            f"--env ANTSPATH=/opt/ants/bin  "
            f"--env PYTHON_GET_PIP_SHA256=1e501cf004eac1b7eb1f97266d28f995ae835d30250bec7f8850562703067dc6 "
            f"--env PYTHON_GET_PIP_URL=https://github.com/pypa/get-pip/raw/66030fa03382b4914d4c4d0896961a0bdeeeb274/public/get-pip.py "
            f"--env PYTHON_PIP_VERSION=22.0.4  "
            f"--env PYTHON_VERSION=3.9.15  "
            f"--env GPG_KEY=E3FF2839C048B25C084DEBE9B26995E310250568  "
            f"--env LANG=C.UTF-8  "
            f"--env PATH=/opt/mrtrix3/bin:/opt/ants/bin:/opt/art/bin:/opt/fsl/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
            f"--env PYTHON_SETUPTOOLS_VERSION=58.1.0 "
            f"--env DISPLAY=:50.0 "
            f"--env QT_QPA_PLATFORM=xcb  "
            f"--env FS_LICENSE=/opt/freesurfer/license.txt  "
            f"--env PWD=/flywheel/v0 "
            f"{container_name} "
            f"-c python run.py 1> {logfilename}.log 2> {logfilename}.err  "
        )

    if container == "rtp2-pipeline":
        cmd = (
            f"{cmd_prefix} "
            f"--bind {deriv_subses_dir}/input:/flywheel/v0/input:ro "
            f"--bind {deriv_subses_dir}/output:/flywheel/v0/output "
            f"--bind {deriv_subses_dir}/output/log/config.json:/flywheel/v0/config.json "
            f"--env PATH=/opt/mrtrix3/bin:/opt/ants/bin:/opt/art/bin:/opt/fsl/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
            f"--env LANG=C.UTF-8 "
            f"--env GPG_KEY=E3FF2839C048B25C084DEBE9B26995E310250568 "
            f"--env PYTHON_VERSION=3.9.15 "
            f"--env PYTHON_PIP_VERSION=22.0.4 "
            f"--env PYTHON_SETUPTOOLS_VERSION=58.1.0 "
            f"--env PYTHON_GET_PIP_URL=https://github.com/pypa/get-pip/raw/66030fa03382b4914d4c4d0896961a0bdeeeb274/public/get-pip.py "
            f"--env PYTHON_GET_PIP_SHA256=1e501cf004eac1b7eb1f97266d28f995ae835d30250bec7f8850562703067dc6 "
            f"--env ANTSPATH=/opt/ants/bin "
            f"--env ARTHOME=/opt/art "
            f"--env FREESURFER_HOME=/opt/freesurfer "
            f"--env FSLDIR=/opt/fsl "
            f"--env FSLOUTPUTTYPE=NIFTI_GZ "
            f"--env FSLMULTIFILEQUIT=TRUE "
            f"--env FSLTCLSH=/opt/fsl/bin/fsltclsh "
            f"--env FSLWISH=/opt/fsl/bin/fslwish "
            f"--env LD_LIBRARY_PATH=/opt/mcr/v99/runtime/glnxa64:/opt/mcr/v99/bin/glnxa64:/opt/mcr/v99/sys/os/glnxa64:/opt/mcr/v99/extern/bin/glnxa64:/opt/fsl/lib: "
            f"--env FLYWHEEL=/flywheel/v0 "
            f"--env TEMPLATES=/templates "
            f"--env XAPPLRESDIR=/opt/mcr/v99/X11/app-defaults "
            f"--env MCR_CACHE_FOLDER_NAME=/flywheel/v0/output/.mcrCache9.9 "
            f"--env MCR_CACHE_ROOT=/flywheel/v0/output "
            f"--env MRTRIX_TMPFILE_DIR=/flywheel/v0/output/tmp "
            f"--env PWD=/flywheel/v0 "
            f"--env TMPDIR=/flywheel/v0/work "
            f"{container_name} "
            f"-c python run.py 1> {logfilename}.log 2> {logfilename}.err  "
        )

    # If after all configuration, we do not have command, raise an error
    if cmd is None:
        console.print(
            "the DWI PIPELINE command is not assigned, please check your config.yaml[general][host] session\n",
            style="red",
        )
        raise ValueError("Launch command is not defined, aborting")

    return cmd
