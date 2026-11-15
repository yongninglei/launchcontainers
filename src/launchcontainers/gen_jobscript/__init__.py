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

from launchcontainers import utils as do
from launchcontainers.gen_jobscript.gen_container_cmd import gen_RTP2_cmd
from launchcontainers.gen_jobscript.gen_matlab_cmd import gen_matlab_cmd
from launchcontainers.gen_jobscript.gen_py_cmd import gen_py_cmd

# Containers that use apptainer/singularity images
_CONTAINER_JOBS = {
    "anatrois",
    "rtppreproc",
    "rtp-pipeline",
    "freesurferator",
    "rtp2-preproc",
    "rtp2-pipeline",
}


def gen_launch_cmd(
    parse_namespace,
    df_subses,
    batch_command_file,
):
    """
    Generate one launch command per requested subject/session row.

    Routes to the appropriate command generator based on the job type
    configured in ``lc_config``:

    * Apptainer/Singularity containers → :func:`gen_container_cmd.gen_RTP2_cmd`
    * Python scripts → :func:`gen_py_cmd.gen_py_cmd`
    * MATLAB scripts → :func:`gen_matlab_cmd.gen_matlab_cmd`

    The command list is also written to ``batch_command_file`` so scheduler
    array jobs can read one command per task index.

    Parameters
    ----------
    parse_namespace : argparse.Namespace
        Parsed CLI arguments for run mode.
    df_subses : list[tuple[str, str]]
        Filtered subject/session pairs to launch.
    batch_command_file : str or path-like
        Output text file that stores the generated command list.

    Returns
    -------
    list[str]
        Launch commands in the same order as ``df_subses``.
    """
    analysis_dir = parse_namespace.workdir
    lc_config_fpath = op.join(analysis_dir, "lc_config.yaml")
    lc_config = do.read_yaml(lc_config_fpath)
    container = lc_config["general"]["container"]

    # Select the per-subject command builder
    if container in _CONTAINER_JOBS:
        _gen_cmd = gen_RTP2_cmd
    elif container == "matlab":
        _gen_cmd = gen_matlab_cmd
    elif container == "python":
        _gen_cmd = gen_py_cmd
    else:
        raise ValueError(
            f"Unknown container/job type '{container}'. "
            f"Expected one of: {sorted(_CONTAINER_JOBS) + ['matlab', 'python']}"
        )

    commands = []
    for sub, ses in df_subses:
        command = _gen_cmd(lc_config, sub, ses, analysis_dir)
        commands.append(command)

    with open(batch_command_file, "w") as f:
        for cmd in commands:
            f.write(f"{cmd}\n")

    return commands
