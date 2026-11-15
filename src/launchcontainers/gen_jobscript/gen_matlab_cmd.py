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


def gen_matlab_cmd(lc_config, sub, ses, analysis_dir):
    """
    Build a MATLAB launch command for one subject/session.

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
        Full shell command to invoke the MATLAB script for one session.
    """
    raise NotImplementedError("MATLAB job generation is not yet implemented.")
