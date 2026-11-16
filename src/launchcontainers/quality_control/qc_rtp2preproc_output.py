# """
# MIT License
# Copyright (c) 2022-2025 Yongning Lei
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to
# whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# """

"""
The output QC will only related to the analysis dir, so it is very good!
TODO: feature, give a analysis name and version, it will look for the things under BIDS/derivatives using
pybids

For rtp2preproc, it seems that the only thing we need to check is that if the DWI file was prepared.

"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer


def find_newest_log(
    log_dir: Path, exts: tuple[str, ...] = (".err", ".e")
) -> Path | None:
    """
    Return the newest log file in a directory.

    The function first looks for files whose names end with a timestamp of the
    form ``YYYY-MM-DD_HH-MM-SS`` followed by one of the requested extensions.
    If no timestamped filename is found, it falls back to non-timestamped log
    files and chooses the only match or the most recently modified one.

    Parameters
    ----------
    log_dir : pathlib.Path
        Directory containing log files.
    exts : tuple[str, ...], default=('.err', '.e')
        File extensions that should be considered valid log files.

    Returns
    -------
    pathlib.Path | None
        Path to the newest matching log file, or ``None`` if no candidate log
        file exists.
    """
    log_dir = Path(log_dir)
    timestamp_re = re.compile(
        r".*_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})("
        + "|".join(re.escape(e) for e in exts)
        + r")$",
    )

    latest: Path | None = None
    latest_dt: datetime | None = None

    # 1) First pass: look for timestamped files
    for p in log_dir.iterdir():
        if not p.is_file():
            continue
        m = timestamp_re.match(p.name)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")
        if latest_dt is None or ts > latest_dt:
            latest_dt, latest = ts, p

    if latest is not None:
        return latest

    # 2) Fallback: no timestamped logs – pick from any file with the right ext
    plain_logs = [p for p in log_dir.iterdir() if p.is_file() and p.suffix in exts]

    if not plain_logs:
        return None

    if len(plain_logs) == 1:
        return plain_logs[0]

    # If multiple “plain” logs, choose the one most recently modified
    return max(plain_logs, key=lambda p: p.stat().st_mtime)


def find_subseslist(analysis_dir):
    """
    Locate ``subseslist.txt`` somewhere under an analysis directory.

    Parameters
    ----------
    analysis_dir : str or path-like
        Root directory to search.

    Returns
    -------
    str
        Full path to the discovered ``subseslist.txt`` file.
    """
    for dirpath, dirnames, filenames in os.walk(analysis_dir):
        for fname in filenames:
            if fname.lower() == "subseslist.txt":
                return os.path.join(dirpath, fname)
    raise FileNotFoundError(f"No subseslist.txt found under {analysis_dir}")


def check_rtp_preproc_logs(analysis_dir):
    """
    Report sessions whose expected preprocessed DWI output is missing.

    Parameters
    ----------
    analysis_dir : str or path-like
        Analysis directory containing session output folders and
        ``subseslist.txt``.
    """
    path_to_subses = find_subseslist(analysis_dir)
    df_subSes = pd.read_csv(path_to_subses, sep=",", dtype=str)
    for row in df_subSes.itertuples(index=True, name="Pandas"):
        sub = row.sub
        ses = row.ses
        RUN = row.RUN
        dwi = row.dwi
        if RUN == "True" and dwi == "True":
            dwi_file = os.path.join(
                analysis_dir,
                f"sub-{sub}",
                f"ses-{ses}",
                "output",
                "dwi.nii.gz",
            )
            # log_file_dir = os.path.join(
            #     analysis_dir,
            #     f'sub-{sub}',
            #     f'ses-{ses}',
            #     'log',
            # )
            # print(os.listdir(log_file_dir))
            # log_file = find_newest_log(log_file_dir)
            if not os.path.isfile(dwi_file):
                print(f"*****for sub-{sub}_ses-{ses}")
                print(f"!!!Issue with sub-{sub}, ses-{ses}*****\n")
            # else:
            #     with open(log_file) as f:
            #         lines = f.readlines()
            #         #print(f'*****for sub-{sub}_ses-{ses}')
            #         #print(lines[-2] + '###')
            #         if not 'Success' in lines[-2].strip():
            #             print(f'!!!Issue with sub-{sub}, ses-{ses}*****\n')


# Example usage:
# check_rtp_preproc_logs(analysis_dir)


if __name__ == "__main__":
    typer.run(check_rtp_preproc_logs)
