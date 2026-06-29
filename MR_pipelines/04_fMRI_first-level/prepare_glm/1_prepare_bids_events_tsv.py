# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2024
# All rights reserved.
#
# This script is distributed under the Apache-2.0 license.
# You may use, distribute, and modify this code under the terms of the Apache-2.0 license.
# See the LICENSE file for details.
#
# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
#
# Author: Yongning Lei
# Email: yl4874@nyu.edu
# GitHub: https://github.com/yongninglei
# -----------------------------------------------------------------------------
from __future__ import annotations

import os
from bids import BIDSLayout
from launchcontainers.utils import force_symlink
import logging
from datetime import datetime
from pathlib import Path

"""
This code should be able to create symbolic link
between sub- /ses- /func/xx.events.tsv to the BIDS/sourcedata/fMRI_log/

"""

basedir = "/bcbl/home/public/Gari/IRAKEINU"
sourcedata_dir = f"{basedir}/BIDS/sourcedata"
bids_dir = f"{basedir}/BIDS"
layout = BIDSLayout(bids_dir, validate=False)
sub_list = layout.get_subject()
task = "BfLocVideo"
force = True

logging.basicConfig(
    filename=os.path.join(
        basedir, f"check_bids_events_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    ),
    filemode="w",  # "a" for append
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("prepare_events_tsv")
logger.info("This will be saved to output.log")


def get_onset_onset_dirname(onset_subses_dir, sub, ses):
    """
    This function we used that can check if the sub- ses- id is the same with BIDS
    """
    onset_subses_dir = os.path.join(sourcedata_dir, f"sub-{sub}", f"ses-{ses}")
    votcloc_logs = [i for i in os.listdir(onset_subses_dir) if "1back" in i]
    for i in votcloc_logs:
        dirname = i
        subses = f"sub-{sub}_ses-{ses}"
        logger.info(subses)
        if subses in dirname:
            onset_dirname = dirname
        else:
            continue
    logger.info(f"Onset_dirname_we got is {onset_dirname}")
    return onset_dirname


# TODO:
# get onset file under the onset_dirname, so that there will be no error

# # loop and linking the bids_events.tsv under BIDS sub- / ses- folder
# for sub in sub_list:
#     ses_list=layout.get_session(subject=sub)
#     for ses in ses_list:
# Fixed subjects and sessions
subjects = [f"{i:02d}" for i in range(1, 12)]  # sub-01 to sub-11
sessions = [f"{i:02d}" for i in range(1, 11)]  # ses-01 to ses-10

# Check each subject/session combination
for sub in subjects:
    for ses in sessions:
        onset_dirname = get_onset_onset_dirname(sourcedata_dir, sub, ses)
        all_onset_dir = os.path.join(
            sourcedata_dir, f"sub-{sub}", f"ses-{ses}", onset_dirname
        )
        runs = layout.get_runs(subject=sub, session=ses, task="fLoc")
        for run in runs:
            # get fLoc func
            bids_func = layout.get(
                subject=sub,
                session=ses,
                datatype="func",
                task="fLoc",
                suffix="bold",
                run=run,
                extension="nii.gz",
                return_type="list",
            )
            if len(bids_func) == 0:
                logger.error(
                    "B" * 20
                    + f"The sub={sub} ses={ses} run={run} bids file doesnt exist \n"
                )
            else:
                bids_event = os.path.join(
                    bids_func[0].dirname,
                    bids_func[0].filename.replace("bold.nii.gz", "events.tsv"),
                )

                src_event = os.path.join(
                    all_onset_dir,
                    bids_func[0].filename.replace("bold.nii.gz", "events.tsv"),
                )

                if not os.path.exists(src_event):
                    # if not have source event raise error
                    logger.error(
                        "E" * 20
                        + f"source event file are not exist for  sub={sub} ses={ses} run={run} \n"
                    )
                elif os.path.exists(bids_event) and not force:
                    logger.info("targ event fiel exist do nothing")
                else:
                    try:
                        force_symlink(src_event, bids_event, force)
                    except Exception as e:
                        logger.error(f"Error is {e}")

## check the source data dir, if there are enough events.tsv
for sub in sub_list:
    ses_list = layout.get_session(subject=sub)
    for ses in ses_list:
        onset_dirname = get_onset_onset_dirname(sourcedata_dir, sub, ses)
        all_onset_dir = os.path.join(
            sourcedata_dir, f"sub-{sub}", f"ses-{ses}", onset_dirname
        )
        all_onset_dir = Path(all_onset_dir)
        # Find all files ending with events.tsv
        events_files = list(all_onset_dir.rglob("*events.tsv"))
        if len(events_files) != 10:
            print(
                f"sub-{sub}, ses-{ses} Number of events.tsv files: {len(events_files)}"
            )

# for i in sub_list:
#     for j in ses_list:
#         print(f'sub is {i}, ses is {j}')

#         onset_dirname = get_onset_onset_dirname(sourcedata_dir, i, j)
#         all_onsets = os.path.join(sourcedata_dir, f'sub-{i}', f'ses-{j}', onset_dirname)
#         for r in range(runs):
#             run_num = f'{(r+1):02}'
#             # get the events.tsv filename
#             src_fname = f'sub-{i}_ses-{j}_task-{task}_run-{run_num}_events.tsv'
#             target_fname = f'sub-{i}_ses-{j}_task-{task}_run-{run_num}_events.tsv'
#             src_onset = path.join(all_onsets, src_fname)
#             target_path = path.join(output_dir, f'sub-{i}', f'ses-{j}', 'func')
#             target = path.join(target_path, target_fname)
#             print(
#                 f'src file exist: {path.exists(src_onset)} \n \
#                     and dst path exists: {os.path.islink(target)}',
#             )

#             if not path.exists(target_path):
#                 os.makedirs(target_path)

#             if os.path.islink(target):
#                 os.unlink(target)
#             try:
#                 os.symlink(src_onset, target)
#                 print(f'symlink create copied to {target}')
#             except Exception as e:
#                 print(f'Error is {e}')
