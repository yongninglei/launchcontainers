# """
# MIT License
# Copyright (c) 2024-2025 Yongning Lei
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# """
# DWI-only version of heuristic_all_VOTCLOC.py: same dwi rules, every other
# modality dropped, so a re-conversion touches only the dwi/ folder.
from __future__ import annotations


def create_key(template, outtype=("nii.gz",), annotation_classes=None):
    if template is None or not template:
        raise ValueError("Template must be a valid format string")
    return template, outtype, annotation_classes


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """
    # dwi
    dwi_votcloc_rpe = create_key(
        "sub-{subject}/{session}/dwi/sub-{subject}_{session}_acq-votcloc1d5_dir-PA_run-{item:02d}_magnitude",
    )
    dwi_votcloc = create_key(
        "sub-{subject}/{session}/dwi/sub-{subject}_{session}_acq-votcloc1d5_dir-AP_run-{item:02d}_magnitude",
    )
    dwi_votcloc_rpe_pha = create_key(
        "sub-{subject}/{session}/dwi/sub-{subject}_{session}_acq-votcloc1d5_dir-PA_run-{item:02d}_phase",
    )
    dwi_votcloc_pha = create_key(
        "sub-{subject}/{session}/dwi/sub-{subject}_{session}_acq-votcloc1d5_dir-AP_run-{item:02d}_phase",
    )
    info = {
        dwi_votcloc_rpe: [],
        dwi_votcloc: [],
        dwi_votcloc_rpe_pha: [],
        dwi_votcloc_pha: [],
    }

    for s in seqinfo:
        # only take the mag
        if (("M" in s.image_type) or ("Pha" not in s.series_description)) and (
            "SBRef" not in s.series_description
        ):
            if ("dMRI" in s.series_description) or ("1d5iso" in s.series_description):
                if ("PA" in s.series_description) and (s.series_files == 6):
                    info[dwi_votcloc_rpe].append(s.series_id)
                if ("AP" in s.series_description) and (s.series_files == 105):
                    info[dwi_votcloc].append(s.series_id)

        if (("P" in s.image_type) or ("Pha" in s.series_description)) and (
            "SBRef" not in s.series_description
        ):
            if ("dMRI" in s.series_description) or ("1d5iso" in s.series_description):
                if ("PA" in s.series_description) and (s.series_files == 6):
                    info[dwi_votcloc_rpe_pha].append(s.series_id)
                if ("AP" in s.series_description) and (s.series_files == 105):
                    info[dwi_votcloc_pha].append(s.series_id)
    return info
