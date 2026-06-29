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
from __future__ import annotations


def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """
    fmap_AP = create_key(
        'sub-{subject}/{session}/fmap/sub-{subject}_{session}_acq-fMRI_dir-AP_run-{item:01d}_epi',
    )
    fmap_PA = create_key(
        'sub-{subject}/{session}/fmap/sub-{subject}_{session}_acq-fMRI_dir-PA_run-{item:01d}_epi',
    )
    # func
    fLoc_sbref = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-fLoc_run-{item:02d}_sbref',
    )

    ret_RW_sbref = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-retRW_run-{item:02d}_sbref',
    )

    ret_FF_sbref = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-retFF_run-{item:02d}_sbref',
    )


    ret_CB_sbref = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-retCB_run-{item:02d}_sbref',
    )

    info = {
        fmap_AP: [], fmap_PA: [],
        fLoc_sbref: [], 
        ret_RW_sbref: [], 
        ret_FF_sbref: [], 
        ret_CB_sbref: [], 

    }
    last_run = len(seqinfo)

    for s in seqinfo:
        """
        The namedtuple `s` contains the following fields:

        * total_files_till_now
        * example_dcm_file
        * series_id
        * dcm_dir_name
        * unspecified2
        * unspecified3
        * dim1
        * dim2
        * dim3
        * dim4
        * TR
        * TE
        * protocol_name
        * is_motion_corrected
        * is_derived
        * patient_id
        * study_description
        * referring_physician_name
        * series_description
        * image_type
        """
        # fmap
        # and ('M' in s.image_type):
        if ('TOPUP' in s.protocol_name.upper()) or ('fmap' in s.protocol_name):
            if (s.dim1 == 92) and (s.dim3 == 80) and (s.series_files == 1):  # and (s.TR==14.956):
                if ('AP' in s.protocol_name) :
                    info[fmap_AP].append(s.series_id)
                if ('PA' in s.protocol_name) :
                    info[fmap_PA].append(s.series_id)

        # functional SBref
        if (s.series_files == 1) and ('Pha' not in s.series_description):
            # pay attention to add a check for language in the s.protocol_name when in the scanner, otherwise the multiple language thing
            # will cause trouble
            if ('fLoc' in s.protocol_name) or ('floc' in s.protocol_name):
                info[fLoc_sbref].append(s.series_id)
            if (('RW' in s.protocol_name) or ('word' in s.protocol_name)) and ('block' not in s.protocol_name):
                info[ret_RW_sbref].append(s.series_id)
            if ('FF' in s.protocol_name) or ('word' in s.protocol_name):
                info[ret_FF_sbref].append(s.series_id)
            if (('CB' in s.protocol_name) or ('fixRWblock' in s.protocol_name)) :
                info[ret_CB_sbref].append(s.series_id)

    return info
