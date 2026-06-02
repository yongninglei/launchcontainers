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
    # T1 MP2RAGE
    t1_i1 = create_key(
        'sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:02d}_T1_inv1',
    )
    t1_i2 = create_key(
        'sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:02d}_T1_inv2',
    )
    t1_un = create_key(
        'sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:02d}_T1_uni',
    )

    # T1 weighted MPRAGE
    t1_w = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:02d}_T1w')
    
    # T2 weighted
    t2_w = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:02d}_T2w')
    # fmap
    fmap_AP = create_key(
        'sub-{subject}/{session}/fmap/sub-{subject}_{session}_acq-fMRI_dir-AP_run-{item:01d}_epi',
    )
    fmap_PA = create_key(
        'sub-{subject}/{session}/fmap/sub-{subject}_{session}_acq-fMRI_dir-PA_run-{item:01d}_epi',
    )

    # dwi
    # dwi
    dwi_votcloc_rpe = create_key(
        'sub-{subject}/{session}/dwi/sub-{subject}_{session}_acq-votcloc1d5_dir-PA_run-{item:02d}_magnitude',
    )
    dwi_votcloc = create_key(
        'sub-{subject}/{session}/dwi/sub-{subject}_{session}_acq-votcloc1d5_dir-AP_run-{item:02d}_magnitude',
    )
    dwi_votcloc_rpe_pha = create_key(
        'sub-{subject}/{session}/dwi/sub-{subject}_{session}_acq-votcloc1d5_dir-PA_run-{item:02d}_phase',
    )
    dwi_votcloc_pha = create_key(
        'sub-{subject}/{session}/dwi/sub-{subject}_{session}_acq-votcloc1d5_dir-AP_run-{item:02d}_phase',
    )

    info = {
        t1_i1: [], t1_i2: [], t1_un: [], t1_w: [], t2_w: [],
        fmap_AP: [], fmap_PA: [],
        dwi_votcloc_rpe: [], dwi_votcloc: [],
        dwi_votcloc_rpe_pha: [], dwi_votcloc_pha: [],
    }
    
    last_run = len(seqinfo)

    for s in seqinfo:
        # T1
        if (s.dim1 == 256) and (s.dim2 == 240) and (s.dim3 == 176) and ('mp2rage' in s.protocol_name):
            if ('_INV1' in s.series_description):
                info[t1_i1].append(s.series_id)
            elif ('_INV2' in s.series_description):
                info[t1_i2].append(s.series_id)
            elif ('_UNI' in s.series_description):
                info[t1_un].append(s.series_id)
        if (s.dim1 == 256) and (s.dim2 == 256) and (s.dim3 == 176) and ('mprage' in s.protocol_name):
            info[t1_w].append(s.series_id)      
        # T2
        if (s.dim1 == 256) and (s.dim2 == 232) and (s.dim3 == 176) and ('MGH' in s.protocol_name):
            info[t2_w].append(s.series_id)
        # fmap
        # and ('M' in s.image_type):
        if ('TOPUP' in s.protocol_name.upper()) or ('fmap' in s.protocol_name):
            if (s.dim1 == 92) and (s.dim3 == 80) and (s.series_files == 1):  # and (s.TR==14.956):
                if ('AP' in s.protocol_name) :
                    info[fmap_AP].append(s.series_id)
                if ('PA' in s.protocol_name) :
                    info[fmap_PA].append(s.series_id)
        # dwi
        if (('M' in s.image_type) or ('Pha' not in s.series_description)) and ('SBRef' not in s.series_description):
            if ('dMRI' in s.series_description) or ('1d5iso' in s.series_description):
                if ('PA' in s.series_description) and (s.series_files == 6):
                    info[dwi_votcloc_rpe].append(s.series_id)
                if ('AP' in s.series_description) and (s.series_files == 105):
                    info[dwi_votcloc].append(s.series_id)

        if (('P' in s.image_type) or ('Pha' in s.series_description)) and ('SBRef' not in s.series_description):
            if ('dMRI' in s.series_description) or ('1d5iso' in s.series_description):
                if ('PA' in s.series_description) and (s.series_files == 6):
                    info[dwi_votcloc_rpe_pha].append(s.series_id)
                if ('AP' in s.series_description) and (s.series_files == 105):
                    info[dwi_votcloc_pha].append(s.series_id)
    return info
