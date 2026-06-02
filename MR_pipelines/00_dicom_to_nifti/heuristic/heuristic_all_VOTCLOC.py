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

    # func
    fLoc_sbref = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-fLoc_run-{item:02d}_sbref',
    )
    fLoc_P = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-fLoc_run-{item:02d}_phase',
    )
    fLoc_M = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-fLoc_run-{item:02d}_magnitude',
    )

    ret_RW_sbref = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-retRW_run-{item:02d}_sbref',
    )
    ret_RW_P = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-retRW_run-{item:02d}_phase',
    )
    ret_RW_M = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-retRW_run-{item:02d}_magnitude',
    )

    ret_FF_sbref = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-retFF_run-{item:02d}_sbref',
    )
    ret_FF_P = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-retFF_run-{item:02d}_phase',
    )
    ret_FF_M = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-retFF_run-{item:02d}_magnitude',
    )

    ret_CB_sbref = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-retCB_run-{item:02d}_sbref',
    )
    ret_CB_P = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-retCB_run-{item:02d}_phase',
    )
    ret_CB_M = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-retCB_run-{item:02d}_magnitude',
    )

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
        fLoc_sbref: [], fLoc_P: [], fLoc_M: [],
        ret_RW_sbref: [], ret_RW_P: [], ret_RW_M: [],
        ret_FF_sbref: [], ret_FF_P: [], ret_FF_M: [],
        ret_CB_sbref: [], ret_CB_P: [], ret_CB_M: [],
        dwi_votcloc_rpe: [], dwi_votcloc: [],
        dwi_votcloc_rpe_pha: [], dwi_votcloc_pha: [],

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

        # TR not working for the XA30 func scan of pRFs

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

        if (s.dim1 == 92) and (s.dim3 == 80):
            if ((s.series_files == 160) or (s.series_files == 159)) and (('fLoc' in s.protocol_name) or ('floc' in s.protocol_name)) :
                if ('Pha' in s.series_description) :
                    info[fLoc_P].append(s.series_id)
                else:
                    info[fLoc_M].append(s.series_id)
            if ((s.series_files == 156) or (s.series_files == 155)) and ('block' not in s.protocol_name) and (('RW' in s.protocol_name) or ('word' in s.protocol_name)):
                if ('Pha' in s.series_description) :
                    info[ret_RW_P].append(s.series_id)
                else:
                    info[ret_RW_M].append(s.series_id)
            if ((s.series_files == 156) or (s.series_files == 155)) and ('FF' in s.protocol_name):
                if ('Pha' in s.series_description) :
                    info[ret_FF_P].append(s.series_id)
                else:
                    info[ret_FF_M].append(s.series_id)
            if ((s.series_files == 156) or (s.series_files == 155)) and (('CB' in s.protocol_name) or ('fixRWblock' in s.protocol_name)):
                if ('Pha' in s.series_description) :
                    info[ret_CB_P].append(s.series_id)
                else:
                    info[ret_CB_M].append(s.series_id)
        # dwi
        # only take the mag
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
