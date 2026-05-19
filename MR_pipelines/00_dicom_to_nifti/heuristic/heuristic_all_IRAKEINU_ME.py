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
    # T1 MP2RAGE
    t1_i1 = create_key(
        "sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:02d}_T1_inv1",
    )
    t1_i2 = create_key(
        "sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:02d}_T1_inv2",
    )
    t1_un = create_key(
        "sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:02d}_T1_uni",
    )

    # T1 weighted MPRAGE
    t1_w = create_key(
        "sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:02d}_T1w"
    )
    # T2 weighted
    t2_w = create_key(
        "sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:02d}_T2w"
    )
    # fmap
    fmap_SE_AP = create_key(
        "sub-{subject}/{session}/fmap/sub-{subject}_{session}_acq-SE_dir-AP_run-{item:01d}_epi",
    )
    fmap_SE_PA = create_key(
        "sub-{subject}/{session}/fmap/sub-{subject}_{session}_acq-SE_dir-PA_run-{item:01d}_epi",
    )

    # fmap
    fmap_ME_AP = create_key(
        "sub-{subject}/{session}/fmap/sub-{subject}_{session}_acq-ME_dir-AP_run-{item:01d}_epi",
    )
    fmap_ME_PA = create_key(
        "sub-{subject}/{session}/fmap/sub-{subject}_{session}_acq-ME_dir-PA_run-{item:01d}_epi",
    )

    # func
    BfLocVideo_SE_sbref = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-BfLocVideo_acq-SE_run-{item:02d}_sbref",
    )
    BfLocVideo_SE_P = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-BfLocVideo_acq-SE_run-{item:02d}_phase",
    )
    BfLocVideo_SE_M = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-BfLocVideo_acq-SE_run-{item:02d}_magnitude",
    )

    # func
    BfLocVideo_ME_sbref = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-BfLocVideo_acq-ME_run-{item:02d}_sbref",
    )
    BfLocVideo_ME_P = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-BfLocVideo_acq-ME_run-{item:02d}_phase",
    )
    BfLocVideo_ME_M = create_key(
        "sub-{subject}/{session}/func/sub-{subject}_{session}_task-BfLocVideo_acq-ME_run-{item:02d}_magnitude",
    )
    info = {
        t1_i1: [],
        t1_i2: [],
        t1_un: [],
        t1_w: [],
        t2_w: [],
        fmap_SE_AP: [],
        fmap_SE_PA: [],
        fmap_ME_AP: [],
        fmap_ME_PA: [],
        BfLocVideo_SE_sbref: [],
        BfLocVideo_SE_P: [],
        BfLocVideo_SE_M: [],
        BfLocVideo_ME_sbref: [],
        BfLocVideo_ME_P: [],
        BfLocVideo_ME_M: [],
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
        if (
            (s.dim1 == 256)
            and (s.dim2 == 240)
            and (s.dim3 == 176)
            and ("mp2rage" in s.protocol_name)
        ):
            if "_INV1" in s.series_description:
                info[t1_i1].append(s.series_id)
            elif "_INV2" in s.series_description:
                info[t1_i2].append(s.series_id)
            elif "_UNI" in s.series_description:
                info[t1_un].append(s.series_id)
        if (
            (s.dim1 == 256)
            and (s.dim2 == 256)
            and (s.dim3 == 176)
            and ("mprage" in s.protocol_name)
        ):
            info[t1_w].append(s.series_id)
        if (
            (s.dim1 == 256)
            and (s.dim2 == 232)
            and (s.dim3 == 176)
            and ("MGH" in s.protocol_name)
        ):
            info[t2_w].append(s.series_id)
        # fmap
        # and ('M' in s.image_type):
        if ("TOPUP" in s.protocol_name.upper()) or ("fmap" in s.protocol_name):
            if (
                (s.dim1 == 100) and (s.dim3 == 80) and (s.series_files == 1)
            ):  # and (s.TR==14.956):
                if "AP" in s.protocol_name:
                    info[fmap_SE_AP].append(s.series_id)
                if "PA" in s.protocol_name:
                    info[fmap_SE_PA].append(s.series_id)
            if (
                (s.dim1 == 120) and (s.dim3 == 68) and (s.series_files == 1)
            ):  # and (s.TR==14.956):
                if "AP" in s.protocol_name or "MultiE" in s.protocol_name:
                    info[fmap_ME_AP].append(s.series_id)
                if "PA" in s.protocol_name:
                    info[fmap_ME_PA].append(s.series_id)

        # TR not working for the XA30 func scan of pRFs

        # functional SBref
        if (
            (s.series_files == 1)
            and ("Pha" not in s.series_description)
            and (s.dim1 == 100)
            and (s.dim3 == 80)
        ):
            # pay attention to add a check for language in the s.protocol_name when in the scanner, otherwise the multiple language thing
            # will cause trouble
            if ("fLoc" in s.protocol_name) or ("fLoc" in s.protocol_name):
                info[BfLocVideo_SE_sbref].append(s.series_id)
        # functional SBref ME
        if (
            (s.series_files == 3)
            and ("Pha" not in s.series_description)
            and (s.dim1 == 120)
            and (s.dim3 == 68)
        ):
            # pay attention to add a check for language in the s.protocol_name when in the scanner, otherwise the multiple language thing
            # will cause trouble
            if ("fLoc" in s.protocol_name) or ("fLoc" in s.protocol_name):
                info[BfLocVideo_ME_sbref].append(s.series_id)
        # functional SE
        if (s.dim1 == 100) and (s.dim3 == 80):
            if (s.series_files == 217) and (
                ("fLoc" in s.protocol_name) or ("fLoc" in s.protocol_name)
            ):
                if "Pha" in s.series_description:
                    info[BfLocVideo_SE_P].append(s.series_id)
                else:
                    info[BfLocVideo_SE_M].append(s.series_id)
        # functional ME
        if (s.dim1 == 120) and (s.dim3 == 68):
            if (s.series_files == 651) and (
                ("fLoc" in s.protocol_name) or ("fLoc" in s.protocol_name)
            ):
                if "Pha" in s.series_description:
                    info[BfLocVideo_ME_P].append(s.series_id)
                else:
                    info[BfLocVideo_ME_M].append(s.series_id)
    return info
