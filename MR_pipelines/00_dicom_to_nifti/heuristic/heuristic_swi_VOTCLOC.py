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
    """Heuristic evaluator for SWI (3D multi-echo EPI) sessions — VOTCLOC project.

    Two acquisition variants observed:
      acq-4TE38 : 4 echoes, dim3=416 slabs  (e.g. sub-05 ses-swi)
      acq-5TE35 : 5 echoes, dim3=352 slabs  (e.g. sub-11 ses-swi)

    Phase-encoding direction:
      dir-PA : 'iPE' or 'RPE' present in protocol_name
      dir-AP : no 'iPE'/'RPE' in protocol_name

    Suffix:
      _magnitude : 'M' in image_type tuple
      _phase     : 'P' in image_type tuple

    Per direction per acquisition block the scanner produces 2 magnitude series
    and 1 phase series. heudiconv auto-increments {item} so each magnitude
    series gets its own run number (run-01, run-02, …) while phase series
    are numbered independently.

    allowed template fields:
        item      - index within category (auto-incremented run number)
        subject   - participant id
        session   - session label
    """

    # --- 4-echo 416-slab keys ---
    swi_4TE_AP_mag = create_key(
        "sub-{subject}/{session}/swi/sub-{subject}_{session}_acq-4TE38_dir-AP_run-{item:02d}_magnitude"
    )
    swi_4TE_AP_pha = create_key(
        "sub-{subject}/{session}/swi/sub-{subject}_{session}_acq-4TE38_dir-AP_run-{item:02d}_phase"
    )
    swi_4TE_PA_mag = create_key(
        "sub-{subject}/{session}/swi/sub-{subject}_{session}_acq-4TE38_dir-PA_run-{item:02d}_magnitude"
    )
    swi_4TE_PA_pha = create_key(
        "sub-{subject}/{session}/swi/sub-{subject}_{session}_acq-4TE38_dir-PA_run-{item:02d}_phase"
    )

    # --- 5-echo 352-slab keys ---
    swi_5TE_AP_mag = create_key(
        "sub-{subject}/{session}/swi/sub-{subject}_{session}_acq-5TE35_dir-AP_run-{item:02d}_magnitude"
    )
    swi_5TE_AP_pha = create_key(
        "sub-{subject}/{session}/swi/sub-{subject}_{session}_acq-5TE35_dir-AP_run-{item:02d}_phase"
    )
    swi_5TE_PA_mag = create_key(
        "sub-{subject}/{session}/swi/sub-{subject}_{session}_acq-5TE35_dir-PA_run-{item:02d}_magnitude"
    )
    swi_5TE_PA_pha = create_key(
        "sub-{subject}/{session}/swi/sub-{subject}_{session}_acq-5TE35_dir-PA_run-{item:02d}_phase"
    )

    info = {
        swi_4TE_AP_mag: [],
        swi_4TE_AP_pha: [],
        swi_4TE_PA_mag: [],
        swi_4TE_PA_pha: [],
        swi_5TE_AP_mag: [],
        swi_5TE_AP_pha: [],
        swi_5TE_PA_mag: [],
        swi_5TE_PA_pha: [],
    }

    for s in seqinfo:
        # Only process the 3D multi-echo SWI sequences
        if "3D-ME-EPI" not in s.protocol_name:
            continue

        # image_type is a tuple, e.g. ('ORIGINAL', 'PRIMARY', 'M', 'NONE')
        image_type = s.image_type if s.image_type else ()
        is_magnitude = "M" in image_type
        is_phase = "P" in image_type

        # Reverse phase encoding: iPE or RPE in protocol name
        proto_upper = s.protocol_name.upper()
        is_pa = ("IPE" in proto_upper) or ("RPE" in proto_upper)

        # Acquisition variant by echo count and slab thickness
        is_4TE = s.dim4 == 4 and s.dim3 == 416
        is_5TE = s.dim4 == 5 and s.dim3 == 352

        if is_4TE:
            if is_pa:
                if is_magnitude:
                    info[swi_4TE_PA_mag].append(s.series_id)
                elif is_phase:
                    info[swi_4TE_PA_pha].append(s.series_id)
            else:
                if is_magnitude:
                    info[swi_4TE_AP_mag].append(s.series_id)
                elif is_phase:
                    info[swi_4TE_AP_pha].append(s.series_id)

        elif is_5TE:
            if is_pa:
                if is_magnitude:
                    info[swi_5TE_PA_mag].append(s.series_id)
                elif is_phase:
                    info[swi_5TE_PA_pha].append(s.series_id)
            else:
                if is_magnitude:
                    info[swi_5TE_AP_mag].append(s.series_id)
                elif is_phase:
                    info[swi_5TE_AP_pha].append(s.series_id)

    return info
