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
from __future__ import annotations

import os.path as op

from nilearn.image import load_img
from scipy import stats
from launchcontainers.log_setup import console

# %%compare data from two folders

basedir = "/bcbl/home//public/Gari/VOTCLOC/derivatives/fmriprep"

srcanalysis = "analysis-okazaki_ST0"
targanalysis = "analysis-okazaki_ST05"

subid = "01"

src_filename = op.join(
    basedir,
    srcanalysis,
    f"sub-{subid}",
    "anat",
    f"sub-{subid}_run-01_desc-preproc_T1w.nii.gz",
)

target_filename = op.join(
    basedir,
    targanalysis,
    f"sub-{subid}",
    "anat",
    f"sub-{subid}_run-01_desc-preproc_T1w.nii.gz",
)


src_file = load_img(src_filename)
target_file = load_img(target_filename)
# compare surf,load_surf will give the data directly
# slope, intercept, r_value, p_value, std_err = stats.linregress(src_file, target_file)
# compare T1 3D, just simply reshape
slope, intercept, r_value, p_value, std_err = stats.linregress(
    src_file.get_fdata().reshape(224 * 240 * 256),
    target_file.get_fdata().reshape(224 * 240 * 256),
)
if r_value**2 < 0.9999:
    console.print(f"Attention the result r square is {r_value**2}", style="yellow")
console.print(r_value)
