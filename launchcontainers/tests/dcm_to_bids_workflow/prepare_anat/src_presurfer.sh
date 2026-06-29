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


# ADD FSL TO THE PATH BEFORE LAUNCHING MATLAB
# then do
# tbUse BCBLViennaSoft;
# this step is to add pressurfer and NORDIC_RAW into the path so that you
# can use it
#module load gcc/7.3.0
module load afni
module load fsl
module load matlab/R2021B
# VIENNA
# baseP = '/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/bcblvie22/BIDS';

# BCBL
# basedir=/bcbl/home/public/Gari/VOTCLOC/main_exp
# bids_dirname=BIDS

# src_dir=$basedir/$bids_dirname
# analysis_name=week1
# outputdir=${basedir}/${bids_dirname}/derivatives/process_nifti/analysis-${analysis_name}

# subs=('03' '06' '08')
# sess=('01')
# force=false # if overwrite exsting file
# for sub in "$subs[@]" ; do
# for ses in $sess[@]; do

echo "Doing PRESURFER for sub: ${sub}, and ses: ${ses}"
matlab -nosplash -nodesktop -r "\
addpath('$script_dir'); \
tbPath='$tbPath'; \
src_dir='$src_dir'; \
outputdir='$outputdir'; \
sub='$sub'; \
ses='$ses'; \
force=$force; \
presurferT1(tbPath, src_dir, outputdir, sub, ses, force); exit"

# done done
