#!/bin/bash
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

#SBATCH --time=06:00:00
#SBATCH -n 1  # this stands for number of task/ or number of nodes being used
#SBATCH --cpus-per-task=30    #6
#SBATCH --mem=80000
#SBATCH --partition=general  # Queue names you can submit to
#SBATCH --qos=regular       #<regular,serial,test,long or xlong>

# if you want to syn the logs to the analysis folder,
# need to create the log folder first
# the sublist is shared across launch.sh and the slurm

echo "basedir is ${basedir}"
echo "bids_dir is ${bids_dir}"

# Read-only BIDS input (on /data — not writable)
BIDS_DIR="${bids_dir}"
# Scratch-side paths (all writable)
CODE_DIR="${basedir}/code"
fp_output_dir="${basedir}/derivatives/fmriprep-${fp_version}_${analysis_name}"
# Prepare output folder (on scratch — writable)
mkdir -p ${fp_output_dir}
echo "fp_output_dir: $fp_output_dir"

export cache_dir=${basedir}/fmriprep_tmps_${fp_version}_${analysis_name}

# Prepare some writeable bind-mount points.
TEMPLATEFLOW_HOST_HOME=$cache_dir/.cache/templateflow
FMRIPREP_HOST_CACHE=$cache_dir/.cache/fmriprep
FMRIPREP_WORK_DIR=$cache_dir/.work/fmriprep
mkdir -p ${TEMPLATEFLOW_HOST_HOME}
mkdir -p ${FMRIPREP_HOST_CACHE}
mkdir -p ${FMRIPREP_WORK_DIR}


# This trick will help you reuse freesurfer results across pipelines and fMRIPrep versions
# mkdir -p ${BIDS_DIR}/derivatives/freesurfer-6.0.1
# if [ ! -d ${BIDS_DIR}/${fp_output_dir}/freesurfer ]; then
#         ln -s ${BIDS_DIR}/derivatives/freesurfer-6.0.1
#         ${BIDS_DIR}/${fp_output_dir}/freesurfer
#         fi

# Make sure FS_LICENSE is defined in the container.
# License file lives in the (read-only) BIDS dir — reading it is fine.
export APPTAINERENV_FS_LICENSE=${BIDS_DIR}/.license
# Designate a templateflow bind-mount point
export APPTAINERENV_TEMPLATEFLOW_HOME="/templateflow"
# Prevent concurrent array tasks from racing to re-extract the skeleton zip (Python 3.12 makedirs race)
export APPTAINERENV_TF_AUTOUPDATE=0
FS_DIR="${fs_dir}"

# Parse the participants.tsv file and extract one subject ID from the
# line corresponding to this SLURM task.
# this is getting the sub from \t operator
# subject=$(
#   sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/^([0-9]+)[[:space:]]+([0-9]+)$/\1/p" \
#   ${basedir}/code/${sublist}
# )
# this is get the sub from , operator
subject=$(
  sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/^([^,]+),.*$/\1/p" \
    "${sublist}"
)
ses=$(
  sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/^[^,]+,([^,]+),.*$/\1/p" \
    "${sublist}"
)
scontrol update JobId=${SLURM_JOB_ID} Name="fp_s${subject}_${ses}"

# Per-subject isolated work dir: each container sees only its own /work,
# preventing fmriprep from finding other subjects' config files.
SUBJECT_WORK_DIR=${FMRIPREP_HOST_CACHE}/sub-${subject}
mkdir -p ${SUBJECT_WORK_DIR}

# BIDS input  → /bids  (read-only)
# Output dir  → /output (writable scratch)
# Work/cache  → /work  (per-subject isolated writable scratch)
fs_bind_opt=""
[[ -n "${FS_DIR}" ]] && fs_bind_opt="-B ${FS_DIR}:/fsdir"
APPTAINER_CMD="unset PYTHONPATH && export PYTHONDONTWRITEBYTECODE=1 && apptainer run --cleanenv --no-home --writable-tmpfs\
                 -B ${BIDS_DIR}:/bids \
                 -B ${fp_output_dir}:/output \
                 -B $CODE_DIR:/code \
                 -B ${TEMPLATEFLOW_HOST_HOME}:${APPTAINERENV_TEMPLATEFLOW_HOME}\
                 -B ${SUBJECT_WORK_DIR}:/work \
                 ${fs_bind_opt}\
                 /scratch/tlei/containers/fmriprep_${fp_version}.sif "

# Compose the command line
cmd_anatonly="module load Apptainer/1.3.5 &&  \
     ${APPTAINER_CMD} \
     /bids \
     /output \
     participant \
      --participant-label $subject \
      -w /work/ -vv \
      --fs-license-file /bids/.license \
      --omp-nthreads 10 --nthreads 20 --mem_mb 50000 \
      --skip-bids-validation \
      --anat-only \
      --output-spaces T1w MNI152NLin2009cAsym fsnative fsaverage "

cmd_runall_SE="module load Apptainer/1.3.5 &&  \
     ${APPTAINER_CMD} \
     /bids \
     /output \
     participant \
      --participant-label $subject \
      -w /work/ -vv \
      --fs-license-file /bids/.license \
      --omp-nthreads 10 --nthreads 20 --mem_mb 50000 \
      --skip-bids-validation \
      --fs-subjects-dir /fsdir \
      $(fp_ver_major=$(echo ${fp_version} | cut -d. -f1); [[ ${fp_ver_major} -ge 24 ]] && echo "--force bbr" || echo "--force-bbr") \
      --bold2anat-init auto \
      --output-spaces T1w func MNI152NLin2009cAsym fsnative fsaverage \
      --bids-filter-file /code/fmriprep/bids_filter_sub-${subject}.json "

cmd_runall_ME="module load Apptainer/1.3.5 &&  \
     ${APPTAINER_CMD} \
     /bids \
     /output \
     participant \
      --participant-label $subject \
      -w /work/ -vv \
      --fs-license-file /bids/.license \
      --omp-nthreads 10 --nthreads 20 --mem_mb 70000 \
      --skip-bids-validation \
      --fs-subjects-dir /fsdir \
      $(fp_ver_major=$(echo ${fp_version} | cut -d. -f1); [[ ${fp_ver_major} -ge 24 ]] && echo "--force bbr" || echo "--force-bbr") \
      --bold2anat-init t2w \
      --output-spaces T1w func MNI152NLin2009cAsym fsnative fsaverage \
      --me-t2s-fit-method curvefit \
      --me-output-echos "

# Add these two lines if you had freesurfer run already
#             --force bbr \
#       --dummy-scans 5 \
#       --bids-filter-file /code/fmriprep/bids_filter_sub-${subject}.json \
#       --fs-subjects-dir /base/derivatives/freesurfer \
#  --bids-filter-file /base/bids_filter.json \
#  --use-syn-sdc \
#  --fs-subjects-dir /base/derivatives/fmriprep/analysis-okazaki_correctfmap/sourcedata/freesurfer
# --project-goodvoxels --notrack --mem_mb 60000 --nprocs 16 --omp-nthreads 8 --slice-time-ref 0
#      --fs-subjects-dir /fsdir"
#     --slice-time-ref 0 \
#     --project-goodvoxels \
#     --notrack \
# --stop-on-first-crash \ this is dangerous, since the rendering step can fail for some subjects, but the rest of the pipeline is fine. Use with caution.
# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID}, for subject ${subject}
if [[ -n "${fs_dir}" ]]; then
    echo "fs_dir set — running full pipeline (cmd_runall_SE)"
    echo Commandline: $cmd_runall_SE
    eval $cmd_runall_SE
else
    echo "fs_dir not set — running anat-only"
    echo Commandline: $cmd_anatonly
    eval $cmd_anatonly
fi
exitcode=$?

# Post-job: write one summary line to job_results.tsv in slurm_log_dir.
# Matches the PRF sbatch pattern — single log dir, no rsync needed.
if [[ -n "$slurm_log_dir" ]]; then
    echo -e "$(date +"%Y-%m-%d %H:%M:%S")\tsub-${subject}\tses-${ses}\t${SLURM_JOB_NAME}\t${SLURM_JOB_ID}\texit=${exitcode}" \
        >> "${slurm_log_dir}/job_results.tsv"
fi

echo "Finished sub-${subject} ses-${ses} (array task ${SLURM_ARRAY_TASK_ID}) with exit code ${exitcode}"
exit $exitcode
