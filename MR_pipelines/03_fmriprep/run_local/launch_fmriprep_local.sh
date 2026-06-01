#!/usr/bin/env bash
# MIT License
# Copyright (c) 2020-2023 Garikoitz Lerma-Usabiaga
# Copyright (c) 2022-2025 Yongning Lei

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
basedir="/bcbl/home/public/Gari/IRAKEINU"
codedir="${basedir}/code"
analysis_name="IRpilot"
fp_version=25.1.4

CODE_DIR="${basedir}/code"
BIDS_DIR="${basedir}/BIDS"
OUTPUT_DIR=derivatives/fmriprep-${fp_version}_${analysis_name}
DERIVS_DIR="${BIDS_DIR}/${OUTPUT_DIR}"
LOCAL_FREESURFER_DIR="${basedir}/BIDS/derivatives/freesurfer"
# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage:"
    echo "  $0 -s <sub>,<ses>         # single sub/ses pair"
    echo "  $0 -f <subseslist_name>   # batch from codedir/<subseslist_name>"
    echo ""
    echo "## Note: fMRIPrep processes all sessions for a subject at once."
    echo "## Only the sub label is passed to --participant-label."
    exit 1
}

subses_arg=""
file_arg=""

while getopts ":s:f:" opt; do
    case $opt in
        s) subses_arg="$OPTARG" ;;
        f) file_arg="$OPTARG" ;;
        *) usage ;;
    esac
done

if [[ -z "$subses_arg" && -z "$file_arg" ]]; then
    usage
fi

# ---------------------------------------------------------------------------
# Build sub/ses list
# ---------------------------------------------------------------------------
tmpfile=$(mktemp)

if [[ -n "$subses_arg" ]]; then
    echo "$subses_arg" > "$tmpfile"
else
    subseslist_path="$codedir/$file_arg"
    if [[ ! -f "$subseslist_path" ]]; then
        echo "Error: subseslist not found: $subseslist_path"
        exit 1
    fi
    tail -n +2 "$subseslist_path" > "$tmpfile"
fi

# ---------------------------------------------------------------------------
# Singularity / cache setup
# ---------------------------------------------------------------------------
export cache_dir=$basedir/fmriprep_tmps_$analysis_name
LOG_DIR=$DERIVS_DIR/logs

TEMPLATEFLOW_HOST_HOME=$cache_dir/.cache/templateflow
FMRIPREP_HOST_CACHE=$cache_dir/.cache/fmriprep
mkdir -p "${TEMPLATEFLOW_HOST_HOME}"
mkdir -p "${FMRIPREP_HOST_CACHE}"
mkdir -p "${LOG_DIR}"
mkdir -p "${DERIVS_DIR}"

export SINGULARITYENV_FS_LICENSE=/export/home/tlei/tlei/linux_settings/license.txt
export SINGULARITYENV_TEMPLATEFLOW_HOME="/templateflow"

# ---------------------------------------------------------------------------
# Run fMRIPrep
# ---------------------------------------------------------------------------
while IFS=',' read -r sub ses; do
    [[ -z "$sub" || -z "$ses" ]] && continue

    echo "### fMRIPrep: sub-${sub} (all sessions) ###"
    now=$(date +"%Y-%m-%dT%H:%M")

    # Per-subject isolated work dir: prevents fmriprep from finding other
    # subjects' config files (avoids skull_strip_template resume bug).
    SUBJECT_WORK_DIR=${FMRIPREP_HOST_CACHE}/sub-${sub}
    mkdir -p "${SUBJECT_WORK_DIR}"

    SINGULARITY_CMD="unset PYTHONPATH && singularity run --cleanenv --no-home \
                         --containall --writable-tmpfs \
                     -B /bcbl:/bcbl \
                     -B /export:/export \
                     -B $BIDS_DIR:/base \
                     -B $CODE_DIR:/code \
                     -B ${LOCAL_FREESURFER_DIR}:/fsdir \
                     -B ${TEMPLATEFLOW_HOST_HOME}:${SINGULARITYENV_TEMPLATEFLOW_HOME} \
                     -B ${SUBJECT_WORK_DIR}:/work \
                     /bcbl/home/public/Gari/containers/fmriprep_${fp_version}.sif"

    cmd="module load apptainer/latest && \
         ${SINGULARITY_CMD} \
         /base \
         /base/${OUTPUT_DIR} \
         participant --participant-label ${sub} \
         -w /work/ -vv \
         --fs-license-file ${SINGULARITYENV_FS_LICENSE} \
         --omp-nthreads 20 --nthreads 50 --mem_mb 80000 \
         --skip-bids-validation \
         --fs-subjects-dir /fsdir \
         --force bbr \
         --bold2anat-init t2w \
         --output-spaces T1w func MNI152NLin2009cAsym fsnative fsaverage \
         --bids-filter-file /code/bids_filter.json \
         > ${LOG_DIR}/${analysis_name}_sub-${sub}_${now}.o \
         2> ${LOG_DIR}/${analysis_name}_sub-${sub}_${now}.e"

    #     --bids-filter-file /base/code/bids_filter.json \
    #     --fs-subjects-dir /base/BIDS/derivatives/freesurfer/analysis-${analysis_name}
    #     --dummy-scans 6 \
    #     --use-syn-sdc \

    echo "Running sub-${sub}"
    echo "$cmd"
    eval "$cmd"

done < "$tmpfile"

rm -f "$tmpfile"
