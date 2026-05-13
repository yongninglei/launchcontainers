#!/usr/bin/env bash

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
nordic_modality="dwi"
script_dir=/export/home/tlei/tlei/soft/launchcontainers/MR_pipelines/01_prepare_nifti/prepare_dwi

TB_PATH="/export/home/tlei/tlei/toolboxes"
SRC_DIR="/bcbl/home/public/Gari/VOTCLOC/main_exp/raw_nifti"
OUTPUT_DIR="/bcbl/home/public/Gari/VOTCLOC/main_exp/BIDS"
codedir=/bcbl/home/public/Gari/VOTCLOC/main_exp/code

# nordic parameters (must match your MATLAB signature)
NORDIC_END=0
FORCE=true
DONORDIC=true

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage:"
    echo "  $0 -s <sub>,<ses>         # single sub/ses pair"
    echo "  $0 -f <full_path_to_subseslist>   # batch mode"
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
    analysis_name="sub$(echo "$subses_arg" | cut -d',' -f1)ses$(echo "$subses_arg" | cut -d',' -f2)"
else
    subseslist_path="$file_arg"
    if [[ ! -f "$subseslist_path" ]]; then
        echo "Error: subseslist not found: $subseslist_path"
        exit 1
    fi
    tail -n +2 "$subseslist_path" > "$tmpfile"
    analysis_name=$(basename "$file_arg" | sed 's/\.[^.]*$//')
fi

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
logdir=${OUTPUT_DIR}/log_nordic_${nordic_modality}/${analysis_name}_$(date +"%Y-%m-%d")
echo "The logdir is $logdir"
echo "The outputdir is $OUTPUT_DIR"
mkdir -p "$logdir"

cp "$0" "$logdir"
[[ -n "$file_arg" ]] && cp "$subseslist_path" "$logdir/subseslist.txt"

# ---------------------------------------------------------------------------
# Run jobs locally
# ---------------------------------------------------------------------------
while IFS=',' read -r sub ses _; do
    [[ -z "$sub" || -z "$ses" ]] && continue

    echo "=== Running sub-${sub} ses-${ses} ==="
    now=$(date +"%H;%M")
    log_file="${logdir}/nordic_${sub}_${ses}_${now}.o"
    error_file="${logdir}/nordic_${sub}_${ses}_${now}.e"

    cmd="bash $script_dir/src_nordic_${nordic_modality}.sh \
        ${TB_PATH} ${SRC_DIR} ${OUTPUT_DIR} \
        ${sub} ${ses} \
        ${NORDIC_END} ${DONORDIC} ${FORCE} \
        ${script_dir}"
    eval "$cmd" > "${log_file}" 2> "${error_file}"

done < "$tmpfile"

rm -f "$tmpfile"
