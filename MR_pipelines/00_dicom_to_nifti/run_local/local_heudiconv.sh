#!/usr/bin/env bash
# MIT License
# Copyright (c) 2024-2025 Yongning Lei

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
project=IRAKEINU
basedir=/bcbl/home/public/Gari/$project
codedir=$basedir/code

outputdir=$basedir/raw_nifti
dcm_dir=/base/dicom
script_dir=/export/home/tlei/tlei/soft/launchcontainers/MR_pipelines/00_dicom_to_nifti
heuristicfile=$script_dir/heuristic/heuristic_all_${project}_ME.py
sing_path=/bcbl/home/public/Gari/containers/heudiconv_1.3.4.sif

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage:"
    echo "  $0 -t <step> -s <sub>,<ses>         # single sub/ses pair"
    echo "  $0 -t <step> -f <subseslist_name>   # batch from codedir/<subseslist_name>"
    echo ""
    echo "  -t  heudiconv step: step1 or step2"
    exit 1
}

step=""
subses_arg=""
file_arg=""

while getopts ":t:s:f:" opt; do
    case $opt in
        t) step="$OPTARG" ;;
        s) subses_arg="$OPTARG" ;;
        f) file_arg="$OPTARG" ;;
        *) usage ;;
    esac
done

if [[ -z "$step" ]]; then
    echo "Error: -t <step> is required (step1 or step2)."
    usage
fi

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
logdir=${outputdir}/log_heudiconv/${analysis_name}_$(date +"%Y-%m-%d")/${step}
echo "The logdir is $logdir"
echo "The outputdir is $outputdir"
mkdir -p "$logdir"

cp "$0" "$logdir"
cp "$script_dir/src_heudiconv_${step}.sh" "$logdir"
[[ -n "$file_arg" ]] && cp "$subseslist_path" "$logdir/subseslist.txt"

# ---------------------------------------------------------------------------
# Run jobs locally
# ---------------------------------------------------------------------------
while IFS=',' read -r sub ses _; do
    [[ -z "$sub" || -z "$ses" ]] && continue

    echo "### CONVERTING TO NIFTI: sub-${sub} ses-${ses} ###"
    now=$(date +"%H_%M")
    log_file="${logdir}/local_${sub}_${ses}_${now}.o"
    error_file="${logdir}/local_${sub}_${ses}_${now}.e"

    export basedir logdir dcm_dir outputdir sub ses heuristicfile sing_path

    cmd="bash $script_dir/src_heudiconv_${step}.sh"
    echo "$cmd"
    eval "$cmd" > "${log_file}" 2> "${error_file}"

done < "$tmpfile"

rm -f "$tmpfile"
