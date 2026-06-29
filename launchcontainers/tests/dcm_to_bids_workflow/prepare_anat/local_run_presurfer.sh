#!/usr/bin/env bash
# MIT License
# Copyright (c) 2024-2025 Yongning Lei

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
script_dir=$(dirname "$(realpath "$0")")

tbPath=/export/home/tlei/tlei/toolboxes
force=true

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage:"
    echo "  $0 -b <basedir> -i <src_dir> -s <sub>,<ses>"
    echo "  $0 -b <basedir> -i <src_dir> -f <full_path_to_subseslist>"
    echo ""
    echo "  -b  project base directory (logs go to <basedir>/logs/)"
    echo "  -i  input raw nifti directory (relative to basedir)"
    echo "  -s  single sub,ses pair"
    echo "  -f  batch mode: path to subseslist file"
    exit 1
}

basedir=""
src_dir=""
subses_arg=""
file_arg=""

while getopts ":b:i:s:f:" opt; do
    case $opt in
        b) basedir="$OPTARG" ;;
        i) src_dir="$OPTARG" ;;
        s) subses_arg="$OPTARG" ;;
        f) file_arg="$OPTARG" ;;
        *) usage ;;
    esac
done

if [[ -z "$basedir" || -z "$src_dir" ]]; then
    echo "Error: -b <basedir> and -i <src_dir> are required."
    usage
fi

if [[ -z "$subses_arg" && -z "$file_arg" ]]; then
    usage
fi

src_dir=${basedir}/${src_dir}
outputdir=${basedir}/BIDS

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
step=presurfer
logdir=${basedir}/logs/log_${step}/${analysis_name}_$(date +"%Y-%m-%d")
echo "The logdir is $logdir"
echo "The outputdir is $outputdir"
mkdir -p "$logdir"

cp "$0" "$logdir"
[[ -n "$file_arg" ]] && cp "$subseslist_path" "$logdir/subseslist.txt"

# ---------------------------------------------------------------------------
# Run jobs locally
# ---------------------------------------------------------------------------
while IFS=',' read -r sub ses _; do
    [[ -z "$sub" || -z "$ses" ]] && continue

    echo "### PRESURFER: sub-${sub} ses-${ses} ###"
    now=$(date +"%H;%M")
    log_file="${logdir}/presurfer_${sub}_${ses}_${now}.o"
    error_file="${logdir}/presurfer_${sub}_${ses}_${now}.e"

    export tbPath src_dir outputdir sub ses force script_dir

    cmd="bash $script_dir/src_${step}.sh"
    echo "$cmd"
    eval "$cmd" > "${log_file}" 2> "${error_file}"

done < "$tmpfile"

rm -f "$tmpfile"
