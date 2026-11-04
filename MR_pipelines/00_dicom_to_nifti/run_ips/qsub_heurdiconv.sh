#!/usr/bin/env bash
# MIT License
# Copyright (c) 2024-2025 Yongning Lei

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
project=votcloc
basedir=/bcbl/home/public/Gari/VOTCLOC/main_exp
codedir=/bcbl/home/public/Gari/VOTCLOC/main_exp/code

outputdir=$basedir/raw_nifti_fix_fmap
dcm_dir=/base/dicom
script_dir=/bcbl/home/home_n-z/tlei/soft/launchcontainers/MR_pipelines/00_dicom_to_nifti
heuristicfile=$script_dir/heuristic/heuristic_sbref_${project}.py
sing_path=/bcbl/home/public/Gari/containers/heudiconv_1.3.4.sif

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage:"
    echo "  $0 -t <step> -s <sub>,<ses>         # single sub/ses pair"
    echo "  $0 -t <step> -f <subseslist_path>    # batch from full path to subseslist"
    echo ""
    echo "  -t  heudiconv step: step1 or step2"
    echo "  -H  heuristic file name under heuristic/ (optional, overrides default)"
    echo "  -o  output dir name under basedir (optional, overrides default)"
    exit 1
}

step=""
subses_arg=""
file_arg=""

while getopts ":t:s:f:H:o:" opt; do
    case $opt in
        t) step="$OPTARG" ;;
        s) subses_arg="$OPTARG" ;;
        f) file_arg="$OPTARG" ;;
        H) heuristicfile="$script_dir/heuristic/$OPTARG" ;;
        o) outputdir="$basedir/$OPTARG" ;;
        *) usage ;;
    esac
done

if [[ ! -f "$heuristicfile" ]]; then
    echo "Error: heuristic not found: $heuristicfile"
    exit 1
fi

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
# Submit jobs
# ---------------------------------------------------------------------------
while IFS=',' read -r sub ses _; do
    [[ -z "$sub" || -z "$ses" ]] && continue

    echo "### CONVERTING TO NIFTI: sub-${sub} ses-${ses} ###"
    now=$(date +"%H-%M")
    log_file="${logdir}/qsub_${sub}_${ses}_${now}.o"
    error_file="${logdir}/qsub_${sub}_${ses}_${now}.e"

    cmd="qsub -q short.q \
        -S /bin/bash \
        -N s${sub}_t${ses}_dcm \
        -o $log_file \
        -e $error_file \
        -l mem_free=16G \
        -v basedir=${basedir},logdir=${logdir},dcm_dir=$dcm_dir,outputdir=${outputdir},sub=${sub},ses=${ses},heuristicfile=$heuristicfile,sing_path=$sing_path \
        $script_dir/src_heudiconv_${step}.sh"

    echo "$cmd"
    eval "$cmd"

done < "$tmpfile"

rm -f "$tmpfile"
