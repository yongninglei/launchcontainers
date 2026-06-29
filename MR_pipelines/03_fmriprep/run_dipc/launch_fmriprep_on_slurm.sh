#!/usr/bin/env bash
# MIT License
# Copyright (c) 2024-2025 Yongning Lei

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

fp_version=25.2.5  # default; override with -v

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage:"
    echo "  $0 -b <basedir> -d <bids_dir> -a <analysis_name> -s <sub>,<ses>"
    echo "  $0 -b <basedir> -d <bids_dir> -a <analysis_name> -f <subseslist_path>"
    echo ""
    echo "  -b  scratch output base directory"
    echo "  -d  BIDS data directory (read-only input)"
    echo "  -a  analysis name"
    echo "  -s  single sub,ses pair"
    echo "  -f  batch mode: path to subseslist file"
    echo "  -v  fMRIPrep version (default: ${fp_version})"
    echo "  -F  FreeSurfer subjects directory (bound as /fsdir inside container)"
    exit 1
}

basedir=""
bids_dir=""
subses_arg=""
file_arg=""
analysis_name=""
fs_dir=""

while getopts ":b:d:a:s:f:v:F:" opt; do
    case $opt in
        b) basedir="$OPTARG" ;;
        d) bids_dir="$OPTARG" ;;
        a) analysis_name="$OPTARG" ;;
        s) subses_arg="$OPTARG" ;;
        f) file_arg="$OPTARG" ;;
        v) fp_version="$OPTARG" ;;
        F) fs_dir="$OPTARG" ;;
        *) usage ;;
    esac
done

if [[ -z "$basedir" || -z "$bids_dir" || -z "$analysis_name" ]]; then
    echo "Error: -b <basedir>, -d <bids_dir>, and -a <analysis_name> are required"
    usage
fi

if [[ -z "$subses_arg" && -z "$file_arg" ]]; then
    usage
fi

# ---------------------------------------------------------------------------
# Logging setup (must happen before building sublist — sublist lives in logdir)
# ---------------------------------------------------------------------------
slurm_log_dir=${basedir}/logs/fmriprep-${fp_version}-${analysis_name}_$(date +"%Y-%m-%d")
mkdir -p "${slurm_log_dir}"

cp "$0" "${slurm_log_dir}/"

# ---------------------------------------------------------------------------
# Build persistent sublist in logdir (SLURM workers read it asynchronously)
# ---------------------------------------------------------------------------
sublist="${slurm_log_dir}/subseslist.txt"

if [[ -n "$subses_arg" ]]; then
    job_name="sub$(echo "$subses_arg" | cut -d',' -f1)ses$(echo "$subses_arg" | cut -d',' -f2)"
    printf "sub,ses\n%s\n" "$subses_arg" > "$sublist"
else
    src_sublist="$file_arg"
    if [[ ! -f "$src_sublist" ]]; then
        echo "Error: subseslist not found: $src_sublist"
        exit 1
    fi
    cp "$src_sublist" "$sublist"
    first_sub=$(awk -F',' 'NR==2{print $1}' "$sublist")
    first_ses=$(awk -F',' 'NR==2{print $2}' "$sublist")
    job_name="fp_s${first_sub}_${first_ses}"
fi

# ---------------------------------------------------------------------------
# Submit SLURM array
# ---------------------------------------------------------------------------
TOTAL_LINES=$(wc -l < "$sublist")
DATA_LINES=$((TOTAL_LINES - 1))

echo ""
echo "========================================"
echo "  fMRIPrep SLURM submission"
echo "========================================"
echo "  analysis    : ${analysis_name}"
echo "  fp_version  : ${fp_version}"
echo "  bids_dir    : ${bids_dir}"
echo "  basedir : ${basedir}"
echo "  fs_dir      : ${fs_dir:-<not set>}"
echo "  log_dir     : ${slurm_log_dir}"
echo "  sublist     : ${sublist}"
echo "  n_jobs      : ${DATA_LINES}"
echo "----------------------------------------"
echo "  subjects:"
awk -F',' 'NR>1 {printf "    [%d] sub-%s  ses-%s\n", NR-1, $1, $2}' "$sublist"
echo "========================================"
echo ""

now=$(date +"%H-%M")

cmd="sbatch \
    --export=analysis_name=${analysis_name},fp_version=${fp_version},slurm_log_dir=${slurm_log_dir},sublist=${sublist},basedir=${basedir},bids_dir=${bids_dir},fs_dir=${fs_dir} \
    --array=1-${DATA_LINES} \
    -J ${job_name} \
    -o ${slurm_log_dir}/%J_%x-%A-%a_${now}.o \
    -e ${slurm_log_dir}/%J_%x-%A-%a_${now}.e \
    ${script_dir}/src_fmriprep.slurm"

echo "sbatch cmd: $cmd"
echo ""
eval "$cmd"
