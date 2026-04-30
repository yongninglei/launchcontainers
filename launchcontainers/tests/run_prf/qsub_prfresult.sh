#!/usr/bin/env bash
# MIT License
# Copyright (c) 2024-2025 Yongning Lei

# ---------------------------------------------------------------------------
# Configuration — IPS / BCBL
# ---------------------------------------------------------------------------
baseP="/bcbl/home/public/Gari/VOTCLOC/main_exp"
codedir="$baseP/code"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOME_DIR="$baseP/singularity_home"
license_path="$baseP/BIDS/.license"

step="prfresult"
version="1.0"
queue="short.q"
mem="16G"
cpus="10"
time="01:00:00"

json_dir="$baseP/code/${step}_jsons"
sif_path="/bcbl/home/public/Gari/singularity_images/${step}_${version}.sif"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage:"
    echo "  $0 -n <log_note> -s <sub>,<ses>               # single sub/ses pair"
    echo "  $0 -n <log_note> -f <full_path_to_subseslist> # batch from file"
    echo ""
    echo "Required:"
    echo "  -n <log_note>   Short label written into the log directory name."
    echo ""
    echo "Optional:"
    echo "  -t <task>       Run only this task (e.g. retCB) instead of auto-detecting"
    echo "                  all tasks from available JSONs. Useful for reruns."
    exit 1
}

subses_arg=""
file_arg=""
log_note=""
task_override=""

while getopts ":n:s:f:t:" opt; do
    case $opt in
        n) log_note="$OPTARG"      ;;
        s) subses_arg="$OPTARG"    ;;
        f) file_arg="$OPTARG"      ;;
        t) task_override="$OPTARG" ;;
        *) usage ;;
    esac
done

if [[ -z "$log_note" ]]; then
    echo "Error: -n <log_note> is required"
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
LOG_DIR="$baseP/log/ips_${step}_logs/$(date +"%Y-%m-%d")_${log_note}_${analysis_name}"
mkdir -p "$LOG_DIR"
mkdir -p "$HOME_DIR"

cp "$0" "$LOG_DIR/"
[[ -n "$file_arg" ]] && cp "$subseslist_path" "$LOG_DIR/subseslist.txt"

echo "Log dir     : $LOG_DIR"
echo "JSON dir    : $json_dir"
echo "Log note    : $log_note"
if [[ -n "$task_override" ]]; then
    echo "Task (fixed): $task_override"
else
    echo "Task        : auto-detect from JSONs"
fi
echo ""

# ---------------------------------------------------------------------------
# Submit jobs — one per detected JSON for each sub/ses
# ---------------------------------------------------------------------------
job_num=1
while IFS=',' read -r sub ses _; do
    [[ -z "$sub" || -z "$ses" ]] && continue

    # Detect tasks: use override if given, otherwise glob all JSONs for this sub/ses
    if [[ -n "$task_override" ]]; then
        json_path="${json_dir}/${task_override}_sub-${sub}_ses-${ses}.json"
        if [[ ! -f "$json_path" ]]; then
            echo "WARNING: JSON not found for task=${task_override} sub-${sub} ses-${ses}: $json_path — skipping"
            continue
        fi
        mapfile -t jsons < <(echo "$json_path")
        echo "sub-${sub} ses-${ses}: using fixed task=${task_override}"
    else
        mapfile -t jsons < <(ls "${json_dir}"/*_sub-${sub}_ses-${ses}.json 2>/dev/null)
        if [[ ${#jsons[@]} -eq 0 ]]; then
            echo "WARNING: no JSONs found for sub-${sub} ses-${ses} — skipping"
            continue
        fi
        echo "sub-${sub} ses-${ses}: found ${#jsons[@]} task(s) (auto-detected)"
    fi

    for json_path in "${jsons[@]}"; do
        # Extract task label from filename: retCB_sub-01_ses-01.json → retCB
        fname=$(basename "$json_path")
        task="${fname%%_sub-*}"

        now=$(date +"%H-%M")
        log_file="${LOG_DIR}/${now}_${sub}-${ses}_${task}.o"
        error_file="${LOG_DIR}/${now}_${sub}-${ses}_${task}.e"

        cmd="qsub -N ${step}_${task}_${job_num} \
            -S /bin/bash \
            -q ${queue} \
            -pe smp ${cpus} \
            -l mem_free=${mem} \
            -l h_rt=${time} \
            -o ${log_file} \
            -e ${error_file} \
            -v baseP=${baseP},license_path=${license_path},version=${version},sub=${sub},ses=${ses},task=${task},json_path=${json_path},sif_path=${sif_path},LOG_DIR=${LOG_DIR} \
            ${script_dir}/${step}_ips.sh"

        echo "  Submitting: sub-${sub} ses-${ses} task=${task}"
        eval "$cmd"

        ((job_num++))
    done

done < "$tmpfile"

rm -f "$tmpfile"

total_jobs=$(( job_num - 1 ))
summary="All ${total_jobs} job(s) submitted. Logs: ${LOG_DIR}"
echo ""
echo "$summary"
echo "$summary" >> "${LOG_DIR}/submit_summary.txt"
