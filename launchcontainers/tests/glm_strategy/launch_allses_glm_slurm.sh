#!/usr/bin/env bash
# =============================================================================
# launch_allses_glm_slurm.sh  —  Slurm launcher for run_allses_glm.py
#
# Submits ONE job per subject.  All sessions for that subject are concatenated
# into a single GLM inside the job.  Sessions are resolved in priority order:
#
#   1. --sessions in run_allses_glm.py is NOT used here — the python script
#      auto-detects valid sessions from fMRIprep when -f is passed.
#   2. -f <subseslist>  →  unique subjects extracted here; file forwarded to
#      run_allses_glm.py which filters sessions by subject automatically.
#   3. -s <sub>         →  single subject; python script auto-detects sessions
#      from fMRIprep (no -f forwarded).
#
# Usage:
#   bash launch_allses_glm_slurm.sh -o <output_name> -s 09
#   bash launch_allses_glm_slurm.sh -o <output_name> -f subseslist.txt
# =============================================================================

# ---------------------------------------------------------------------------
# Edit these variables before running
# ---------------------------------------------------------------------------
PYTHON_SCRIPT="/scratch/tlei/lc/launchcontainers/tests/run_allses_glm.py"
LOGBASE="/scratch/tlei/Japan/logs/glm_allses/surface"

# Slurm resource settings
# first_level_from_bids is replaced by direct file reads, so memory is now
# dominated by the data arrays (~7 GB/hemi for 10 ses) + layouts (~5 GB).
# 32G is sufficient; bump to 48G if jobs still OOM.
CPUS="10"
MEM="96G"
TIME="00:40:00"
QOS="regular"           # regular | test
PARTITION="general"

# run_allses_glm.py arguments
BASE="/scratch/tlei/Japan"
FP_ANA_NAME="25.1.4_japan26ses"
TASK="fLoc"
SPACE="fsnative"
START_SCANS="6"
CONTRAST="/scratch/tlei/lc/launchcontainers/tests/glm_strategy/contrast_Japan_all.yaml"
RERUN_MAP=""   # leave empty "" to skip
INPUT_DIR="BIDS_bcbl"

# ---------------------------------------------------------------------------
# Per-subject session overrides
# When a subject is listed here, ONLY these sessions are used — ignoring both
# the subseslist file and auto-detection from fMRIprep.
# Format:  ["<sub>"]="<ses1>,<ses2>,..."   (zero-padded, comma-separated)
# ---------------------------------------------------------------------------
declare -A SESSION_OVERRIDES
# Add per-subject session overrides here if needed, e.g.:
# SESSION_OVERRIDES["05"]="day1VA,day1VB,day2VA,day2VB,day3PF"

# Python / conda environment
CONDA_INIT="/home/tlei/soft/miniconda3/bin/activate"
CONDA_ENV="lc"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage:"
    echo "  bash $0 -o <output_name> -s <sub>                   # single subject (auto-detect sessions)"
    echo "  bash $0 -o <output_name> -f <path/subseslist>       # batch from file (RUN==True only)"
    echo ""
    echo "Required:"
    echo "  -o <output_name>   GLM output label (--output-name) and log dir suffix"
    echo ""
    echo "Optional:"
    echo "  -p <space>         Space: T1w | fsnative | fsaverage | MNI152NLin2009cAsym (default: ${SPACE})"
    exit 1
}

sub_arg=""
file_arg=""
output_name=""

while getopts ":o:s:f:p:" opt; do
    case $opt in
        o) output_name="$OPTARG" ;;
        s) sub_arg="$OPTARG"     ;;
        f) file_arg="$OPTARG"    ;;
        p) SPACE="$OPTARG"       ;;
        *) usage ;;
    esac
done

if [[ -z "$output_name" ]]; then
    echo "Error: -o <output_name> is required"
    usage
fi

if [[ -z "$sub_arg" && -z "$file_arg" ]]; then
    usage
fi

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
LOG_DIR="${LOGBASE}/$(date +"%Y-%m-%d")_${output_name}"
mkdir -p "${LOG_DIR}"
chmod -R 777 "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Build subject list
# Single subject: SUBS=("09")
# From file: extract unique subjects where RUN==True
# ---------------------------------------------------------------------------
if [[ -n "$sub_arg" ]]; then
    SUB=$(printf "%02d" "$((10#${sub_arg// /}))")
    SUBS=("$SUB")
else
    if [[ ! -f "$file_arg" ]]; then
        echo "Error: subseslist not found: $file_arg"
        exit 1
    fi
    mapfile -t SUBS < <(awk -F',' '
        NR==1 {
            for(i=1;i<=NF;i++) {
                if(tolower($i)=="run")  runcol=i
                if(tolower($i)=="sub")  subcol=i
            }
            next
        }
        {
            if(runcol && $runcol!="True") next
            sub=$subcol
            gsub(/ /, "", sub)
            printf "%02d\n", sub+0
        }
    ' "$file_arg" | sort -u)
fi

# Save a copy of this launcher for reproducibility
cp "$0" "${LOG_DIR}/launcher_$(date +"%Y-%m-%d_%H-%M-%S").sh"
[[ -n "$file_arg" ]] && cp "$file_arg" "${LOG_DIR}/subseslist.txt"

SOURCE="${sub_arg:-$file_arg}"
echo "============================================================"
echo "  All-sessions GLM Slurm launcher"
echo "  Output name : ${output_name}"
echo "  Input       : ${SOURCE}"
echo "  Subjects    : ${#SUBS[@]}  (${SUBS[*]})"
echo "  Task        : ${TASK}"
echo "  Start scans : ${START_SCANS}"
echo "  Space       : ${SPACE}"
echo "  QOS         : ${QOS}"
echo "  Partition   : ${PARTITION}"
echo "  CPUs / job  : ${CPUS}"
echo "  Mem / job   : ${MEM}"
echo "  Time / job  : ${TIME}"
echo "  Log dir     : ${LOG_DIR}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Submit one job per subject
# ---------------------------------------------------------------------------
job_num=1
for SUB in "${SUBS[@]}"; do

    # Build python command
    PY_CMD="python ${PYTHON_SCRIPT} --base ${BASE} --sub ${SUB} --fp-ana-name ${FP_ANA_NAME} --task ${TASK} --space ${SPACE} --start-scans ${START_SCANS} --contrast ${CONTRAST} --output-name ${output_name} --input-dir ${INPUT_DIR}"

    # Session source priority:
    #   1. SESSION_OVERRIDES[sub]  →  pass --sessions explicitly
    #   2. -f subseslist file      →  forward file, python filters by sub
    #   3. (neither)               →  python auto-detects from fMRIprep
    if [[ -n "${SESSION_OVERRIDES[$SUB]+x}" ]]; then
        PY_CMD="${PY_CMD} --sessions ${SESSION_OVERRIDES[$SUB]}"
    elif [[ -n "${file_arg}" ]]; then
        PY_CMD="${PY_CMD} -f ${file_arg}"
    fi

    if [[ -n "${RERUN_MAP}" ]]; then
        PY_CMD="${PY_CMD} --rerun-map ${RERUN_MAP}"
    fi

    now=$(date +"%H-%M")
    OUT_FILE="${LOG_DIR}/${now}_%j_sub-${SUB}.o"
    ERR_FILE="${LOG_DIR}/${now}_%j_sub-${SUB}.e"

    JOBID=$(sbatch \
        --job-name="allses_glm_${SUB}" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --qos="${QOS}" \
        --partition="${PARTITION}" \
        --output="${OUT_FILE}" \
        --error="${ERR_FILE}" \
        --wrap="
set -euo pipefail
source \"${CONDA_INIT}\"
conda activate ${CONDA_ENV}

echo '============================================================'
echo '  Job ID      : '\${SLURM_JOB_ID}
echo '  sub         : ${SUB}'
echo '  sessions    : ${SESSION_OVERRIDES[$SUB]:-"auto / from file"}'
echo '  output_name : ${output_name}'
echo '  Node        : '\$(hostname)
echo '  Start       : '\$(date '+%Y-%m-%d %H:%M:%S')
echo '============================================================'
echo ''

time ${PY_CMD}
EXIT_CODE=\$?

echo ''
echo '============================================================'
echo '  End         : '\$(date '+%Y-%m-%d %H:%M:%S')
echo '  Exit code   : '\${EXIT_CODE}
echo '============================================================'
exit \${EXIT_CODE}
" | awk '{print $NF}')

    echo "  [${job_num}/${#SUBS[@]}]  sub-${SUB}  → job ${JOBID}  (.o/.e in ${LOG_DIR})"
    ((job_num++))
done

echo ""
echo "All ${#SUBS[@]} job(s) submitted. Logs: ${LOG_DIR}"
