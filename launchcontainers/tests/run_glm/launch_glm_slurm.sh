#!/usr/bin/env bash
# =============================================================================
# run_glm_slurm.sh  —  Slurm launcher for run_glm.py
#
# Usage:
#   bash run_glm_slurm.sh -n <analysis_name> -s 01,10
#   bash run_glm_slurm.sh -n <analysis_name> -f subseslist.txt
# =============================================================================

# ---------------------------------------------------------------------------
# Edit these variables before running
# ---------------------------------------------------------------------------
PYTHON_SCRIPT="/scratch/tlei/lc/launchcontainers/tests/run_glm/run_glm.py"
LOGBASE="/scratch/tlei/dipc_glm"

# Slurm resource settings
CPUS="8"
MEM="32G"
TIME="00:40:00"
QOS="regular"           # regular | test
PARTITION="general"

# run_glm.py arguments
BASE="/scratch/tlei/VOTCLOC"
FP_ANA_NAME="25.1.4_newest"
TASK="fLoc"
SPACE="fsnative"
START_SCANS="6"
CONTRAST="/scratch/tlei/lc/launchcontainers/tests/run_glm/contrast_votcloc_all.yaml"
RERUN_MAP="/scratch/tlei/VOTCLOC/BIDS/sourcedata/qc/rerun_check.tsv"   # leave empty "" to skip

# Python / conda environment
# Set CONDA_INIT to the path that makes `conda` available in your shell.
# Do NOT use $(conda info --base) here — it runs at parse time before conda is ready.
# Examples:
#   tlei   : "/home/tlei/soft/miniconda3/etc/profile.d/conda.sh"
#   glerma : "/home/tlei/soft/miniconda3/bin/activate"
CONDA_INIT="/home/tlei/soft/miniconda3/bin/activate"
CONDA_ENV="lc"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage:"
    echo "  bash $0 -o <analysis_name> -s <sub>,<ses>          # single session"
    echo "  bash $0 -o <analysis_name> -f <path/subseslist>    # batch from file (RUN==True only)"
    echo ""
    echo "Required:"
    echo "  -o <analysis_name>   GLM output label (--analysis-name) and log dir suffix"
    exit 1
}

subses_arg=""
file_arg=""
analysis_name=""

while getopts ":o:s:f:" opt; do
    case $opt in
        o) analysis_name="$OPTARG"  ;;
        s) subses_arg="$OPTARG"   ;;
        f) file_arg="$OPTARG"     ;;
        *) usage ;;
    esac
done

if [[ -z "$analysis_name" ]]; then
    echo "Error: -n <analysis_name> is required"
    usage
fi

if [[ -z "$subses_arg" && -z "$file_arg" ]]; then
    usage
fi

# ---------------------------------------------------------------------------
# Log directory:  <LOGBASE>/<date>_<analysis_name>/
# ---------------------------------------------------------------------------
LOG_DIR="${LOGBASE}/$(date +"%Y-%m-%d")_${analysis_name}"
mkdir -p "${LOG_DIR}"
chmod -R 777 "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Build sub/ses pairs
# ---------------------------------------------------------------------------
if [[ -n "$subses_arg" ]]; then
    mapfile -t PAIRS < <(echo "$subses_arg")
else
    if [[ ! -f "$file_arg" ]]; then
        echo "Error: subseslist not found: $file_arg"
        exit 1
    fi
    mapfile -t PAIRS < <(awk -F',' 'NR==1{
        for(i=1;i<=NF;i++) if(tolower($i)=="run") runcol=i
        next
    }
    {
        if(runcol && $runcol!="True") next
        printf "%s,%s\n", $1, $2
    }' "$file_arg")
fi

# Save a copy of this launcher for reproducibility
cp "$0" "${LOG_DIR}/launcher_$(date +"%Y-%m-%d_%H-%M-%S").sh"
[[ -n "$file_arg" ]] && cp "$file_arg" "${LOG_DIR}/subseslist.txt"

SOURCE="${subses_arg:-$file_arg}"
echo "============================================================"
echo "  GLM Slurm launcher"
echo "  Analysis name : ${analysis_name}"
echo "  Input       : ${SOURCE}"
echo "  Sessions    : ${#PAIRS[@]}"
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
# Submit one job per sub/ses
# ---------------------------------------------------------------------------
job_num=1
for pair in "${PAIRS[@]}"; do
    SUB=$(echo "$pair" | cut -d',' -f1 | tr -d ' ')
    SES=$(echo "$pair" | cut -d',' -f2 | tr -d ' ')

    # Build the python command
    PY_CMD="python ${PYTHON_SCRIPT} --base ${BASE} -s ${SUB},${SES} --fp-ana-name ${FP_ANA_NAME} --task ${TASK} --space ${SPACE} --start-scans ${START_SCANS} --contrast ${CONTRAST} --analysis-name ${analysis_name}"
    if [[ -n "${RERUN_MAP}" ]]; then
        PY_CMD="${PY_CMD} --rerun-map ${RERUN_MAP}"
    fi

    now=$(date +"%H-%M")
    OUT_FILE="${LOG_DIR}/${now}_%j_sub-${SUB}_ses-${SES}.o"
    ERR_FILE="${LOG_DIR}/${now}_%j_sub-${SUB}_ses-${SES}.e"

    JOBID=$(sbatch \
        --job-name="glm_${SUB}_${SES}" \
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
echo '  ses         : ${SES}'
echo '  analysis_name : ${analysis_name}'
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

    echo "  [${job_num}/${#PAIRS[@]}]  sub-${SUB} ses-${SES}  → job ${JOBID}  (.o/.e in ${LOG_DIR})"
    ((job_num++))
done

echo ""
echo "All ${#PAIRS[@]} jobs submitted. Logs: ${LOG_DIR}"
