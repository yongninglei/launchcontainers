#!/usr/bin/env bash
# =============================================================================
# launch_glm_local.sh  —  Local launcher for run_glm.py
#
# Usage:
#   bash launch_glm_local.sh -n <analysis_name> -s 01,10
#   bash launch_glm_local.sh -n <analysis_name> -f subseslist.txt
#   bash launch_glm_local.sh -n <analysis_name> -f subseslist.txt --dry-run
# =============================================================================

# ---------------------------------------------------------------------------
# Edit these variables before running
# ---------------------------------------------------------------------------
PYTHON_SCRIPT="/export/home/tlei/tlei/soft/launchcontainers/launchcontainers/tests/run_glm/run_glm.py"
LOGBASE="/bcbl/home/public/Gari/VOTCLOC/main_exp/logs/l1_surface"

# run_glm.py arguments
BASE="/bcbl/home/public/Gari/VOTCLOC/main_exp"
FP_ANA_NAME="25.1.4_t2w_fmapsbref_newest"
TASK="fLoc"
SPACE="fsnative"
START_SCANS="6"
CONTRAST="/export/home/tlei/tlei/soft/launchcontainers/launchcontainers/tests/run_glm/contrast_votcloc_all.yaml"
RERUN_MAP="/bcbl/home/public/Gari/VOTCLOC/main_exp/BIDS/sourcedata/qc/rerun_check.tsv"   # leave empty "" to skip
INPUT_DIR="BIDS"           # input BIDS dir name under BASE; use BIDS_WC for WC runs

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage:"
    echo "  bash $0 -n <analysis_name> -s <sub>,<ses>         # single session"
    echo "  bash $0 -n <analysis_name> -f <path/subseslist>   # batch from file (RUN==True only)"
    echo ""
    echo "  Optional:"
    echo "    -i <input_dir>   BIDS dir name under BASE (default: BIDS, use BIDS_WC for WC)"
    echo "    --dry-run        print commands without running"
    echo "    --use-smoothed"
    echo ""
    echo "Required:"
    echo "  -n <analysis_name>   maps to --analysis-name in run_glm.py"
    exit 1
}

subses_arg=""
file_arg=""
analysis_name=""
dry_run=0
extra_flags=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n) analysis_name="$2"; shift 2 ;;
        -s) subses_arg="$2";    shift 2 ;;
        -f) file_arg="$2";      shift 2 ;;
        -i) INPUT_DIR="$2"; shift 2 ;;
        --dry-run)      dry_run=1; extra_flags="${extra_flags} --dry-run"; shift ;;
        --use-smoothed) extra_flags="${extra_flags} --use-smoothed"; shift ;;
        *) echo "Unknown option: $1"; usage ;;
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
# Log directory
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
echo "  GLM local launcher"
echo "  Analysis name : ${analysis_name}"
echo "  Input         : ${SOURCE}"
echo "  Sessions      : ${#PAIRS[@]}"
echo "  Task          : ${TASK}"
echo "  Start scans   : ${START_SCANS}"
echo "  Space         : ${SPACE}"
echo "  Input dir     : ${INPUT_DIR}"
echo "  Log dir       : ${LOG_DIR}"
echo "  Dry run       : ${dry_run}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Run sessions sequentially
# ---------------------------------------------------------------------------
job_num=1
for pair in "${PAIRS[@]}"; do
    SUB=$(printf "%02d" "$((10#$(echo "$pair" | cut -d',' -f1 | tr -d ' ')))")
    SES=$(printf "%02d" "$((10#$(echo "$pair" | cut -d',' -f2 | tr -d ' ')))")

    LOG_OUT="${LOG_DIR}/$(date +"%H-%M")_sub-${SUB}_ses-${SES}.o"
    LOG_ERR="${LOG_DIR}/$(date +"%H-%M")_sub-${SUB}_ses-${SES}.e"

    PY_CMD="python ${PYTHON_SCRIPT} \
        --base ${BASE} \
        -s ${SUB},${SES} \
        --fp-ana-name ${FP_ANA_NAME} \
        --task ${TASK} \
        --space ${SPACE} \
        --start-scans ${START_SCANS} \
        --contrast ${CONTRAST} \
        --analysis-name ${analysis_name} \
        --input-dir ${INPUT_DIR}"

    [[ -n "${RERUN_MAP}" ]] && PY_CMD="${PY_CMD} --rerun-map ${RERUN_MAP}"
    [[ -n "${extra_flags}" ]] && PY_CMD="${PY_CMD} ${extra_flags}"

    echo "  [${job_num}/${#PAIRS[@]}]  sub-${SUB} ses-${SES}  → log: ${LOG_OUT}"

    {
        echo "============================================================"
        echo "  sub         : ${SUB}"
        echo "  ses         : ${SES}"
        echo "  analysis    : ${analysis_name}"
        echo "  Start       : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"
        echo ""
        time ${PY_CMD}
        EXIT_CODE=$?
        echo ""
        echo "============================================================"
        echo "  End         : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  Exit code   : ${EXIT_CODE}"
        echo "============================================================"
    } >"${LOG_OUT}" 2>"${LOG_ERR}"

    ((job_num++))
done

echo ""
echo "All ${#PAIRS[@]} sessions done. Logs: ${LOG_DIR}"
