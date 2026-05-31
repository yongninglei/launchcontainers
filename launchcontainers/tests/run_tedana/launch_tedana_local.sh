#!/usr/bin/env bash
# =============================================================================
# launch_tedana_local.sh  —  Local launcher for run_tedana.py
#
# Usage:
#   bash launch_tedana_local.sh -n <analysis> -c <cores> -r <runs> -s sub,ses
#   bash launch_tedana_local.sh -n <analysis> -c <cores> -r <runs> -f subseslist.txt
#   bash launch_tedana_local.sh -n <analysis> -c 40 -r 3 -f subseslist.txt --dry-run
#
# Core allocation (calculated automatically):
#   N_RUN_JOBS = min(runs, cores)          # runs processed in parallel
#   N_THREADS  = cores / N_RUN_JOBS        # tedana ICA threads per run
# =============================================================================

# ---------------------------------------------------------------------------
# Fixed project settings  (edit these, not the CLI flags)
# ---------------------------------------------------------------------------
BIDS_DIR="/bcbl/home/public/Gari/IRAKEINU/BIDS"
FP_DIR="${BIDS_DIR}/derivatives/fmriprep-25.1.4_IRpilot"
OUT_DIR_NAME="tedana-26.0.3"
TASKS="BfLocVideo"
ACQ="ME"
FITTYPE="curvefit"
LOGBASE="${BIDS_DIR}/../logs/tedana"
ANALYSIS_NAME=""   # set via -n

PYTHON="micromamba run -n lc python"
SCRIPT="$(dirname "$0")/run_tedana.py"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage:"
    echo "  bash $0 -n <analysis> -c <cores> -r <runs_per_session> -s <sub>,<ses>"
    echo "  bash $0 -n <analysis> -c <cores> -r <runs_per_session> -f <subseslist>"
    echo ""
    echo "Required:"
    echo "  -n <analysis>   analysis label  (analysis-{name} under output dir)"
    echo "  -c <cores>      total CPU cores available on this machine"
    echo "  -r <runs>       number of ME runs per session"
    echo "  -s <sub>,<ses>  single session"
    echo "  -f <file>       batch file  (Run==True rows only)"
    echo ""
    echo "Optional:"
    echo "  --dry-run       print plan without running"
    echo "  --overwrite     overwrite existing tedana outputs"
    exit 1
}

TOTAL_CORES=""
N_RUNS=""
subses_arg=""
file_arg=""
dry_run=0
extra_flags=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n) ANALYSIS_NAME="$2"; shift 2 ;;
        -c) TOTAL_CORES="$2";   shift 2 ;;
        -r) N_RUNS="$2";        shift 2 ;;
        -s) subses_arg="$2";    shift 2 ;;
        -f) file_arg="$2";      shift 2 ;;
        --dry-run)   dry_run=1; extra_flags="${extra_flags} --dry-run";  shift ;;
        --overwrite) extra_flags="${extra_flags} --overwrite"; shift ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Validate required args
for flag_name in ANALYSIS_NAME TOTAL_CORES N_RUNS; do
    if [[ -z "${!flag_name}" ]]; then
        echo "Error: -${flag_name:0:1} $(echo $flag_name | tr '[:upper:]' '[:lower:]') is required"
        usage
    fi
done
if [[ -z "$subses_arg" && -z "$file_arg" ]]; then
    usage
fi

# ---------------------------------------------------------------------------
# Core allocation
# ---------------------------------------------------------------------------
N_RUN_JOBS=$(( N_RUNS < TOTAL_CORES ? N_RUNS : TOTAL_CORES ))
N_THREADS=$(( TOTAL_CORES / N_RUN_JOBS ))
CORES_USED=$(( N_RUN_JOBS * N_THREADS ))

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

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
LOG_DIR="${LOGBASE}/$(date +"%Y-%m-%d")_${ANALYSIS_NAME}"
mkdir -p "${LOG_DIR}"
chmod -R 777 "${LOG_DIR}" 2>/dev/null || true

cp "$0" "${LOG_DIR}/launcher_$(date +"%Y-%m-%d_%H-%M-%S").sh"
[[ -n "$file_arg" ]] && cp "$file_arg" "${LOG_DIR}/subseslist.txt"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
SOURCE="${subses_arg:-$file_arg}"
echo "============================================================"
echo "  tedana local launcher"
echo "  Analysis name : ${ANALYSIS_NAME}"
echo "  Input         : ${SOURCE}"
echo "  Sessions      : ${#PAIRS[@]}  (processed serially)"
echo "  BIDS dir      : ${BIDS_DIR}"
echo "  fmriprep dir  : ${FP_DIR}"
echo "  Output dir    : ${OUT_DIR_NAME}"
echo "  Tasks         : ${TASKS}"
echo "  Acq           : ${ACQ}"
echo "  Fit type      : ${FITTYPE}"
echo "  ── Core allocation ────────────────────────────────────"
echo "  Cores avail   : ${TOTAL_CORES}"
echo "  Runs/session  : ${N_RUNS}"
echo "  Runs parallel : ${N_RUN_JOBS}   (N_RUN_JOBS = min(runs, cores))"
echo "  Threads/run   : ${N_THREADS}   (N_THREADS  = cores / N_RUN_JOBS)"
echo "  Cores used    : ${CORES_USED} / ${TOTAL_CORES}   (${N_RUN_JOBS} × ${N_THREADS})"
echo "  ───────────────────────────────────────────────────────"
echo "  Log dir       : ${LOG_DIR}"
echo "  Dry run       : ${dry_run}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Run sessions serially
# ---------------------------------------------------------------------------
job_num=1
for pair in "${PAIRS[@]}"; do
    SUB=$(echo "$pair" | cut -d',' -f1 | tr -d ' ')
    SES=$(echo "$pair" | cut -d',' -f2 | tr -d ' ')

    LOG_OUT="${LOG_DIR}/$(date +"%H-%M")_sub-${SUB}_ses-${SES}.o"
    LOG_ERR="${LOG_DIR}/$(date +"%H-%M")_sub-${SUB}_ses-${SES}.e"

    PY_CMD="${PYTHON} ${SCRIPT} \
        -b ${BIDS_DIR} \
        -fp ${FP_DIR} \
        -o ${OUT_DIR_NAME} \
        -n ${ANALYSIS_NAME} \
        -s ${SUB},${SES} \
        --tasks ${TASKS} \
        --acq ${ACQ} \
        --fittype ${FITTYPE} \
        --n-threads ${N_THREADS} \
        --n-run-jobs ${N_RUN_JOBS}"

    [[ -n "${extra_flags}" ]] && PY_CMD="${PY_CMD} ${extra_flags}"

    echo "  [${job_num}/${#PAIRS[@]}]  sub-${SUB} ses-${SES}  → log: ${LOG_OUT}"

    {
        echo "============================================================"
        echo "  sub           : ${SUB}"
        echo "  ses           : ${SES}"
        echo "  analysis      : ${ANALYSIS_NAME}"
        echo "  tasks         : ${TASKS}"
        echo "  fittype       : ${FITTYPE}"
        echo "  Cores avail   : ${TOTAL_CORES}"
        echo "  Runs parallel : ${N_RUN_JOBS}"
        echo "  Threads/run   : ${N_THREADS}"
        echo "  Cores used    : ${CORES_USED} / ${TOTAL_CORES}"
        echo "  Start         : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"
        echo ""
        time ${PY_CMD}
        EXIT_CODE=$?
        echo ""
        echo "============================================================"
        echo "  End           : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  Exit code     : ${EXIT_CODE}"
        echo "============================================================"
    } >"${LOG_OUT}" 2>"${LOG_ERR}"

    ((job_num++))
done

echo ""
echo "All ${#PAIRS[@]} sessions done. Logs: ${LOG_DIR}"
