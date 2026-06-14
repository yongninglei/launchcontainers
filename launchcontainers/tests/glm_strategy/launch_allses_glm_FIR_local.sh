#!/usr/bin/env bash
# =============================================================================
# launch_allses_glm_FIR_local.sh  —  Local launcher for run_allses_glm_FIR.py
#                                     + fir_curve_metrics.py
#
# Fits ONE FIR GLM per subject (all sessions/runs concatenated) for VOTCLOC
# task-fLoc, then computes per-vertex AllStim FIR curve metrics.
#
# Usage:
#   bash launch_allses_glm_FIR_local.sh -o <output_name> -s 03
#   bash launch_allses_glm_FIR_local.sh -o <output_name> -s 03 --sessions 01,02
#   bash launch_allses_glm_FIR_local.sh -o <output_name> -f subseslist.txt
#   bash launch_allses_glm_FIR_local.sh -o <output_name> -s 03 --dry-run
# =============================================================================

# ---------------------------------------------------------------------------
# Edit these variables before running
# ---------------------------------------------------------------------------
PROJECT="VOTCLOC"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLM_SCRIPT="${SCRIPT_DIR}/run_allses_glm_FIR.py"
METRICS_SCRIPT="${SCRIPT_DIR}/fir_curve_metrics.py"
LOGBASE="/bcbl/home/public/Gari/${PROJECT}/main_exp/logs/glm_FIR/surface"

# run_allses_glm_FIR.py arguments
BASE="/bcbl/home/public/Gari/${PROJECT}/main_exp"
FP_ANA_NAME="25.1.4_t2w_fmapsbref_newest"
TASK="fLoc"
SPACE="fsnative"
START_SCANS="6"
INPUT_DIR="BIDS"
CONDITIONS="RW,CS,FF,SC,bodylimb,face"
N_DELAYS="10"
N_GLM_JOBS="$(( $(nproc) - 2 ))"
DEFAULT_SESSIONS="01,02,03,04,05,06,07,08,09"
RERUN_MAP="/bcbl/home/public/Gari/${PROJECT}/main_exp/BIDS/sourcedata/qc/rerun_check.tsv"

# Python ("lc" conda/micromamba env) — use the active env's python if available,
# otherwise search common env locations under $HOME.
PYTHON_BIN="$(command -v python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
    for base in "${HOME}/micromamba/envs" "${HOME}/miniconda3/envs" "${HOME}/anaconda3/envs"; do
        if [[ -x "${base}/lc/bin/python" ]]; then
            PYTHON_BIN="${base}/lc/bin/python"
            break
        fi
    done
fi
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "Error: could not find python. Activate the 'lc' env first (conda/micromamba activate lc)."
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage:"
    echo "  bash $0 -o <output_name> -s <sub> [--sessions 01,02,...]"
    echo "  bash $0 -o <output_name> -f <path/subseslist>"
    echo ""
    echo "  Optional:"
    echo "    --sessions <list>   comma-separated sessions (default: ${DEFAULT_SESSIONS})"
    echo "    --dry-run           print design matrices, write nothing"
    echo "    --no-metrics        skip fir_curve_metrics.py step"
    echo ""
    echo "Required:"
    echo "  -o <output_name>   GLM output label (--output-name) and log dir suffix"
    exit 1
}

sub_arg=""
file_arg=""
output_name=""
sessions_arg="${DEFAULT_SESSIONS}"
extra_flags=""
run_metrics=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o) output_name="$2"; shift 2 ;;
        -s) sub_arg="$2";     shift 2 ;;
        -f) file_arg="$2";    shift 2 ;;
        --sessions)   sessions_arg="$2"; shift 2 ;;
        --dry-run)    extra_flags="${extra_flags} --dry-run"; run_metrics=0; shift ;;
        --no-metrics) run_metrics=0; shift ;;
        *) echo "Unknown option: $1"; usage ;;
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
cp "$0" "${LOG_DIR}/launcher_$(date +"%Y-%m-%d_%H-%M-%S").sh"

# ---------------------------------------------------------------------------
# Build subject list
# ---------------------------------------------------------------------------
if [[ -n "$sub_arg" ]]; then
    SUB=$(printf "%02d" "$((10#${sub_arg// /}))")
    SUBS=("$SUB")
else
    if [[ ! -f "$file_arg" ]]; then
        echo "Error: subseslist not found: $file_arg"
        exit 1
    fi
    cp "$file_arg" "${LOG_DIR}/subseslist.txt"
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

echo "============================================================"
echo "  All-sessions FIR GLM local launcher"
echo "  Output name : ${output_name}"
echo "  Subjects    : ${#SUBS[@]}  (${SUBS[*]})"
echo "  Sessions    : ${sessions_arg}"
echo "  Task        : ${TASK}"
echo "  Conditions  : ${CONDITIONS}"
echo "  N delays    : ${N_DELAYS}"
echo "  Space       : ${SPACE}"
echo "  Log dir     : ${LOG_DIR}"
echo "============================================================"
echo ""

job_num=1
for SUB in "${SUBS[@]}"; do
    LOG_OUT="${LOG_DIR}/$(date +"%H-%M")_sub-${SUB}.o"
    LOG_ERR="${LOG_DIR}/$(date +"%H-%M")_sub-${SUB}.e"

    PY_CMD="${PYTHON_BIN} ${GLM_SCRIPT} \
        --base ${BASE} --sub ${SUB} --sessions ${sessions_arg} \
        --fp-ana-name ${FP_ANA_NAME} --task ${TASK} --space ${SPACE} \
        --start-scans ${START_SCANS} --input-dir ${INPUT_DIR} \
        --output-name ${output_name} --conditions ${CONDITIONS} \
        --n-delays ${N_DELAYS} --n-glm-jobs ${N_GLM_JOBS}"

    [[ -n "${RERUN_MAP}" && -f "${RERUN_MAP}" ]] && PY_CMD="${PY_CMD} --rerun-map ${RERUN_MAP}"
    [[ -n "${extra_flags}" ]] && PY_CMD="${PY_CMD} ${extra_flags}"

    echo "  [${job_num}/${#SUBS[@]}]  sub-${SUB}  → log: ${LOG_OUT}"

    {
        echo "============================================================"
        echo "  sub      : ${SUB}"
        echo "  output   : ${output_name}"
        echo "  Start    : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"
        echo ""
        time ${PY_CMD}
        EXIT_CODE=$?

        if [[ ${EXIT_CODE} -eq 0 && ${run_metrics} -eq 1 ]]; then
            BETAS_DIR="${BASE}/${INPUT_DIR}/derivatives/l1_surface_fir/analysis-${output_name}/sub-${SUB}/allses"
            echo ""
            echo "------------------------------------------------------------"
            echo "  FIR curve metrics → ${BETAS_DIR}"
            echo "------------------------------------------------------------"
            time ${PYTHON_BIN} ${METRICS_SCRIPT} \
                --betas-dir "${BETAS_DIR}" --sub ${SUB} --task ${TASK} \
                --space ${SPACE} --n-delays ${N_DELAYS}
            EXIT_CODE=$?
        fi

        echo ""
        echo "============================================================"
        echo "  End       : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  Exit code : ${EXIT_CODE}"
        echo "============================================================"
    } >"${LOG_OUT}" 2>"${LOG_ERR}"

    ((job_num++))
done

echo ""
echo "All ${#SUBS[@]} subject(s) done. Logs: ${LOG_DIR}"
