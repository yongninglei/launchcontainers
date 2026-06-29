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
PROJECT="IRAKEINU"
analysis_space="surface"  # "volume" or "surface"; determines which run_glm script to use
PYTHON_SCRIPT="/export/home/tlei/tlei/soft/launchcontainers/launchcontainers/tests/glm_strategy/glm_surface_check_model_strategy.py"
LOGBASE="/bcbl/home/public/Gari/${PROJECT}/logs/glm/${analysis_space}"

# run_glm.py arguments
BASE="/bcbl/home/public/Gari/${PROJECT}"
FP_ANA_NAME="25.1.4_IRpilot "
TASK="BfLocVideo"
SPACE="T1w"
START_SCANS="6"
CONTRAST="/export/home/tlei/tlei/soft/launchcontainers/launchcontainers/tests/glm_strategy/contrast_${PROJECT}_new.yaml"
STRATEGY_YAML="/export/home/tlei/tlei/soft/launchcontainers/launchcontainers/tests/glm_strategy/strategy.yaml"
STRATEGY="basic_MC"   # single strategy name to run, or leave empty "" to run all strategies in the YAML
RERUN_MAP=""          # leave empty "" to skip
INPUT_DIR="BIDS"      # input BIDS dir name under BASE; use BIDS_WC for WC runs
ACQ=""                # acquisition label: "ME" | "SE" | leave empty "" for no filter
BOLD_DESC=""          # desc label for bold query: "denoised" | "optcom" | leave empty ""
N_VOLS=213              # truncate to first N volumes (0 = no truncation)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage:"
    echo "  bash $0 -n <analysis_name> -s <sub>,<ses>         # single session"
    echo "  bash $0 -n <analysis_name> -f <path/subseslist>   # batch from file (RUN==True only)"
    echo ""
    echo "  Optional:"
    echo "    -i <input_dir>   BIDS dir name under BASE (default: BIDS)"
    echo "    -p <space>       Space: T1w | fsnative | fsaverage | MNI152NLin2009cAsym (default: ${SPACE})"
    echo "    -a <acq>         acquisition filter: ME | SE (default: run all combos below)"
    echo "    -d <bold_desc>   bold desc filter: denoised | optcom (default: run all combos below)"
    echo ""
    echo "  If neither -a nor -d is given, all acq/desc combos are run, one job"
    echo "  per combo per session: SE, ME+denoised, ME+optcom."
    echo "    -v <n_vols>      truncate timeseries to first N volumes (default: 0 = no truncation)"
    echo "    -r <rerun_map>   path to rerun_check.tsv/csv to exclude compensated runs"
    echo "    --dry-run        print design matrix / confounds; do not write outputs"
    echo "    --use-smoothed"
    echo "    --save-betas     also save per-regressor betas/residuals/fitted (space-tagged GIFTI/NIfTI)"
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
acq_set=0
bold_desc_set=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n) analysis_name="$2"; shift 2 ;;
        -s) subses_arg="$2";    shift 2 ;;
        -f) file_arg="$2";      shift 2 ;;
        -i) INPUT_DIR="$2";     shift 2 ;;
        -p) SPACE="$2";         shift 2 ;;
        -a) ACQ="$2";           acq_set=1;      shift 2 ;;
        -d) BOLD_DESC="$2";     bold_desc_set=1; shift 2 ;;
        -v) N_VOLS="$2";        shift 2 ;;
        -r) RERUN_MAP="$2";     shift 2 ;;
        --dry-run)      dry_run=1; extra_flags="${extra_flags} --dry-run"; shift ;;
        --use-smoothed) extra_flags="${extra_flags} --use-smoothed"; shift ;;
        --save-betas)   extra_flags="${extra_flags} --save-betas"; shift ;;
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
# acq/desc combos to run
#   - if -a and/or -d given explicitly, run just that one combo
#   - otherwise run all combos: SE (no bold_desc), ME+denoised, ME+optcom
# ---------------------------------------------------------------------------
if [[ "$acq_set" -eq 1 || "$bold_desc_set" -eq 1 ]]; then
    COMBOS=("${ACQ},${BOLD_DESC}")
else
    COMBOS=("SE," "ME,denoised" "ME,optcom")
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
echo "  Strategy YAML : ${STRATEGY_YAML}"
echo "  Strategy      : ${STRATEGY:-"(all strategies in YAML)"}"
echo "  Acq/desc combos:"
for combo in "${COMBOS[@]}"; do
    c_acq="${combo%%,*}"
    c_desc="${combo#*,}"
    echo "    - acq=${c_acq:-none}  desc=${c_desc:-none}"
done
echo "  N vols        : ${N_VOLS}"
echo "  Log dir       : ${LOG_DIR}"
echo "  Dry run       : ${dry_run}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Run sessions sequentially
# ---------------------------------------------------------------------------
total_jobs=$(( ${#PAIRS[@]} * ${#COMBOS[@]} ))
job_num=1
for pair in "${PAIRS[@]}"; do
    SUB=$(echo "$pair" | cut -d',' -f1 | tr -d ' ')
    SES=$(echo "$pair" | cut -d',' -f2 | tr -d ' ')

    for combo in "${COMBOS[@]}"; do
        CUR_ACQ="${combo%%,*}"
        CUR_DESC="${combo#*,}"

        TAG="${CUR_ACQ:-noacq}${CUR_DESC:+_${CUR_DESC}}"

        LOG_OUT="${LOG_DIR}/$(date +"%H-%M")_sub-${SUB}_ses-${SES}_${TAG}.o"
        LOG_ERR="${LOG_DIR}/$(date +"%H-%M")_sub-${SUB}_ses-${SES}_${TAG}.e"

        PY_CMD="python ${PYTHON_SCRIPT} \
            --base ${BASE} \
            -s ${SUB},${SES} \
            --fp-ana-name ${FP_ANA_NAME} \
            --task ${TASK} \
            --space ${SPACE} \
            --start-scans ${START_SCANS} \
            --contrast ${CONTRAST} \
            --analysis-name ${analysis_name} \
            --input-dir ${INPUT_DIR} \
            --strategy-yaml ${STRATEGY_YAML} \
            --n-workers 20"

        [[ -n "${STRATEGY}" ]]    && PY_CMD="${PY_CMD} --strategy ${STRATEGY}"
        [[ -n "${RERUN_MAP}" ]]   && PY_CMD="${PY_CMD} --rerun-map ${RERUN_MAP}"
        [[ -n "${CUR_ACQ}" ]]     && PY_CMD="${PY_CMD} --acq ${CUR_ACQ}"
        [[ -n "${CUR_DESC}" ]]    && PY_CMD="${PY_CMD} --bold-desc ${CUR_DESC}"
        [[ "${N_VOLS}" -gt 0 ]]   && PY_CMD="${PY_CMD} --n-vols ${N_VOLS}"
        [[ -n "${extra_flags}" ]] && PY_CMD="${PY_CMD} ${extra_flags}"

        echo "  [${job_num}/${total_jobs}]  sub-${SUB} ses-${SES} acq-${CUR_ACQ:-none} desc-${CUR_DESC:-none}  → log: ${LOG_OUT}"

        {
            echo "============================================================"
            echo "  sub         : ${SUB}"
            echo "  ses         : ${SES}"
            echo "  acq         : ${CUR_ACQ:-(none)}"
            echo "  bold_desc   : ${CUR_DESC:-(none)}"
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
done

echo ""
echo "All ${total_jobs} job(s) (${#PAIRS[@]} session(s) x ${#COMBOS[@]} combo(s)) done. Logs: ${LOG_DIR}"
