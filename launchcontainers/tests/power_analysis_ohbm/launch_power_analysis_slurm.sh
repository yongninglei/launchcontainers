#!/usr/bin/env bash
# =============================================================================
# launch_power_analysis_slurm.sh  —  Slurm launcher for power_analysis_ohbm.py
#
# Submits ONE job per subject.  All sessions for that subject are concatenated
# into a single 90-run design matrix; the power loop runs inside the job.
#
# Usage:
#   bash launch_power_analysis_slurm.sh -o <output_name> -s 09
#   bash launch_power_analysis_slurm.sh -o <output_name> -f subseslist.txt
# =============================================================================

# ---------------------------------------------------------------------------
# Edit these variables before running
# ---------------------------------------------------------------------------
PYTHON_SCRIPT="/scratch/tlei/lc/launchcontainers/tests/power_analysis_ohbm/power_analysis_ohbm.py"
LOGBASE="/scratch/tlei/VOTCLOC/logs/power_analysis_ohbm"

# Slurm resource settings  (2 h / subject is the initial budget for checking)
CPUS="10"
MEM="64G"
TIME="04:00:00"
QOS="regular"           # regular | test
PARTITION="general"

# Script arguments
BASE="/scratch/tlei/VOTCLOC"
FP_ANA_NAME="25.1.4_newest"
TASK="fLoc"
SPACE="fsnative"
START_SCANS="6"
CONTRAST="/scratch/tlei/lc/launchcontainers/tests/glm_strategy/contrast_VOTCLOC_3.yaml"
STRATEGY_YAML="/scratch/tlei/lc/launchcontainers/tests/glm_strategy/strategy.yaml"
STRATEGY="basic_MC"
ROI_YAML="/scratch/tlei/lc/launchcontainers/tests/power_analysis_ohbm/roi_config_VOTCLOC.yaml"
FS_ANA_NAME="freesurfer-with_t2"
LABEL_SUBDIR="manual_label_clusters_analysis_12_v3"
RERUN_MAP="/scratch/tlei/VOTCLOC/BIDS/sourcedata/qc/rerun_check.tsv"   # leave empty "" to skip
N_ITER="5"
SEED="42"
ACQ=""                # acquisition label: "ME" | "SE" | leave empty "" for no filter
BOLD_DESC=""          # desc label for bold query: "denoised" | "optcom" | leave empty ""

# ---------------------------------------------------------------------------
# Per-subject session overrides
# When a subject is listed here ONLY these sessions are used, ignoring the
# subseslist file and auto-detection from fMRIprep.
# Format:  ["<sub>"]="<ses1>,<ses2>,..."   (zero-padded, comma-separated)
# ---------------------------------------------------------------------------
declare -A SESSION_OVERRIDES
SESSION_OVERRIDES["02"]="01,02,03,04,05,06"

# Python / conda environment
CONDA_INIT="/home/tlei/soft/miniconda3/bin/activate"
CONDA_ENV="lc"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage:"
    echo "  bash $0 -o <output_name> -s <sub>                   # single subject"
    echo "  bash $0 -o <output_name> -f <path/subseslist>       # batch from file (RUN==True only)"
    echo ""
    echo "Optional:"
    echo "  -a <acq>         acquisition filter: ME | SE (default: no filter)"
    echo "  -d <bold_desc>   bold desc filter: denoised | optcom (default: no filter)"
    echo ""
    echo "Required:"
    echo "  -o <output_name>   output label and log dir suffix"
    exit 1
}

sub_arg=""
file_arg=""
output_name=""

while getopts ":o:s:f:a:d:" opt; do
    case $opt in
        o) output_name="$OPTARG" ;;
        s) sub_arg="$OPTARG"     ;;
        f) file_arg="$OPTARG"    ;;
        a) ACQ="$OPTARG"         ;;
        d) BOLD_DESC="$OPTARG"   ;;
        *) usage ;;
    esac
done

[[ -z "$output_name" ]] && { echo "Error: -o <output_name> is required"; usage; }
[[ -z "$sub_arg" && -z "$file_arg" ]] && usage

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
LOG_DIR="${LOGBASE}/$(date +"%Y-%m-%d")_${output_name}"
mkdir -p "${LOG_DIR}"
chmod -R 777 "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Build subject list
# ---------------------------------------------------------------------------
if [[ -n "$sub_arg" ]]; then
    SUB=$(printf "%02d" "$((10#${sub_arg// /}))")
    SUBS=("$SUB")
else
    [[ ! -f "$file_arg" ]] && { echo "Error: subseslist not found: $file_arg"; exit 1; }
    mapfile -t SUBS < <(awk -F',' '
        NR==1 {
            for(i=1;i<=NF;i++) {
                if(tolower($i)=="run") runcol=i
                if(tolower($i)=="sub") subcol=i
            }
            next
        }
        {
            if(runcol && $runcol!="True") next
            subid=$subcol
            gsub(/ /, "", subid)
            printf "%02d\n", subid+0
        }
    ' "$file_arg" | sort -u)
fi

# Save a copy of this launcher for reproducibility
cp "$0" "${LOG_DIR}/launcher_$(date +"%Y-%m-%d_%H-%M-%S").sh"
[[ -n "$file_arg" ]] && cp "$file_arg" "${LOG_DIR}/subseslist.txt"

SOURCE="${sub_arg:-$file_arg}"
echo "============================================================"
echo "  Power Analysis OHBM — Slurm launcher"
echo "  Output name : ${output_name}"
echo "  Input       : ${SOURCE}"
echo "  Subjects    : ${#SUBS[@]}  (${SUBS[*]})"
echo "  Task        : ${TASK}  Space: ${SPACE}"
echo "  Strategy    : ${STRATEGY}  n_iter: ${N_ITER}  seed: ${SEED}"
echo "  Acq filter  : ${ACQ:-"(none)"}  Bold desc: ${BOLD_DESC:-"(none)"}"
echo "  CPUs/job    : ${CPUS}  Mem: ${MEM}  Time: ${TIME}"
echo "  QOS         : ${QOS}  Partition: ${PARTITION}"
echo "  Log dir     : ${LOG_DIR}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Submit one job per subject
# ---------------------------------------------------------------------------
job_num=1
for SUB in "${SUBS[@]}"; do

    PY_CMD="python ${PYTHON_SCRIPT} \
        --base ${BASE} \
        --sub ${SUB} \
        --fp-ana-name ${FP_ANA_NAME} \
        --fs-ana-name ${FS_ANA_NAME} \
        --label-subdir ${LABEL_SUBDIR} \
        --task ${TASK} \
        --space ${SPACE} \
        --start-scans ${START_SCANS} \
        --contrast ${CONTRAST} \
        --output-name ${output_name} \
        --roi-yaml ${ROI_YAML} \
        --strategy-yaml ${STRATEGY_YAML} \
        --strategy ${STRATEGY} \
        --n-iter ${N_ITER} \
        --seed ${SEED}"

    # Session source priority:
    #   1. SESSION_OVERRIDES[sub]  →  pass --sessions explicitly
    #   2. -f subseslist file      →  forward file, Python filters by sub
    #   3. (neither)               →  Python auto-detects from fMRIprep
    if [[ -n "${SESSION_OVERRIDES[$SUB]+x}" ]]; then
        PY_CMD="${PY_CMD} --sessions ${SESSION_OVERRIDES[$SUB]}"
    elif [[ -n "${file_arg}" ]]; then
        PY_CMD="${PY_CMD} -f ${file_arg}"
    fi

    [[ -n "${RERUN_MAP}" ]]  && PY_CMD="${PY_CMD} --rerun-map ${RERUN_MAP}"
    [[ -n "${ACQ}" ]]        && PY_CMD="${PY_CMD} --acq ${ACQ}"
    [[ -n "${BOLD_DESC}" ]]  && PY_CMD="${PY_CMD} --bold-desc ${BOLD_DESC}"

    now=$(date +"%H-%M")
    OUT_FILE="${LOG_DIR}/${now}_%j_sub-${SUB}.o"
    ERR_FILE="${LOG_DIR}/${now}_%j_sub-${SUB}.e"

    JOBID=$(sbatch \
        --job-name="pwr_${SUB}" \
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
echo '  output_name : ${output_name}'
echo '  n_iter      : ${N_ITER}   seed: ${SEED}'
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

    echo "  [${job_num}/${#SUBS[@]}]  sub-${SUB}  → job ${JOBID}  (logs: ${LOG_DIR})"
    ((job_num++))
done

echo ""
echo "All ${#SUBS[@]} job(s) submitted.  Logs: ${LOG_DIR}"
