#!/usr/bin/env bash
# =============================================================================
# launch_power_analysis_cajal.sh — run power_analysis_local_parallel.py on one
# big BCBL node (cajal04, 96 cores) in the background, overnight.
#
# uv cache + virtualenv live on /scratch/tlei (local to the node), not /home.
#
# Usage (on cajal04):
#   bash launch_power_analysis_cajal.sh            # all subjects, 90 workers
#   bash launch_power_analysis_cajal.sh --subs 01  # extra args go to the script
# Follow:
#   tail -f <LOG_DIR>/power_<timestamp>.log
# =============================================================================
set -euo pipefail

N_WORKERS=90
N_ITER=3
SEED=42
OUTPUT_NAME="final_v3_poweranalysis"

# resolve from this file's location, so it works whatever $HOME is on the node
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(cd "$HERE/../../.." && pwd)          # .../launchcontainers (has pyproject.toml)
SCRIPT="$HERE/power_analysis_local_parallel.py"
[[ -f "$PROJECT/pyproject.toml" ]] || { echo "pyproject.toml not found in $PROJECT"; exit 1; }
LOG_DIR="/bcbl/home/public/Gari/VOTCLOC/main_exp/logs/l1_surfaces_log/analysis-${OUTPUT_NAME}"

# count cores BEFORE setting OMP_NUM_THREADS: GNU nproc honours that variable
# and would report 1. env -u keeps cgroup/affinity limits but ignores OMP.
N_CORES=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)

export UV_CACHE_DIR="/scratch/tlei/uv_cache"
export UV_PROJECT_ENVIRONMENT="/scratch/tlei/venvs/launchcontainers"
# one BLAS thread per worker process; the pool provides the parallelism
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "$UV_CACHE_DIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")" "$LOG_DIR"

# uv may not be on PATH in a non-login shell on the node
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "uv not found (looked on PATH and \$HOME/.local/bin)"; exit 1; }

# never ask for more workers than the node has cores (leave 2 for the system)
MAX_W=$(( N_CORES - 2 ))
(( MAX_W < 1 )) && MAX_W=1
(( N_WORKERS > MAX_W )) && { echo "N_WORKERS=$N_WORKERS > $MAX_W usable cores on $(hostname); using $MAX_W"; N_WORKERS=$MAX_W; }

echo "host      : $(hostname)   cores: $N_CORES   workers: $N_WORKERS"
echo "project   : $PROJECT"
echo "uv env    : $UV_PROJECT_ENVIRONMENT"
echo "uv cache  : $UV_CACHE_DIR"
echo "syncing environment (first time builds it) ..."
uv sync --project "$PROJECT" --frozen

TS=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/power_${TS}.log"
cp "${BASH_SOURCE[0]}" "$LOG_DIR/launcher_${TS}.sh"

nohup uv run --project "$PROJECT" --frozen python "$SCRIPT" \
    --n-workers "$N_WORKERS" --n-iter "$N_ITER" --seed "$SEED" \
    --output-name "$OUTPUT_NAME" "$@" \
    > "$LOG" 2>&1 &

echo "started PID $!  ->  $LOG"
echo "follow: tail -f $LOG"
