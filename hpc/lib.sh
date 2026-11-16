# Shared by every stage: settings, environment activation, and the training wrapper.
# shellcheck shell=bash
set -euo pipefail

# The stage scripts set HPC_DIR before sourcing this file; by hand it is this file's folder.
HPC_DIR="${HPC_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export HPC_DIR

# shellcheck source=config.sh
source "$HPC_DIR/config.sh"
# shellcheck disable=SC1091
[[ -f "$HPC_DIR/config.local.sh" ]] && source "$HPC_DIR/config.local.sh"
# On a workstation (no /scratch), fall back to the checkout this file is in.
if [[ ! -d "$PROJECT_DIR" ]]; then
  PROJECT_DIR="$(cd "$HPC_DIR/.." && pwd)"
  [[ -d "$SUPERMIMIC_ROOT" ]] || SUPERMIMIC_ROOT="$(cd "$PROJECT_DIR/.." && pwd)/supermimic-quadruped"
  export QUADRUPED_OUTPUT_DIR="${QUADRUPED_OUTPUT_DIR_OVERRIDE:-$PROJECT_DIR/outputs}"
  PIPELINE_DIR="$QUADRUPED_OUTPUT_DIR/pipelines"
fi
export SUPERMIMIC_ROOT

# Where this submission's stages leave notes for each other (run directories, status).
PIPELINE_ID="${PIPELINE_ID:-manual}"
STATE_DIR="$PIPELINE_DIR/$PIPELINE_ID"

activate_env() {
  # conda's scripts (and env activate.d hooks) read unset variables and may contain
  # commands that fail harmlessly; under -e / pipefail such a failure would end the job
  # here with no message. The GPU check in stage `check` reports a broken driver instead.
  set +eu +o pipefail
  if command -v module >/dev/null 2>&1; then
    module purge || true
    module load "$CONDA_MODULE" || true
  fi
  if [[ -f "$CONDA_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_ACTIVATE"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
  echo "===================== Activating Conda Env: $CONDA_ENV ======================="
  conda activate "$CONDA_ENV" || { echo "ERROR: could not activate conda env '$CONDA_ENV'"; exit 1; }
  set -euo pipefail
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
  # Offscreen rendering: a workstation with a display keeps its window system; a compute
  # node has none and uses the GPU through EGL.
  if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then export MUJOCO_GL="${MUJOCO_GL:-egl}"; fi
  # Compiled-kernel caches on scratch: home quotas are small and the caches are reused.
  if [[ -d "$SCRATCH_ROOT" ]]; then
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SCRATCH_ROOT/.cache/torchinductor}"
    export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$SCRATCH_ROOT/.cache/warp}"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$WARP_CACHE_PATH"
  fi
  mkdir -p "$STATE_DIR" "$QUADRUPED_OUTPUT_DIR"
  cd "$PROJECT_DIR"
}

print_header() {
  echo "============================================================"
  echo "Stage:    $1"
  echo "Job:      ${SLURM_JOB_NAME:-local}  ID: ${SLURM_JOB_ID:-none}  array task: ${SLURM_ARRAY_TASK_ID:--}"
  echo "Host:     $(hostname)   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Project:  $PROJECT_DIR"
  echo "Outputs:  $QUADRUPED_OUTPUT_DIR"
  echo "Pipeline: $PIPELINE_ID   (state: $STATE_DIR)"
  echo "CPUs:     ${SLURM_CPUS_PER_TASK:-$(nproc)}   smoke test: ${SMOKE:-0}"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/GPU:      /' || echo "GPU:      none visible"
  echo "============================================================"
}

# status_field RUN_DIR FIELD -> prints the field of the run's status.json ("" if missing)
status_field() {
  python - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "status.json"
print(json.loads(p.read_text()).get(sys.argv[2], "") if p.exists() else "")
PY
}

# train_run RUN_DIR [train.py arguments for a fresh start...]
# Starts the run, or continues it if RUN_DIR already holds one. A run that stalls
# (exit code 75 from --stall-timeout) is continued from its last checkpoint, up to
# STALL_RETRIES times. Returns train.py's exit code otherwise.
train_run() {
  local run_dir="$1"; shift
  local attempt=0 rc=0
  local stall=()
  [[ "${STALL_TIMEOUT_S:-0}" != "0" ]] && stall=(--stall-timeout "$STALL_TIMEOUT_S")
  while true; do
    rc=0
    if [[ -f "$run_dir/config.yaml" ]]; then
      echo "---- continuing $run_dir (attempt $((attempt + 1)))"
      python scripts/train.py --continue-run "$run_dir" "${stall[@]}" ${TRAIN_EXTRA:-} || rc=$?
    else
      echo "---- starting $run_dir"
      python scripts/train.py --run-dir "$run_dir" "${stall[@]}" ${TRAIN_EXTRA:-} "$@" || rc=$?
    fi
    if [[ $rc -eq 75 && $attempt -lt ${STALL_RETRIES:-3} ]]; then
      attempt=$((attempt + 1))
      echo "!!!! run stalled (exit 75); restart $attempt/${STALL_RETRIES:-3} from the last checkpoint"
      continue
    fi
    return $rc
  done
}
