#!/bin/bash
# One-time setup, run on the LOGIN node (it needs internet; compute nodes usually have none):
#
#     bash hpc/00_setup.sh
#
# 1. creates the conda environment and installs this repository with its GPU stack
# 2. checks that the SuperMimic reference repository is in place (it is not on GitHub:
#    copy it from your workstation, including src/supermimic/robots/go1.py)
# 3. downloads the raw motion captures (~470 MB) unless a built library is already there
#
# Safe to run again: every step is skipped when its result exists.
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$HPC_DIR/lib.sh"
echo "Root:       $SCRATCH_ROOT"
echo "Project:    $PROJECT_DIR"
echo "SuperMimic: $SUPERMIMIC_ROOT"
echo "Outputs:    $QUADRUPED_OUTPUT_DIR"
echo "Logs:       $LOG_DIR"
mkdir -p "$LOG_DIR" "$QUADRUPED_OUTPUT_DIR"

echo "==== [1] Conda environment '$CONDA_ENV' ===="
set +u
if command -v module >/dev/null 2>&1; then module purge || true; module load "$CONDA_MODULE" || true; fi
# shellcheck disable=SC1090
[[ -f "$CONDA_ACTIVATE" ]] && source "$CONDA_ACTIVATE"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda create -n "$CONDA_ENV" python=3.12 -y
fi
conda activate "$CONDA_ENV"
set -u
cd "$PROJECT_DIR"
pip install -e ".[mjlab,dev,video,supermimic]"
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA build:", torch.version.cuda,
      "| GPU visible here:", torch.cuda.is_available(), "(login nodes often have none)")
PY

echo "==== [2] SuperMimic reference repository ===="
if [[ ! -d "$SUPERMIMIC_ROOT/src/supermimic" ]]; then
  echo "MISSING: $SUPERMIMIC_ROOT"
  echo "Copy it from your workstation (it is not a git repository), for example:"
  echo "  rsync -a --exclude logs --exclude data/raw --exclude '*.pt' \\"
  echo "      ~/code/supermimic-quadruped/ <cluster>:$SUPERMIMIC_ROOT/"
  echo "Including data/npz/go1/ in the copy saves rebuilding the motion library here."
  exit 1
fi
[[ -f "$SUPERMIMIC_ROOT/src/supermimic/robots/go1.py" ]] || {
  echo "MISSING: the Go1 robot spec src/supermimic/robots/go1.py (and its line in robots/__init__.py)"; exit 1; }
echo "found, with the Go1 spec"

echo "==== [3] Motion data ===="
LIB="$SUPERMIMIC_ROOT/data/npz/$ROBOT"
if [[ -f "$LIB/manifest.csv" ]] && head -1 "$LIB/manifest.csv" | grep -q group; then
  echo "built library present: $(find "$LIB" -name '*.npz' | wc -l) clips (stage 3 will skip the build)"
elif [[ -d "$SUPERMIMIC_ROOT/data/raw" && -n "$(ls -A "$SUPERMIMIC_ROOT/data/raw" 2>/dev/null)" ]]; then
  echo "raw captures present; stage 3 builds the library on a compute node"
else
  echo "downloading the raw captures (~470 MB; CC BY-NC / research-only data)"
  (cd "$SUPERMIMIC_ROOT" && PYTHONPATH="$SUPERMIMIC_ROOT/src" python -u scripts/data/fetch_sources.py)
fi

echo
echo "Setup done. Next:"
echo "  SMOKE=1 hpc/submit_pipeline.sh        # ~20 min dry run of every stage with tiny settings"
echo "  hpc/submit_pipeline.sh                # the real thing"
