#!/bin/bash
#SBATCH --job-name=quad_motions
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --mail-type=FAIL
#
# Stage 3: the reference-motion library SuperMimic trains on (dog motion capture
# retargeted to the robot, 50 Hz clips, with train / hard / holdout groups).
# Skipped when the library is already there, e.g. copied from a workstation:
#   rsync -a <workstation>:~/code/supermimic-quadruped/data/npz/go1/ $SUPERMIMIC_ROOT/data/npz/go1/
# Building it needs the raw captures in $SUPERMIMIC_ROOT/data/raw (~470 MB). Compute nodes
# usually have no internet, so hpc/00_setup.sh downloads them on the login node.

# Find hpc/ (Slurm runs a copy of this file from its spool directory).
for d in "${HPC_DIR:-}" "${SLURM_SUBMIT_DIR:-}/hpc" "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; do
  if [[ -f "$d/lib.sh" ]]; then HPC_DIR="$d"; break; fi
done
# shellcheck source=lib.sh
source "$HPC_DIR/lib.sh"
activate_env
print_header "SuperMimic motion library"

SPEC="$ROBOT"                       # the robot's RobotSpec name in the reference repository
LIB="$SUPERMIMIC_ROOT/data/npz/$SPEC"
cd "$SUPERMIMIC_ROOT"
export PYTHONPATH="$SUPERMIMIC_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

has_groups() { [[ -f "$LIB/manifest.csv" ]] && head -1 "$LIB/manifest.csv" | grep -q group; }

if has_groups; then
  echo "Library already built: $(find "$LIB" -name '*.npz' | wc -l) clips in $LIB"
else
  if [[ ! -d data/raw ]] || [[ -z "$(ls -A data/raw 2>/dev/null)" ]]; then
    echo "==== [0] Raw captures missing: downloading (needs internet) ===="
    python -u scripts/data/fetch_sources.py || {
      echo "ERROR: could not download the raw captures. Run hpc/00_setup.sh on the login node,"
      echo "       or copy data/npz/$SPEC from a machine that has the library."; exit 1; }
  fi
  echo "==== [1] Retarget every capture to $SPEC ===="
  python -u scripts/data/build_dataset.py --robot "$SPEC"
  echo "==== [2] Check the library (joint limits, ground contact, velocities) ===="
  python -u scripts/data/check_library.py --robot "$SPEC"
  echo "==== [3] Screen which clips the robot can physically follow (kp $SM_STIFFNESS) ===="
  python -u scripts/data/screen_clips.py --robot "$SPEC" --stiffness "$SM_STIFFNESS"
  echo "==== [4] Assign train / hard / holdout groups ===="
  python -u scripts/data/assign_groups.py --robot "$SPEC"
fi

python - "$LIB/manifest.csv" <<'PY'
import csv, sys
from collections import Counter
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get("keep", "1") in ("1", "True", "true")]
print("clips kept per group:", dict(Counter(r.get("group", "?") for r in rows)))
assert rows, "the manifest keeps no clips"
PY
echo "DONE. Motion library ready: $LIB"
