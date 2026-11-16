#!/bin/bash
#SBATCH --job-name=quad_check
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --mail-type=FAIL
#
# Stage 1: is this node able to run the pipeline? Fails fast, before the long jobs start:
# GPU visible to PyTorch and Warp, the repository's tests pass, both PPO trainers train
# for a few iterations, offscreen rendering works, and the SuperMimic reference
# repository is where config.sh says.

# Find hpc/ (Slurm runs a copy of this file from its spool directory).
for d in "${HPC_DIR:-}" "${SLURM_SUBMIT_DIR:-}/hpc" "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; do
  if [[ -f "$d/lib.sh" ]]; then HPC_DIR="$d"; break; fi
done
# shellcheck source=lib.sh
source "$HPC_DIR/lib.sh"
activate_env
print_header "environment check"

echo "==== [1] GPU ===="
python - <<'PY'
import torch, warp as wp
assert torch.cuda.is_available(), "PyTorch does not see a GPU"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0),
      f"| {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
wp.init()
assert wp.is_cuda_available(), "Warp does not see a GPU"
import mujoco
print("mujoco", mujoco.__version__, "| warp", wp.__version__)
PY

echo "==== [2] Tests ===="
python -m pytest -q -x

echo "==== [3] Algorithm x task table ===="
python scripts/train.py --list

echo "==== [4] SuperMimic backend ===="
ROBOT="$ROBOT" python - <<'PY'
import os
from quadruped_rl.algorithms.supermimic import backend
from quadruped_rl.utils.config import load_group
root = backend.ensure_importable(load_group("algorithms", "supermimic"))
print("reference repository:", root)
import supermimic.robots as R
print("robot specs:", R.ROBOT_NAMES)
robot = backend.spec_name(os.environ["ROBOT"], load_group("algorithms", "supermimic"))
assert robot in R.ROBOT_NAMES, (f"the {robot} spec (src/supermimic/robots/{robot}.py) is missing from the reference "
                                 "repository; for an imported robot run scripts/import_robot.py --sync")
PY

echo "==== [5] Three-iteration training with both PPO trainers ===="
# A fresh folder per check, so nothing has to be deleted first (a few MB each).
CHECK_DIR="$STATE_DIR/check_$(date +%Y%m%d_%H%M%S)"
for algo in ppo ppo_rslrl; do
  python scripts/train.py --algo "$algo" --num-envs 64 --max-iterations 3 --run-dir "$CHECK_DIR/$algo"
done

echo "==== [6] Offscreen rendering (MUJOCO_GL=${MUJOCO_GL:-default}) ===="
python scripts/record.py "$CHECK_DIR/ppo" --command 0.5 0 0 --duration 2 --num-envs 4 \
  --width 640 --height 360 --out "$CHECK_DIR/recording"
test -s "$CHECK_DIR/recording/video.mp4"

echo "DONE. Environment check passed (files: $CHECK_DIR)."
