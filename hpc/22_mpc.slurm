#!/bin/bash
#SBATCH --job-name=quad_mpc
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --mail-type=FAIL
#
# Stage mpc: the convex MPC baseline. Nothing is learned: train.py builds the controller
# from the robot model and saves it as the run's checkpoint (checkpoints/model_0.pt) in a
# few seconds. Skipped when that checkpoint already exists. The run is listed in runs.txt
# so stage evaluate evaluates and films it like the trained policies.

# Find hpc/ (Slurm runs a copy of this file from its spool directory).
for d in "${HPC_DIR:-}" "${SLURM_SUBMIT_DIR:-}/hpc" "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; do
  if [[ -f "$d/lib.sh" ]]; then HPC_DIR="$d"; break; fi
done
# shellcheck source=lib.sh
source "$HPC_DIR/lib.sh"
activate_env
print_header "MPC baseline"

RUN_DIR="$QUADRUPED_OUTPUT_DIR/mpc/flat_velocity/${PIPELINE_ID}_${ROBOT}_s0"
echo "Run: $RUN_DIR"
grep -qxF "$RUN_DIR" "$STATE_DIR/runs.txt" 2>/dev/null || echo "$RUN_DIR" >> "$STATE_DIR/runs.txt"

if [[ -f "$RUN_DIR/checkpoints/model_0.pt" ]]; then
  echo "Controller already built: $RUN_DIR/checkpoints/model_0.pt"
else
  python scripts/train.py --algo mpc --task flat_velocity --robot "$ROBOT" --run-dir "$RUN_DIR"
fi
test -s "$RUN_DIR/checkpoints/model_0.pt"
echo "DONE. MPC baseline ready."
