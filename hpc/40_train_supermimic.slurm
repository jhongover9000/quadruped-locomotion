#!/bin/bash
#SBATCH --job-name=quad_supermimic
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#
# Stage 4: SuperMimic motion tracking (learned world model + BPTT policy improvement).
# One *segment* of the run: training stops cleanly after SM_SEGMENT_HOURS (inside the
# 20 h limit) and the next segment, submitted with a dependency on this one, continues
# the same run directory from its last checkpoint. A segment that finds the run already
# finished exits at once. submit_pipeline.sh chains SM_SEGMENTS of them.

# Find hpc/ (Slurm runs a copy of this file from its spool directory).
for d in "${HPC_DIR:-}" "${SLURM_SUBMIT_DIR:-}/hpc" "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; do
  if [[ -f "$d/lib.sh" ]]; then HPC_DIR="$d"; break; fi
done
# shellcheck source=lib.sh
source "$HPC_DIR/lib.sh"
activate_env
print_header "SuperMimic training (segment ${SM_SEGMENT:-1}/${SM_SEGMENTS})"

RUN_DIR="$QUADRUPED_OUTPUT_DIR/supermimic/motion_tracking/${PIPELINE_ID}_${ROBOT}_s${SM_SEED}"
echo "Run: $RUN_DIR"
echo "$SM_ITERATIONS iterations x $SM_NUM_ENVS envs, at most $SM_SEGMENT_HOURS h in this segment"
grep -qxF "$RUN_DIR" "$STATE_DIR/runs.txt" 2>/dev/null || echo "$RUN_DIR" >> "$STATE_DIR/runs.txt"

TRAIN_EXTRA="--max-hours $SM_SEGMENT_HOURS"
train_run "$RUN_DIR" --algo supermimic --robot "$ROBOT" --seed "$SM_SEED" \
  --num-envs "$SM_NUM_ENVS" --max-iterations "$SM_ITERATIONS"

finished="$(status_field "$RUN_DIR" finished)"
iteration="$(status_field "$RUN_DIR" iteration)"
echo "Status: iteration $iteration/$SM_ITERATIONS, finished: $finished"
if [[ "$finished" != "True" && "${SM_SEGMENT:-1}" == "$SM_SEGMENTS" ]]; then
  echo "WARNING: the last segment ended before training finished. Evaluation uses the latest"
  echo "         checkpoint. To train the rest:"
  echo "         SM_SEGMENTS=2 hpc/submit_pipeline.sh --id $PIPELINE_ID --only supermimic"
fi
echo "DONE. SuperMimic segment finished."
