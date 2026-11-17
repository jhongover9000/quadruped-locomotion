#!/bin/bash
#SBATCH --job-name=quad_mimic
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#
# Stage mimic: DeepMimic (PPO on the motion-tracking reward) and AMP (PPO with a
# discriminator style reward). Both read the motion library of stage motions.
# One array task per (algo:task pair in MIMIC_RUNS, seed in SEEDS); submit_pipeline.sh
# sets --array to cover them all and --time to MIMIC_TIME. Run by hand (bash, no array
# index) it trains them one after another. Each run trains for at most MIMIC_MAX_HOURS in
# this job; a run that already exists is continued, so resubmitting
# (hpc/submit_pipeline.sh --id <id> --only mimic) trains what is left.

# Find hpc/ (Slurm runs a copy of this file from its spool directory).
for d in "${HPC_DIR:-}" "${SLURM_SUBMIT_DIR:-}/hpc" "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; do
  if [[ -f "$d/lib.sh" ]]; then HPC_DIR="$d"; break; fi
done
# shellcheck source=lib.sh
source "$HPC_DIR/lib.sh"
activate_env
print_header "DeepMimic and AMP training"

combos=()
for pair in $MIMIC_RUNS; do for seed in $SEEDS; do
  combos+=("${pair%%:*} ${pair#*:} $seed")
done; done
echo "${#combos[@]} runs in total: [$MIMIC_RUNS] x seeds [$SEEDS], at most $MIMIC_MAX_HOURS h each"

train_one() {
  local algo task seed envs iterations
  read -r algo task seed <<< "$1"
  case "$algo" in
    deepmimic) envs="$DM_NUM_ENVS"; iterations="$DM_ITERATIONS" ;;
    amp) envs="$AMP_NUM_ENVS"; iterations="$AMP_ITERATIONS" ;;
    *) echo "ERROR: MIMIC_RUNS entry '$algo:$task': algo must be deepmimic or amp"; return 2 ;;
  esac
  local run_dir="$QUADRUPED_OUTPUT_DIR/$algo/$task/${PIPELINE_ID}_${ROBOT}_s${seed}"
  echo "==== $algo on $task, seed $seed: $iterations iterations x $envs envs ===="
  grep -qxF "$run_dir" "$STATE_DIR/runs.txt" 2>/dev/null || echo "$run_dir" >> "$STATE_DIR/runs.txt"
  TRAIN_EXTRA="--max-hours $MIMIC_MAX_HOURS" \
    train_run "$run_dir" --algo "$algo" --task "$task" --robot "$ROBOT" --seed "$seed" \
    --num-envs "$envs" --max-iterations "$iterations"

  local finished iteration
  finished="$(status_field "$run_dir" finished)"
  iteration="$(status_field "$run_dir" iteration)"
  echo "Status: iteration $iteration/$iterations, finished: $finished"
  if [[ "$finished" != "True" ]]; then
    echo "WARNING: $algo on $task, seed $seed stopped before its last iteration (time limit)."
    echo "         Evaluation uses the latest checkpoint. To train the rest:"
    echo "         hpc/submit_pipeline.sh --only mimic --id $PIPELINE_ID"
  fi
}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  train_one "${combos[$SLURM_ARRAY_TASK_ID]}"
else
  for c in "${combos[@]}"; do train_one "$c"; done
fi
echo "DONE. DeepMimic and AMP training finished."
