#!/bin/bash
#SBATCH --job-name=quad_ppo
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#
# Stage 2: PPO velocity policies. One array task per (algorithm, task, seed):
#   algorithms PPO_ALGOS (ours and the rsl_rl reference) x tasks PPO_TASKS x seeds SEEDS.
# submit_pipeline.sh sets --array to cover them all; run by hand (bash, no array index)
# it trains them one after another. A run that already exists is continued, so
# resubmitting after a time-out or a node failure picks up where it stopped.

# Find hpc/ (Slurm runs a copy of this file from its spool directory).
for d in "${HPC_DIR:-}" "${SLURM_SUBMIT_DIR:-}/hpc" "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; do
  if [[ -f "$d/lib.sh" ]]; then HPC_DIR="$d"; break; fi
done
# shellcheck source=lib.sh
source "$HPC_DIR/lib.sh"
activate_env
print_header "PPO training"

combos=()
for algo in $PPO_ALGOS; do for task in $PPO_TASKS; do for seed in $SEEDS; do
  combos+=("$algo $task $seed")
done; done; done
echo "${#combos[@]} runs in total: algorithms [$PPO_ALGOS] x tasks [$PPO_TASKS] x seeds [$SEEDS]"

train_one() {
  local algo task seed iterations
  read -r algo task seed <<< "$1"
  iterations="$PPO_ITERATIONS_FLAT"
  [[ "$task" == rough* ]] && iterations="$PPO_ITERATIONS_ROUGH"
  local run_dir="$QUADRUPED_OUTPUT_DIR/$algo/$task/${PIPELINE_ID}_${ROBOT}_s${seed}"
  echo "==== $algo on $task, seed $seed: $iterations iterations x $PPO_NUM_ENVS envs ===="
  echo "$run_dir" >> "$STATE_DIR/runs.txt"
  train_run "$run_dir" --algo "$algo" --task "$task" --robot "$ROBOT" --seed "$seed" \
    --num-envs "$PPO_NUM_ENVS" --max-iterations "$iterations"
}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  train_one "${combos[$SLURM_ARRAY_TASK_ID]}"
else
  for c in "${combos[@]}"; do train_one "$c"; done
fi
echo "DONE. PPO training finished."
