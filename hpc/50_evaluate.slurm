#!/bin/bash
#SBATCH --job-name=quad_eval
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#
# Stage 5: evaluate, plot and film every run of this pipeline, and gather what you would
# want to look at into one folder to copy off the cluster:
#   <pipeline state>/report/
#     evaluation.csv            one row per run (falls, velocity error, smoothness, clip benchmark)
#     training_<run>.png        training curves of each run
#     compare_<task>.png        the runs of one task side by side
#     <run>/video.mp4, plots.png, summary.txt, rollout files for scripts/replay.py
# Runs after training whether or not every training job succeeded; runs without a
# checkpoint are skipped.

# Find hpc/ (Slurm runs a copy of this file from its spool directory).
for d in "${HPC_DIR:-}" "${SLURM_SUBMIT_DIR:-}/hpc" "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; do
  if [[ -f "$d/lib.sh" ]]; then HPC_DIR="$d"; break; fi
done
# shellcheck source=lib.sh
source "$HPC_DIR/lib.sh"
activate_env
print_header "evaluation and report"

REPORT="$STATE_DIR/report"
mkdir -p "$REPORT"
mapfile -t ALL < <(sort -u "$STATE_DIR/runs.txt" 2>/dev/null || true)
RUNS=()
for r in "${ALL[@]}"; do
  if [[ -n "$r" && -f "$r/config.yaml" ]] && { compgen -G "$r/checkpoints/model_*.pt" >/dev/null \
      || compgen -G "$r/model_*.pt" >/dev/null || compgen -G "$r/ckpt_[0-9]*.pt" >/dev/null; }; then
    RUNS+=("$r")
  else
    echo "skipping (no checkpoint): $r"
  fi
done
[[ ${#RUNS[@]} -gt 0 ]] || { echo "ERROR: no trained runs listed in $STATE_DIR/runs.txt"; exit 1; }
echo "${#RUNS[@]} runs to evaluate"

# outputs/<algo>/<task>/<id>_<robot>_s<seed>  ->  <algo>_<task>_<robot>_s<seed>
name_of() {
  local run="$1" task algo
  task="$(basename "$(dirname "$run")")"
  algo="$(basename "$(dirname "$(dirname "$run")")")"
  echo "${algo}_${task}_$(basename "$run" | sed "s/^${PIPELINE_ID}_//")"
}

echo "==== [1] Training curves ===="
for r in "${RUNS[@]}"; do
  python scripts/plot_training.py "$r" --out "$REPORT/training_$(name_of "$r").png" || echo "  (plot failed for $r)"
done
for task in $(for r in "${RUNS[@]}"; do basename "$(dirname "$r")"; done | sort -u); do
  same=()
  for r in "${RUNS[@]}"; do [[ "$(basename "$(dirname "$r")")" == "$task" ]] && same+=("$r"); done
  if [[ ${#same[@]} -gt 1 && ${#same[@]} -le 8 ]]; then
    python scripts/plot_training.py "${same[@]}" --out "$REPORT/compare_$task.png" || true
  fi
done

echo "==== [2] Evaluation (configs/eval/default.yaml) ===="
python scripts/evaluate.py "${RUNS[@]}" --seeds "$EVAL_SEEDS" --out "$REPORT/evaluation.csv"

echo "==== [3] Recordings (video, plots, summary, replay files) ===="
for r in "${RUNS[@]}"; do
  # shellcheck disable=SC2086
  python scripts/record.py "$r" --robots $RECORD_ROBOTS ${RECORD_DURATION:+--duration $RECORD_DURATION} \
    --out "$REPORT/$(name_of "$r")" || echo "  (recording failed for $r)"
done

echo
echo "DONE. Report: $REPORT"
echo "Copy it to your machine:  rsync -a $(hostname):$REPORT/ ./report_${PIPELINE_ID}/"
echo "Then watch a rollout offline:  python scripts/replay.py ./report_${PIPELINE_ID}/<run>/"
