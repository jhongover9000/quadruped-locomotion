#!/bin/bash
# Submit the training pipeline: each stage starts when the one before it has finished.
#
#     hpc/submit_pipeline.sh                    # check -> ppo -> motions -> mimic + supermimic -> evaluate
#     SMOKE=1 hpc/submit_pipeline.sh            # the same with tiny settings (a dry run, ~20 min)
#     hpc/submit_pipeline.sh --only ppo         # one stage (also: check mpc motions mimic supermimic evaluate)
#     hpc/submit_pipeline.sh --skip ppo         # leave stages out (repeatable)
#     hpc/submit_pipeline.sh --parallel         # PPO and the motion-library branch at the same time
#     hpc/submit_pipeline.sh --id paper_v1      # name the submission (default: date_time)
#     hpc/submit_pipeline.sh --id paper_v1 --only evaluate    # re-run a stage of an earlier submission
#     hpc/submit_pipeline.sh --dry-run          # print the sbatch commands, submit nothing
#     hpc/submit_pipeline.sh --local            # no Slurm: run the stages here, one after another
#
# Stages (hpc/<number>_<stage>.slurm), in order:
#   check       GPU, tests, a few training iterations, offscreen rendering        (45 min limit)
#   ppo         PPO velocity policies: array over algorithm x task x seed          (8 h each)
#   mpc         build the convex MPC baseline (nothing to train); after check      (10 min)
#   motions     build the SuperMimic motion library, unless it exists              (3 h)
#   mimic       DeepMimic and AMP: array over MIMIC_RUNS x seed; after motions     (MIMIC_TIME each)
#   supermimic  SuperMimic training in SM_SEGMENTS chained jobs on one run         (20 h each)
#   evaluate    evaluation table, training curves, videos -> one report folder     (4 h)
#
# Order: by default one after another; mpc needs only check, and mimic runs next to
# supermimic (both need only motions). A failed check cancels everything after it; a
# failed PPO task does not stop the motion branch; evaluation runs on whatever trained.
# Settings: hpc/config.sh (override in hpc/config.local.sh or the environment).
set -euo pipefail
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ONLY="" ; SKIP=" " ; DRY=0 ; LOCAL=0 ; PARALLEL=0
PIPELINE_ID="${PIPELINE_ID:-$(date +%Y%m%d_%H%M%S)}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --skip) SKIP+="$2 "; shift 2 ;;
    --id) PIPELINE_ID="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --local) LOCAL=1; shift ;;
    --parallel) PARALLEL=1; shift ;;
    -h|--help) sed -n '2,/^set -euo/{/^#/p}' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
done
export PIPELINE_ID
# shellcheck source=lib.sh
source "$HPC_DIR/lib.sh"

STAGES=(check ppo mpc motions mimic supermimic evaluate)
for s in $ONLY $SKIP; do
  [[ " ${STAGES[*]} " == *" $s "* ]] || { echo "unknown stage '$s' (stages: ${STAGES[*]})"; exit 2; }
done
wanted() { [[ -z "$ONLY" || "$ONLY" == "$1" ]] && [[ "$SKIP" != *" $1 "* ]]; }

# shellcheck disable=SC2206
seeds=($SEEDS); algos=($PPO_ALGOS); tasks=($PPO_TASKS); mimic_runs=($MIMIC_RUNS)
N_PPO=$(( ${#seeds[@]} * ${#algos[@]} * ${#tasks[@]} ))
N_MIMIC=$(( ${#seeds[@]} * ${#mimic_runs[@]} ))

echo "Pipeline $PIPELINE_ID   (smoke test: ${SMOKE:-0})"
echo "  project  $PROJECT_DIR"
echo "  outputs  $QUADRUPED_OUTPUT_DIR"
echo "  state    $STATE_DIR"
echo "  ppo      $N_PPO runs: [$PPO_ALGOS] x [$PPO_TASKS] x seeds [$SEEDS], $PPO_NUM_ENVS envs"
echo "  mpc      1 run: mpc on flat_velocity (built, not trained)"
echo "  mimic    $N_MIMIC runs: [$MIMIC_RUNS] x seeds [$SEEDS]; deepmimic $DM_ITERATIONS iterations x $DM_NUM_ENVS envs,"
echo "           amp $AMP_ITERATIONS iterations x $AMP_NUM_ENVS envs; <= $MIMIC_MAX_HOURS h each"
echo "  supermimic  $SM_ITERATIONS iterations x $SM_NUM_ENVS envs, $SM_SEGMENTS segments of <= $SM_SEGMENT_HOURS h"

# ---- no Slurm: run the stages in order on this machine --------------------------------
if [[ $LOCAL -eq 1 ]]; then
  export HPC_DIR SMOKE="${SMOKE:-0}"
  run_local() { echo; echo "######## $* ########"; if [[ $DRY -eq 1 ]]; then echo "(dry run) bash $*"; else bash "$@"; fi; }
  if wanted check; then run_local "$HPC_DIR/10_check.slurm"; fi
  if wanted ppo; then run_local "$HPC_DIR/20_train_ppo.slurm"; fi
  if wanted mpc; then run_local "$HPC_DIR/22_mpc.slurm"; fi
  if wanted motions; then run_local "$HPC_DIR/30_supermimic_motions.slurm"; fi
  if wanted mimic; then run_local "$HPC_DIR/35_train_mimic.slurm"; fi
  if wanted supermimic; then
    for seg in $(seq 1 "$SM_SEGMENTS"); do SM_SEGMENT=$seg run_local "$HPC_DIR/40_train_supermimic.slurm"; done
  fi
  if wanted evaluate; then run_local "$HPC_DIR/50_evaluate.slurm"; fi
  echo; echo "Local pipeline $PIPELINE_ID finished. Report: $STATE_DIR/report"
  exit 0
fi

# ---- Slurm ----------------------------------------------------------------------------
if [[ $DRY -eq 0 ]]; then
  command -v sbatch >/dev/null 2>&1 || { echo "sbatch not found: run this on the cluster, or use --local / --dry-run"; exit 1; }
  mkdir -p "$LOG_DIR" "$STATE_DIR"
fi
COMMON=(--parsable --partition "$SLURM_PARTITION" --gres "$SLURM_GRES" --kill-on-invalid-dep=yes
        --chdir "$PROJECT_DIR"
        --export "ALL,HPC_DIR=$HPC_DIR,PIPELINE_ID=$PIPELINE_ID,SMOKE=${SMOKE:-0}")
[[ -n "$MAIL_USER" ]] && COMMON+=(--mail-user "$MAIL_USER")
[[ "${SMOKE:-0}" == "1" ]] && COMMON+=(--time 01:00:00)
FAKE_ID=1000

# submit NAME DEPENDENCY SCRIPT [extra sbatch options...] -> job id in $JOB
submit() {
  local name="$1" dep="$2" script="$3"; shift 3
  local cmd=(sbatch "${COMMON[@]}" --job-name "$name" --output "$LOG_DIR/%x_%j.out" --error "$LOG_DIR/%x_%j.err")
  [[ -n "$dep" ]] && cmd+=(--dependency "$dep")
  cmd+=("$@" "$script")
  if [[ $DRY -eq 1 ]]; then
    FAKE_ID=$((FAKE_ID + 1)); JOB=$FAKE_ID
    printf '  [dry run] %s\n' "${cmd[*]}"
  else
    JOB="$("${cmd[@]}")"; JOB="${JOB%%;*}"
  fi
  local array="" prev="" a
  for a in "$@"; do [[ "$prev" == "--array" ]] && array="$a"; prev="$a"; done
  printf '  %-22s job %s%s%s\n' "$name" "$JOB" "${array:+   array $array}" "${dep:+   (after: $dep)}"
  [[ $DRY -eq 1 ]] || echo "$name $JOB" >> "$STATE_DIR/jobs.txt"
}

CHECK="" ; LAST="" ; TRAINED=()
if wanted check; then
  submit quad_check "" "$HPC_DIR/10_check.slurm"; CHECK="$JOB"; LAST="afterok:$JOB"
fi
if wanted ppo; then
  submit quad_ppo "$LAST" "$HPC_DIR/20_train_ppo.slurm" --array "0-$((N_PPO - 1))" \
    --output "$LOG_DIR/%x_%A_%a.out" --error "$LOG_DIR/%x_%A_%a.err"
  TRAINED+=("$JOB")
  # The motion branch waits for PPO to end (however it ends) unless --parallel.
  if [[ $PARALLEL -eq 0 ]]; then LAST="afterany:$JOB"; fi
fi
if wanted mpc; then
  # Built in seconds; needs only the check.
  submit quad_mpc "${CHECK:+afterok:$CHECK}" "$HPC_DIR/22_mpc.slurm"
  TRAINED+=("$JOB")
fi
if wanted motions; then
  dep="$LAST"; [[ $PARALLEL -eq 1 && -n "$CHECK" ]] && dep="afterok:$CHECK"
  submit quad_motions "$dep" "$HPC_DIR/30_supermimic_motions.slurm"; LAST="afterok:$JOB"
fi
if wanted mimic; then
  # Same dependency as the first SuperMimic segment: runs next to SuperMimic.
  submit quad_mimic "$LAST" "$HPC_DIR/35_train_mimic.slurm" --array "0-$((N_MIMIC - 1))" \
    --time "$MIMIC_TIME" --output "$LOG_DIR/%x_%A_%a.out" --error "$LOG_DIR/%x_%A_%a.err"
  TRAINED+=("$JOB")
fi
if wanted supermimic; then
  for seg in $(seq 1 "$SM_SEGMENTS"); do
    submit "quad_supermimic_$seg" "$LAST" "$HPC_DIR/40_train_supermimic.slurm" \
      --export "ALL,HPC_DIR=$HPC_DIR,PIPELINE_ID=$PIPELINE_ID,SMOKE=${SMOKE:-0},SM_SEGMENT=$seg"
    LAST="afterok:$JOB"
  done
  TRAINED+=("$JOB")
fi
if wanted evaluate; then
  dep=""
  if [[ ${#TRAINED[@]} -gt 0 ]]; then dep="afterany:$(IFS=:; echo "${TRAINED[*]}")"; fi
  submit quad_eval "$dep" "$HPC_DIR/50_evaluate.slurm"
fi

echo
echo "Watch:   squeue -u $USER        logs: $LOG_DIR/quad_*"
echo "Cancel:  scancel \$(awk '{print \$2}' $STATE_DIR/jobs.txt)"
echo "Report (when quad_eval ends): $STATE_DIR/report"
