#!/bin/bash
# Run the local smoke pipeline one stage at a time; retry a stage up to 3 times, since this
# workstation has random one-off process failures. Each attempt's log is kept.
#
#   scripts/dev/run_stages.sh [log_dir]                  # default: outputs/debug/run_stages
#   STAGES="ppo evaluate" scripts/dev/run_stages.sh      # only some stages
source ~/miniconda3/etc/profile.d/conda.sh; conda activate mujoco
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
S="${1:-$REPO/outputs/debug/run_stages}"; mkdir -p "$S"; S="$(cd "$S" && pwd)"
export SMOKE=1 PIPELINE_ID=smoke_local
for stage in ${STAGES:-check ppo motions supermimic evaluate}; do
  for attempt in 1 2 3; do
    log="$S/stage_${stage}_$attempt.log"
    start=$(date +%s)
    timeout -s KILL 3000 hpc/submit_pipeline.sh --local --only "$stage" > "$log" 2>&1; rc=$?
    echo "$(date +%H:%M:%S) stage $stage attempt $attempt rc=$rc ($(( $(date +%s) - start )) s)"
    [[ $rc -eq 0 ]] && break
    grep -E "Error|ERROR|MALFORMED|Killed|assert" "$log" | tail -3 | cut -c1-200
  done
  [[ $rc -eq 0 ]] || { echo "stage $stage failed 3 times; stopping"; exit 1; }
done
echo ALL STAGES PASSED
