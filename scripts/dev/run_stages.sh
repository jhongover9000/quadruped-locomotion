#!/bin/bash
# Run the local smoke pipeline one stage at a time; retry a stage up to ATTEMPTS times, since
# this workstation has random one-off process failures (fresh Python processes sometimes hang
# during import at 100% CPU with no output). Each attempt's log is kept.
#
#   scripts/dev/run_stages.sh [log_dir]                  # default: outputs/debug/run_stages
#   STAGES="ppo evaluate" scripts/dev/run_stages.sh      # only some stages
#
# Each attempt runs in its own process group. It is declared hung, and its whole process
# group killed, when its log has not grown for HANG_S seconds AND the GPU memory in use is
# below HANG_GPU_MIB (every observed hang happened before CUDA init; legitimate quiet
# periods such as video rendering or a cold torch.compile hold GPU memory). If nvidia-smi
# fails, only the hard limit ATTEMPT_LIMIT_S applies.
#
# Under WSL the GPU passthrough is checked before each attempt and at every poll. When it
# is lost the run stops at once with exit code 3 and the time of the loss in the log,
# since no retry can succeed until WSL is restarted.
#
# Settings (environment): STAGES, ATTEMPTS (5), HANG_S (300), HANG_GPU_MIB (300),
# POLL_S (20), ATTEMPT_LIMIT_S (3000), PIPELINE_ID (smoke_local), and STAGE_CMD: a shell
# command run instead of the real stage, with $STAGE set (for testing this driver).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate mujoco
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
S="${1:-$REPO/outputs/debug/run_stages}"; mkdir -p "$S"; S="$(cd "$S" && pwd)"
export SMOKE=1 PIPELINE_ID="${PIPELINE_ID:-smoke_local}"
ATTEMPTS="${ATTEMPTS:-5}"
HANG_S="${HANG_S:-300}"
HANG_GPU_MIB="${HANG_GPU_MIB:-300}"
POLL_S="${POLL_S:-20}"
ATTEMPT_LIMIT_S="${ATTEMPT_LIMIT_S:-3000}"

# Total GPU memory in use, in MiB; empty if nvidia-smi fails.
gpu_mib() {
  local out total=0 v
  out="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" || return 0
  [[ -n "$out" ]] || return 0
  for v in $out; do [[ "$v" =~ ^[0-9]+$ ]] || return 0; total=$((total + v)); done
  echo "$total"
}

# True while the WSL GPU passthrough is up (always true outside WSL). The driver mount
# becoming unreadable is the symptom of a lost passthrough; only `wsl --shutdown` from
# Windows brings it back, so nothing is worth retrying after that.
gpu_alive() { [[ ! -d /usr/lib/wsl ]] || ls /usr/lib/wsl/drivers > /dev/null 2>&1; }

# TERM the whole process group, then KILL what is left after 5 s.
kill_group() {
  local pgid="$1" i
  kill -TERM -- "-$pgid" 2>/dev/null
  for i in 1 2 3 4 5; do kill -0 -- "-$pgid" 2>/dev/null || return 0; sleep 1; done
  kill -KILL -- "-$pgid" 2>/dev/null
}

# run_attempt STAGE ATTEMPT LOG -> sets RC and HUNG (1 if killed as hung)
run_attempt() {
  local stage="$1" attempt="$2" log="$3" pid start now size last_size last_growth last_poll quiet mib
  : > "$log"
  # Not a process-group leader (no job control), so setsid execs in place: pid = pgid.
  if [[ -n "${STAGE_CMD:-}" ]]; then
    STAGE="$stage" setsid bash -c "$STAGE_CMD" > "$log" 2>&1 < /dev/null &
  else
    setsid hpc/submit_pipeline.sh --local --only "$stage" > "$log" 2>&1 < /dev/null &
  fi
  pid=$!
  start=$(date +%s); last_growth=$start; last_poll=$start; last_size=0; HUNG=0; RC=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1
    now=$(date +%s)
    if (( now - start >= ATTEMPT_LIMIT_S )); then
      echo "!!!! stage $stage attempt $attempt: hard limit of $ATTEMPT_LIMIT_S s reached; killing" | tee -a "$log"
      kill_group "$pid"; break
    fi
    (( now - last_poll >= POLL_S )) || continue
    last_poll=$now
    if ! gpu_alive; then
      echo "!!!! GPU LOST at $(date +%H:%M:%S), $((now - start)) s into stage $stage attempt $attempt; killing its process group" | tee -a "$log"
      GPU_LOST=1; kill_group "$pid"; break
    fi
    size=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [[ "$size" != "$last_size" ]]; then last_size=$size; last_growth=$now; continue; fi
    quiet=$((now - last_growth))
    (( quiet >= HANG_S )) || continue
    mib="$(gpu_mib)"
    [[ -n "$mib" ]] || continue                     # GPU use unknown: no hang verdict
    if (( mib < HANG_GPU_MIB )); then
      echo "!!!! HANG: stage $stage attempt $attempt: no output for $quiet s, GPU memory in use $mib MiB (< $HANG_GPU_MIB); killing its process group" | tee -a "$log"
      HUNG=1; kill_group "$pid"; break
    fi
  done
  wait "$pid" 2>/dev/null; RC=$?
  if [[ $HUNG -eq 1 && $RC -eq 0 ]]; then RC=1; fi
  if [[ $GPU_LOST -eq 1 && $RC -eq 0 ]]; then RC=1; fi
}

declare -A RESULT USED HANGS WALL
ORDER=(); FAILED=0; GPU_LOST=0
for stage in ${STAGES:-check ppo mpc motions mimic supermimic evaluate}; do ORDER+=("$stage"); done
for stage in "${ORDER[@]}"; do
  RESULT[$stage]="not run"; USED[$stage]=0; HANGS[$stage]=0; WALL[$stage]="-"
done
for stage in "${ORDER[@]}"; do
  rc=1
  for attempt in $(seq 1 "$ATTEMPTS"); do
    log="$S/stage_${stage}_$attempt.log"
    if ! gpu_alive; then
      echo "$(date +%H:%M:%S) GPU LOST before stage $stage attempt $attempt; not starting it"
      GPU_LOST=1; break
    fi
    start=$(date +%s)
    run_attempt "$stage" "$attempt" "$log"; rc=$RC
    secs=$(( $(date +%s) - start ))
    USED[$stage]=$attempt; HANGS[$stage]=$(( ${HANGS[$stage]} + HUNG ))
    echo "$(date +%H:%M:%S) stage $stage attempt $attempt rc=$rc ($secs s)$([[ $HUNG -eq 1 ]] && echo ', killed as hung')$([[ $GPU_LOST -eq 1 ]] && echo ', GPU lost')"
    if [[ $rc -eq 0 ]]; then RESULT[$stage]="passed"; WALL[$stage]="$secs s"; break; fi
    [[ $GPU_LOST -eq 0 ]] || break
    grep -E "Error|ERROR|MALFORMED|Killed|assert" "$log" | tail -3 | cut -c1-200
  done
  if [[ $GPU_LOST -eq 1 ]]; then
    RESULT[$stage]="GPU LOST"; FAILED=1
    echo "GPU passthrough lost (cannot read /usr/lib/wsl/drivers); run 'wsl --shutdown' from Windows. Stopping."; break
  fi
  if [[ $rc -ne 0 ]]; then
    RESULT[$stage]="FAILED"; FAILED=1
    echo "stage $stage failed $ATTEMPTS times; stopping"; break
  fi
done

echo
printf '%-12s %-8s %-9s %-6s %s\n' stage result attempts hangs "wall time (passing attempt)"
for stage in "${ORDER[@]}"; do
  printf '%-12s %-8s %-9s %-6s %s\n' "$stage" "${RESULT[$stage]}" "${USED[$stage]}" "${HANGS[$stage]}" "${WALL[$stage]}"
done
if [[ $GPU_LOST -eq 1 ]]; then exit 3; fi
if [[ $FAILED -eq 1 ]]; then exit 1; fi
echo ALL STAGES PASSED
