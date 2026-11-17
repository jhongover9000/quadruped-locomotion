#!/bin/bash
# Freeze a real training run and check: the supervisor ends it with exit code 75, and
# --continue-run picks it up from the last checkpoint. Needs the GPU.
#
#   scripts/dev/stall_test.sh [work_dir]     # default: outputs/debug/stall_test
#
# Exits 0 only if the supervisor exit code is 75, ckpt_latest.pt exists, and the continued
# run resumes from iteration 100 or later.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate mujoco
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
S="${1:-$REPO/outputs/debug/stall_test}"; mkdir -p "$S"; S="$(cd "$S" && pwd)"
rm -rf "$S/stall_run" "$S/stall1.log"
python scripts/train.py --algo supermimic --num-envs 256 --max-iterations 3000 --stall-timeout 40 \
    --run-dir "$S/stall_run" > "$S/stall1.log" 2>&1 &
SUP=$!
for i in $(seq 1 150); do grep -q "^it 200 " "$S/stall1.log" 2>/dev/null && break; sleep 2; done
CHILD=$(grep -o "supervising pid [0-9]*" "$S/stall1.log" | awk '{print $3}')
if [[ -z "$CHILD" ]]; then
  echo "FAIL: no 'supervising pid' line in $S/stall1.log"; kill "$SUP" 2>/dev/null; exit 1
fi
echo "training child pid $CHILD; freezing it after: $(grep '^it ' "$S/stall1.log" | tail -1 | cut -c1-30)"
kill -STOP "$CHILD"; t0=$(date +%s)
wait $SUP; rc=$?
echo "supervisor exit code $rc after $(( $(date +%s) - t0 )) s frozen"
grep -E "watchdog\] pid" "$S/stall1.log" | tail -2 | cut -c1-160
echo "checkpoints: $(ls "$S/stall_run" | grep ckpt | tr '\n' ' ')"
python scripts/train.py --continue-run "$S/stall_run" --max-hours 0.006 > "$S/stall2.log" 2>&1
echo "continue exit code $?"; grep -E "iterations [0-9]+ ->|Stopped|Finished" "$S/stall2.log" | cut -c1-110

# ---- verdict ------------------------------------------------------------------------
resumed=$(grep -oE "iterations [0-9]+ ->" "$S/stall2.log" | head -1 | grep -oE "[0-9]+")
fail=0
[[ $rc -eq 75 ]] || { echo "FAIL: supervisor exit code $rc, expected 75"; fail=1; }
[[ -f "$S/stall_run/ckpt_latest.pt" ]] || { echo "FAIL: no ckpt_latest.pt in $S/stall_run"; fail=1; }
[[ -n "$resumed" && $resumed -ge 100 ]] \
    || { echo "FAIL: continued run resumed from iteration '${resumed:-?}', expected >= 100"; fail=1; }
[[ $fail -eq 0 ]] && echo "PASS: exit code 75, ckpt_latest.pt present, resumed from iteration $resumed"
exit $fail
