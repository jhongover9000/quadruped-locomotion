#!/usr/bin/env bash
# Phase A queue (docs/my_tasks.md A2 + A3): runs one step after another, unattended.
#   1. A2: gauntlet, our PPO vs rsl_rl, both at their 1000-iteration checkpoint (minutes)
#   2. A3: astra_J_rough seeds 1 and 2 (about 1.6 h each)
#   3. A3: gauntlet and beacon on all three J seeds
#
# Start it so it survives closing the terminal (from the repo root):
#   nohup bash scripts/dev/queue_phaseA.sh > /dev/null 2>&1 &
# Follow it:       tail -f outputs/queue/phaseA_*.log
# Stop it:         pkill -f queue_phaseA.sh   (then Ctrl-C or kill the python it started)
#
# A failed step is logged and the queue moves on: a failed seed does not stop the other
# seed, and the course step skips runs that did not finish. A training run that was cut off
# can be picked up later with: python scripts/train.py --continue-run <run dir>/

set -u
cd "$(dirname "$0")/../.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mujoco
export MUJOCO_GL=glfw

mkdir -p outputs/queue
LOG="outputs/queue/phaseA_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

J0=outputs/ppo/rough_velocity/20261122_001801_astra_stance2_s0
RSL=outputs/ppo_rslrl/rough_velocity/20261123_133324_astra_stance2_s0
J1=outputs/ppo/rough_velocity/astra_J_rough_s1
J2=outputs/ppo/rough_velocity/astra_J_rough_s2

step() {   # step "<label>" cmd...
    local label=$1; shift
    echo; echo "=== $(date '+%F %T')  START  $label"; echo "    $*"
    if "$@"; then echo "=== $(date '+%F %T')  OK     $label"
    else echo "=== $(date '+%F %T')  FAILED $label (exit $?)"; fi
}

finished() {   # true if the run's status.json says finished
    python - "$1" <<'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "status.json"
sys.exit(0 if p.exists() and json.loads(p.read_text()).get("finished") else 1)
EOF
}

# 1. A2: equal-iteration check against rsl_rl
step "A2 gauntlet @1000: ppo vs ppo_rslrl" \
    python scripts/course.py "$J0" "$RSL" --course gauntlet --checkpoint model_1000.pt \
        --out outputs/courses/A2_ppo_vs_rslrl_1000

# 2. A3: two more seeds (an existing, unfinished run dir is continued instead of restarted)
for s in 1 2; do
    dir=outputs/ppo/rough_velocity/astra_J_rough_s$s
    if [ -f "$dir/config.yaml" ]; then
        step "A3 train J seed $s (continue)" python scripts/train.py --continue-run "$dir/"
    else
        step "A3 train J seed $s" python scripts/train.py --algo ppo --preset astra_J_rough \
            --seed "$s" --run-dir "$dir"
    fi
done

# 3. A3: courses on every finished seed
runs=("$J0")
for d in "$J1" "$J2"; do
    if finished "$d"; then runs+=("$d"); else echo "skip $d: not finished"; fi
done
step "A3 gauntlet, J seeds" python scripts/course.py "${runs[@]}" --course gauntlet \
    --out outputs/courses/A3_J_seeds_gauntlet
step "A3 beacon, J seeds" python scripts/course.py "${runs[@]}" --course beacon \
    --seeds 1000 1001 1002 1003 --out outputs/courses/A3_J_seeds_beacon

echo; echo "=== $(date '+%F %T')  queue done. Log: $LOG"
grep -E "^=== .*(OK|FAILED)" "$LOG"
