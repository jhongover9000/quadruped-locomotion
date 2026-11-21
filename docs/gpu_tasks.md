# GPU tasks to run by hand

Jobs that need the GPU and that Claude could not run, either because the WSL GPU link
hung in its session or because the job is long. Run them from the repo root in the
`mujoco` env:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mujoco
export MUJOCO_GL=glfw
```

Runs watched by hand need no timeouts: importing packages and compiling kernels the first
time can be silent for many minutes, and the watchdogs then print stack dumps or kill the
run. Add `--no-timeout` to `scripts/train.py` (also on `scripts/evaluate.py`), or
`export QUADRUPED_NO_TIMEOUT=1` once to turn them off for everything started from that
shell, child processes included. For `scripts/dev/run_stages.sh`, use `NO_TIMEOUT=1`.

Tick the task when it is done and note the run folder next to it. Claude reads this file
and the run folders, so the run folder is all it needs to take the results from there.

The phase 5 pipeline validation has its own runbook: `docs/gpu_validation.md`.

---

## G1. Astra: first walking run (PPO, flat ground)

Status: [x] done 11-18. Run folder: `outputs/ppo/flat_velocity/20261118_132425_astra_s0`
(4096 envs, 1500 iterations, return 21.5). It walks: 0.97 m/s when commanded 1.0 m/s, and
no falls. Its joint torques exceed the 5 N m continuous rating; see `docs/orchestration.md`,
Phase 2.

The run on 11-18 at 13:04 (`outputs/ppo/flat_velocity/20261118_130438_astra_s0`) used
`--num-envs 64`. That gave 0.46M steps, against 29.5M for the Go1 run it was compared with
(`20261114_220614_go1_s0`, 4096 envs, 300 iterations). The policy had barely moved off
its initial noise (action_std 0.99 at the end, against 0.31 for Go1), so it only holds the
standing pose. Do not pass `--num-envs`: the default is 4096.

```bash
# Same budget as the Go1 comparison run: 300 iterations, about 10 min on the 4080
python scripts/train.py --algo ppo --robot astra --max-iterations 300

# Or the full default of 1500 iterations, about 45 min
python scripts/train.py --algo ppo --robot astra

# Watch it, with the run folder the train command printed
python scripts/record.py outputs/ppo/flat_velocity/<run>/ --scenario forward
python scripts/record.py outputs/ppo/flat_velocity/<run>/ --scenario tour
```

Compare it with Go1 (iteration, steps, mean return, episode length, action_std):

```bash
for r in 20261114_220614_go1_s0 <run>; do echo "== $r"
  awk -F, 'NR==1||NR==51||NR==101||NR==201||NR==301' outputs/ppo/flat_velocity/$r/progress.csv | cut -d, -f1-4,9
done
```

Go1 at iteration 300: mean return about 22.8, episode length about 996, action_std about
0.31. If Astra's return stays low while action_std falls, the model is what limits it,
not the training budget. Expect a weaker gait than Go1 in any case, because the gains are
placeholders and the URDF stops the thigh at vertical (hip pitch 0 to 90 deg). Tell Claude
the run folder, and it will break the reward down by term from the tfevents file.

---

## G3. Astra: cut the joint torque (three PPO runs)

Status: [ ] not run. Run folders: a ... / b ... / c ...

Why: each joint can hold only about 5 N m continuously (driver temperature). The G1 policy
uses an RMS knee torque of 9 N m at 0.5 m/s and 12.6 N m at 1.0 m/s, and 6.4 N m just standing
(`outputs/debug/astra/forward_torque/summary.txt`). A stance sweep
(`outputs/debug/astra/stance_sweep.md`) found a pose that roughly halves the standing knee
and thigh load and almost removes the hip load. That pose is in the experimental config
`astra_stance2`: hips 0.2 rad inward, thigh 0.65, calf -1.10, trunk at 0.48 m instead of
0.39 m. Since G1, `astra` has also been re-imported with an estimated armature of 0.015 kg m^2.

Each run is 1500 iterations with 4096 envs, about 45 min. Do not pass `--num-envs`.

```bash
# a: baseline on the re-imported model (armature added, old stance)
python scripts/train.py --algo ppo --robot astra
# b: the low-torque stance
python scripts/train.py --algo ppo --robot astra_stance2
# c: the low-torque stance plus a 5x torque penalty (write -1.0e-3 with the dot, so YAML stores a float)
python scripts/train.py --algo ppo --robot astra_stance2 --set rewards.torques=-1.0e-3

# Then record each one with the run folder train.py printed
python scripts/record.py outputs/ppo/flat_velocity/<run>/ --scenario forward
```

`record.py`'s `summary.txt` now ends with a torque table: RMS and peak per joint kind,
largest over the legs. A `*` marks an RMS above 5 N m. Compare the three runs with:

```bash
for r in <run_a> <run_b> <run_c>; do echo "== $r"
  sed -n '/^segment/,/^Columns/p;/^Joint torque/,/^\*/p' \
    outputs/ppo/flat_velocity/$r/recordings/*_forward/summary.txt
done
```

The winner has the lowest knee RMS at 0.5 and 1.0 m/s, while it still tracks 1.0 m/s (rmse
xy under about 0.15) with no falls. If b or c wins, Claude makes it the default `astra`
stance through the importer and removes `astra_stance2`. Expect the knee to stay above
5 N m when walking in every run. That gap is what the team's thermal bench test (how long a
joint lasts at 8, 10 and 12 N m) has to answer.

---

## G4. Astra: bend the knees and lift the feet (presets D, E, F, then G)

Status: D, E and F done 11-20/21 (seed 0 only), tours recorded. G not run.
Run folders: D `20261120_173549_astra_stance2_s0`, E `20261120_184405_astra_stance2_s0`,
F `20261120_225002_astra_stance2_s0` (all under `outputs/ppo/flat_velocity/`).
Verdict (Claude, 11-21): F goes to rough terrain (`astra_G`). E failed: it does not walk
forward at all. Details in `docs/orchestration.md`.

Why: all G3 policies walk on near-straight legs. The mean knee angle is about -0.77 rad
against the stance's -1.10, the trunk rides at 0.50-0.53 m, the knee swings only 0.26-0.36
rad, and the median swing clearance is 25 mm. New cost terms and modeling (11-20) target
this. Each preset is a file in `configs/experiments/` with comments.

| Preset | Adds | Tests |
|---|---|---|
| `astra_D` | swing-height reward (7 cm), foot-slip penalty, knee bend floor 0.9 rad, trunk band 0.43-0.47 m | does shaping alone bend the knees and lift the feet? |
| `astra_E` | D + torque penalty relative to the 5 N m rating, ramped in over iterations 300-1000 (replaces the plain torque term) | can torque come down without losing the gait? |
| `astra_F` | E + fixed action scale 0.25 rad per unit (was 0.094 at the knee) | does more action authority help? |
| `astra_G` | F on `rough_velocity` (stairs up to 10 cm, slopes, rough patches) | the rough-terrain version |

All presets use `astra_stance2`, which now also has the torque-speed curve (24 V and 0.1 ohm
assumed), knee friction randomized over 2.0-2.5 N m, and the armature randomized by ±50%.

First do `wsl --shutdown`; WSL was in its fault state at the end of 11-20. Then run a quick
check, since the CPU smoke of D could not finish on the faulty kernel:

```bash
python scripts/train.py --algo ppo --preset astra_D --num-envs 512 --max-iterations 3 --run-dir outputs/debug/astra/preset_D_gpu
```

Then the runs, two seeds each, about 45 min per run (more for G). `--no-timeout` is
optional:

```bash
for p in astra_D astra_E astra_F; do for s in 0 1; do
  python scripts/train.py --algo ppo --preset $p --seed $s
done; done
# Record each run with the folder train.py printed
python scripts/record.py outputs/ppo/flat_velocity/<run>/ --scenario tour
```

Each recording's `summary.txt` now has three tables: tracking, joint torque (`*` = RMS
above 5 N m), and gait (clearance, joint excursion, mean calf angle, trunk height, stride,
duty, power). What to look for, at walk forward 0.5 and forward 1.0:
- clearance median at or above 50 mm;
- mean calf angle at or beyond -0.9 rad;
- trunk 0.43-0.47 m;
- knee RMS as low as possible;
- rmse xy at or below 0.15, with no falls.

Pick the best of D/E/F and run G for it (G is F on rough terrain; for D or E, run that preset
with `--task rough_velocity`):

```bash
python scripts/train.py --algo ppo --preset astra_G --seed 0
python scripts/record.py outputs/ppo/rough_velocity/<run>/ --scenario tour
```

Give Claude the run folders; it compares them and picks the next step.

---

## G5. Astra: no swing-height reward, plus joint-impulse penalties (presets I, H)

Status: [ ] not run. Run folders: I ... / I_rough ... / H (optional) ...

Why:
- With the swing-height reward (D, F), the policies make long, high airborne steps on
  diagonal pairs (about 75-80 mm), step in place when told to stand, undershoot 0.5 m/s and
  use twice the power. Joseph: smaller steps are fine.
- F is also jittery: joint jerk 3.8e8 against 1.4e8 for D, and a high-frequency torque
  share of 0.26 against 0.13. Its action_rate penalty is about 2.7x weaker in joint terms
  at the knee, because the action scale went from 0.094 to 0.25 rad.

| Preset | What it is |
|---|---|
| `astra_H` | F with `foot_clearance` 0. Keeps the knee bend floor, the trunk band 0.43-0.47 m, foot slip, the ramped rated-torque penalty and the 0.25 rad action scale. |
| `astra_I` | H plus `torque_rate` (sum of ((tau_t - tau_t-1) / 5 N m)^2, ramped in from 25% over iterations 100-500) plus `feet_impact` (foot loads above 0.75 x body weight, normalized by body weight). The aim: no jitter, and soft landings, which need a bent knee. |
| `astra_I_rough`, `astra_H_rough` | the same on `rough_velocity` |

Watch one thing: E (the rated-torque penalty with the small action scale) stopped walking
forward. F, with the larger scale, did not. Check the forward segments first.

```bash
python scripts/train.py --algo ppo --preset astra_I --seed 0
python scripts/record.py outputs/ppo/flat_velocity/<run>/ --scenario tour
# If I walks forward (about 0.4+ m/s when commanded 0.5, 0.85+ at 1.0), go to rough terrain:
python scripts/train.py --algo ppo --preset astra_I_rough --seed 0
python scripts/record.py outputs/ppo/rough_velocity/<run>/ --scenario tour
# Optional, to see what the impulse terms add on their own:
python scripts/train.py --algo ppo --preset astra_H --seed 0
```

What to compare in `summary.txt`, against G3's run B and against F:
- joint_jerk and torque_hf_fraction (B 2.2e7 / 0.056, F 3.8e8 / 0.26);
- mean calf angle, -0.9 rad or beyond;
- tracking at 0.5 and 1.0 m/s;
- power and knee RMS.

---

## G2. Phase 5 pipeline validation

Status: see `docs/gpu_validation.md`.
