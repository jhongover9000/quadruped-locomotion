# Spec: DeepMimic, AMP and MPC in the `hpc/` pipeline (task P5.4)

Written 2026-11-16 by the main session, from a survey of `hpc/` at commit `4c39b41` and
the three reviewed implementations (specs `deepmimic.md`, `amp.md`, `mpc.md`). Work in the
integration worktree `.claude/worktrees/integration-p5`, which holds all three
implementations merged on top of `4c39b41`. No commits.

## Goal

One `hpc/submit_pipeline.sh` run trains and evaluates every implemented method: PPO,
rsl_rl PPO, SuperMimic, and now DeepMimic, AMP and the MPC baseline. The report compares
them. `SMOKE=1 ... --local` still finishes on the laptop.

## Decisions already taken

1. **Two new stages.** The order becomes
   `check ppo mpc motions mimic supermimic evaluate`.
   - `mpc` (file `hpc/22_mpc.slurm`): builds the controller run. Needs `check` only.
   - `mimic` (file `hpc/35_train_mimic.slurm`): trains DeepMimic and AMP. Both read the
     motion library `data/npz/<robot>` that `motions` builds, so the stage needs
     `afterok:motions`, exactly like the first SuperMimic segment, and runs alongside
     SuperMimic on Slurm.
2. **`mimic` is a job array like `ppo`**, over `MIMIC_RUNS` x `SEEDS`. `MIMIC_RUNS` is a
   list of `algo:task` pairs, default `"deepmimic:motion_tracking amp:flat_velocity"`.
   Reuse `train_run` from `hpc/lib.sh` (stall restart, `--continue-run` when the run
   exists). Run folders follow the existing rule
   `$QUADRUPED_OUTPUT_DIR/<algo>/<task>/${PIPELINE_ID}_${ROBOT}_s<seed>` and are appended
   to `runs.txt` without duplicates (use the `grep -qxF` guard SuperMimic uses; give the
   PPO stage the same guard while you are there).
3. **One time-limited segment per mimic run**, not a chain. Pass
   `--max-hours $MIMIC_MAX_HOURS` so the run stops cleanly before the Slurm limit, and
   print the same kind of warning SuperMimic's last segment prints when a run did not reach
   its last iteration ("resubmit with `--only mimic --id <id>` to continue"). Defaults
   mirror one SuperMimic segment (same time limit and hours). They are provisional until
   the A100 speed is measured (P1.4); say so in a comment.
4. **New settings in `hpc/config.sh`**, next to the PPO ones, each overridable in
   `config.local.sh` like the rest: `MIMIC_RUNS`, `DM_NUM_ENVS=4096`, `DM_ITERATIONS=2000`,
   `AMP_NUM_ENVS=4096`, `AMP_ITERATIONS=1500`, `MIMIC_MAX_HOURS`, and the stage's Slurm
   time. Under `SMOKE=1`: `DM_NUM_ENVS=256`, `DM_ITERATIONS=3`, `AMP_NUM_ENVS=64`,
   `AMP_ITERATIONS=3`, `MIMIC_MAX_HOURS=1`. If `amp.yaml` / `deepmimic.yaml` set a value
   that conflicts with these defaults, the YAML's value is the default and you say so.
5. **MPC is built, not trained.** The stage runs
   `python scripts/train.py --algo mpc --task flat_velocity --robot $ROBOT --run-dir <run>`
   directly (not through `train_run`: no stall watchdog, no restart), skips the build when
   `<run>/checkpoints/model_0.pt` already exists, and records the run in `runs.txt`. Short
   Slurm time limit (10 minutes). Its job ID joins `TRAINED` so `evaluate` waits for it.
6. **`train.py --continue-run` must not raise `KeyError`** for an algorithm whose config
   has no `max_iterations` (MPC; see `scripts/train.py` near line 128). Treat a missing
   value as 0. Smallest possible change.
7. **`check` stage** (`hpc/10_check.slurm`): after the existing PPO checks add
   - `train.py --algo mpc` into the check's scratch run folder (seconds);
   - a 3-iteration run of `amp` and of `deepmimic`, only when the motion library's
     `manifest.csv` already exists; otherwise print one line saying they are skipped
     because the library is built later by `motions`. Must not fail on a fresh cluster.
8. **`evaluate` stage** (`hpc/50_evaluate.slurm`) and report:
   - The per-task comparison plot is only made for 2 to 8 runs. With the new runs
     `flat_velocity` has 10 (PPO x3, rsl_rl x3, AMP x3, MPC). Keep the comparison: runs
     without training curves (MPC) are left out of training plots quietly, and the limit
     is raised or the plot reduced to what stays legible. Your call; say what you did.
   - A run with no training curves must not print a traceback or fail the stage.
   - `evaluation.csv` and the recordings must include the DeepMimic, AMP and MPC runs.
9. **Docs in the same change:** `hpc/README.md` (stage table, flow diagram, the new
   settings), `README.md` (method table rows and the paragraph near lines 20 to 23 that
   still calls DeepMimic, AMP and MPC planned; point to the three specs' Deviations for
   known limits: the MPC falls at 1.5 m/s, DeepMimic's learning check is pending),
   `submit_pipeline.sh --help`, and the default stage list in `scripts/dev/run_stages.sh`.
   Do not rewrite the part of `hpc/README.md` that says what was run end to end; the main
   session rewrites that from the validation evidence (P0.7).

## Files

`hpc/submit_pipeline.sh` (stage list, `wanted`, the `--local` block, Slurm dependencies,
`TRAINED`, array size for `mimic`, help text), `hpc/config.sh`, `hpc/lib.sh` only if a
helper is needed, new `hpc/22_mpc.slurm` and `hpc/35_train_mimic.slurm`,
`hpc/10_check.slurm`, `hpc/50_evaluate.slurm`, `hpc/20_train_ppo.slurm` (dedup guard
only), `scripts/train.py` (decision 6 only), `scripts/dev/run_stages.sh`, `README.md`,
`hpc/README.md`. Touch `src/` only if decision 8 cannot be met otherwise, and then as
little as possible.

## Acceptance checks

No GPU is needed for these; run them all from the worktree.

```
for f in hpc/*.sh hpc/*.slurm scripts/dev/*.sh; do bash -n "$f" || echo "SYNTAX $f"; done
hpc/submit_pipeline.sh --dry-run --id spec_dry                # 7 stages, dependencies as in decision 1
hpc/submit_pipeline.sh --dry-run --id spec_dry --parallel     # motions no longer waits for ppo; mimic still waits for motions
SMOKE=1 hpc/submit_pipeline.sh --dry-run --local --id spec_dry_smoke
hpc/submit_pipeline.sh --dry-run --only mimic --id spec_dry
hpc/submit_pipeline.sh --dry-run --skip mpc --skip mimic --id spec_dry   # identical plan to today's five stages
hpc/submit_pipeline.sh --only nonsense                         # still rejected
CUDA_VISIBLE_DEVICES="" pytest -q
```

If `--dry-run` does not print enough to check the dependencies and the array size, make
it print them. Also run on the CPU, as far as it goes without a GPU, the two new stage
scripts in local smoke mode (`SMOKE=1 hpc/submit_pipeline.sh --local --only mpc --id
spec_cpu`, and the same for `mimic` if the stage can be pointed at the CPU; if it cannot,
do not add a device switch for it, just list the command).

GPU checks, run later by the runner, list them in your report as exact commands:
`SMOKE=1 PIPELINE_ID=<id> hpc/submit_pipeline.sh --local` end to end, with the report
holding `evaluation.csv` rows and a recording for all six methods.

## Out of scope

Tuning any algorithm; changing what `motions` builds; sweeps; anything that submits to a
real Slurm cluster; commits.

## Deviations

Record here anything you had to do differently from this spec and why. If you hit a
choice the spec does not cover, take the conservative option, note it here and continue.

Implementation (worktree `integration-p5`), 2026-11-16:

- **Comparison plot (decision 8, "your call"):** the limit stays at 8 lines (the palette
  size in `evaluation/curves.py`, `MAX_SERIES`). A task with more than 8 trained runs gets
  one `compare_<task>_s<seed>.png` per seed, each with one run per method (flat_velocity:
  ppo, ppo_rslrl, amp). Up to 8 runs: one `compare_<task>.png` as before. Runs without
  TensorBoard event files (MPC) are filtered out of all training plots in the stage
  before `plot_training.py` is called, so no message or traceback. No `src/` change.
- **Slurm time of `mimic`** is a setting, `MIMIC_TIME` (default `20:00:00`, `01:00:00`
  under `SMOKE=1`), passed as `sbatch --time`; the file's `#SBATCH --time=20:00:00` covers
  submission by hand. `MIMIC_MAX_HOURS` default 19 (as `SM_SEGMENT_HOURS`).
- **No YAML conflict:** `deepmimic.yaml` max_iterations 2000 and `ppo.yaml` (inherited by
  AMP) 1500 / 4096 envs equal the new defaults.
- **`check` sizes:** AMP 64 envs, DeepMimic 256 envs (the SMOKE values), 3 iterations.
  The library test is `$SUPERMIMIC_ROOT/data/npz/$ROBOT/manifest.csv`, the path stage
  `motions` builds.
- **`--dry-run` prints the array range** on each array job's summary line
  (`quad_mimic job 1005 array 0-5 (after: afterok:1004)`). With `--skip mpc --skip mimic`
  the sbatch commands are byte-identical to before; only the summary lines differ.
- **`--help`** now prints the header comment up to `set -euo` instead of a fixed line range.
- **File modes:** `hpc/submit_pipeline.sh` and `scripts/dev/run_stages.sh` are committed
  100644, so the acceptance commands could not run them directly in the worktree. They
  were made executable (as in the main checkout's working tree, where the same mode change
  is uncommitted).
- **README.md:** the methods table marks what is built; the paragraph and the Workflow row
  for phase 4 now say DeepMimic, AMP and MPC are built, with links to the three specs'
  Deviations.
- **CPU checks:** `SMOKE=1 ... --local --only mpc` and `--only mimic` both ran to the end
  on the CPU (`CUDA_VISIBLE_DEVICES=""`; `train.py` falls back to the CPU, no device switch
  added), as did the evaluate stage's plotting section on those three runs (MPC skipped
  quietly). Two machine hangs (pytest, and AMP's `train.py` before it created its run
  folder), both killed and rerun successfully.
- Pre-existing, not changed: `LOG_DIR` is set in `config.sh` before `lib.sh`'s workstation
  fallback for `PROJECT_DIR`, so on a workstation dry runs print `/scratch/$USER/...` log
  paths.
