# Spec: DeepMimic (`--algo deepmimic`)

Task P5.1 in `docs/orchestration.md`. Written 2026-11-16 against the working tree after
commit `c0abbe8`. Line numbers are from that tree and will drift.

## Goal

DeepMimic here is the model-free counterpart of SuperMimic: the in-house PPO trained to
track the same reference clips, with the same rewards, terminations and reference-state
resets. The value of having it is the comparison, so the two methods must see the same
environment. A reader of `evaluation.csv` should be able to put a DeepMimic row next to
a SuperMimic row and compare them.

## Decisions already taken

These are the choices most likely to be questioned, so they come first.

1. **PPO runs on the SuperMimic backend's tracking environment**, not on a new native
   task. `src/quadruped_rl/envs/builder.py` cannot build `configs/tasks/motion_tracking.yaml`
   today (it has no observation, reward or action section, and `builder.py:199` rejects
   imitation reward terms), and `envs/tasks/motion_tracking.py` is an empty stub. The
   backend's environment already has multi-clip sampling, reference-state initialisation
   and early termination. The cost is that DeepMimic needs `~/code/supermimic-quadruped`,
   as SuperMimic does. `deepmimic/agent.py`'s docstring names this route;
   `deepmimic/README.md` describes the native one and should be corrected.
2. **`DeepMimicAgent` subclasses `PPOAgent`.** `PPOAgent.learn()` should run unchanged.
   What differs is the environment and the checkpoint identity.
3. **Actor and critic see the same observation to start with**: the backend's `policy`
   group (history of 10) concatenated with its `goal` group. A privileged critic is a
   later improvement, not part of this task.
4. **The backend's action noise is off for DeepMimic.** The backend uses
   `NoisyJointPositionAction` (default std 0.1) because SuperMimic needs it for
   exploration. PPO explores through its own action distribution.

## What exists

- Contract: `src/quadruped_rl/algorithms/base.py` (`Agent`, `NotImplementedAgent`).
  Registry and `check_compatible`: `src/quadruped_rl/algorithms/__init__.py`.
- PPO: `src/quadruped_rl/algorithms/ppo/agent.py`. It reads `obs["actor"]` and
  `obs["critic"]` (hard-coded near lines 54, 82, 124, 178, 254) and expects
  `env.step(actions) -> obs, rewards, terminated, truncated, extras`. Checkpoints are
  `checkpoints/model_{iteration}.pt` with keys `model, optimizer, iteration, learning_rate`.
- Tracking environment and clip discovery: `src/quadruped_rl/algorithms/supermimic/agent.py`
  (`make_env`, `clip_files`, `benchmark`) and `backend.py` (`find_root`, `resolve_path`,
  `build_train_config`). The backend env's observation groups are `policy` and `goal`.
- Evaluation: `src/quadruped_rl/evaluation/rollout.py` `load_run` (line 114) branches on
  the task kind. For tracking runs it calls `agent.clip_files(...)`,
  `agent.make_env(n, play=True, seed=, clips=, terminations=)` and
  `agent.make_policy(env, ckpt)`. `scripts/evaluate.py` `evaluate_tracking` (line 110)
  also calls `agent.benchmark(ckpt, groups, num_envs=, out=)`, which only
  `SuperMimicAgent` has.
- Checkpoint format detection: `src/quadruped_rl/algorithms/checkpoints.py` `_SIGNATURES`
  takes the first match in dict order, so a file with PPO's keys is detected as `ppo`.

## Requirements

- `python scripts/train.py --algo deepmimic` trains on `motion_tracking` and
  `train.py --list` shows it as implemented for tracking tasks only.
- One source of truth for the clip set, rewards, terminations and reset sampling shared
  with SuperMimic. Share the code that builds the environment; do not copy it.
  SuperMimic's own behaviour must not change.
- `--continue-run`, `--max-hours` and `--stall-timeout` work as they do for PPO.
- Resuming or playing a checkpoint that came from a different algorithm, including a
  velocity PPO run, fails with the same kind of clear message the repo gives today.
- `scripts/record.py`, `scripts/play.py` and `scripts/plot_training.py` work on a
  DeepMimic run with no special flags.
- `scripts/evaluate.py` writes a DeepMimic row with the same columns as a SuperMimic
  row. If the reference evaluator behind `benchmark` can take a different policy through
  an adapter, use it, so the numbers are computed identically. If it cannot, compute
  them from the tracking rollouts and say in the report which columns are not exactly
  comparable.
- `tests/test_compatibility.py` is updated for the new status. New tests cover the
  observation adapter on CPU, and a `sim`-marked test trains for a couple of iterations
  and builds a policy.

## Acceptance checks

Run from the repo root in the `mujoco` conda env. Put outputs under `outputs/debug/p5/`.

```
pytest -q
pytest -m sim -q
python scripts/train.py --list
python scripts/train.py --algo deepmimic --num-envs 256 --max-iterations 30 --run-dir outputs/debug/p5/deepmimic_smoke
python scripts/train.py --continue-run outputs/debug/p5/deepmimic_smoke --max-hours 0.01
python scripts/record.py outputs/debug/p5/deepmimic_smoke
python scripts/evaluate.py outputs/debug/p5/deepmimic_smoke outputs/supermimic/motion_tracking/smoke_local_go1_s0
python scripts/train.py --algo supermimic --num-envs 256 --max-iterations 30 --run-dir outputs/debug/p5/supermimic_regress
```

Then a learning check: 300 iterations at 1024 envs. Report mean return and mean episode
length at the start and the end. They should rise. If they do not, find out why (reward
scale, discount, observation normalisation, reset distribution) before reporting; do not
tune the check until it passes. Use the result to choose a sensible default
`max_iterations` for `configs/algorithms/deepmimic.yaml` and say what you chose.

## Out of scope

A native tracking task in `envs/builder.py`; a privileged critic; changes to the
SuperMimic repo; the `hpc/` pipeline and the README method table (task P5.4); committing.

## Deviations

Record here anything you had to do differently from this spec and why. If you hit a
choice the spec does not cover, take the conservative option, note it here and continue.
