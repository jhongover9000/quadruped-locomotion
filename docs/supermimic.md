# SuperMimic in this repository

SuperMimic (Liao et al., arXiv 2610.28480) learns a differentiable world model of the
robot from simulator data and improves a motion-tracking policy by backpropagating
through short imagined rollouts. Here it tracks dog motion capture retargeted to the Go1.

## How it is wired in

The implementation is the reference repository `supermimic-quadruped`; this repository
adapts it rather than copying it.

| Piece | Where |
|---|---|
| World model, imagined rollouts, policy update, tracking environment, GPU motion library | reference repository, `src/supermimic/` |
| Go1 robot spec (dimensions, calibrated state scales, reward scales, mirror map) | reference repository, `src/supermimic/robots/go1.py` |
| Agent adapter: train, play, evaluate through the common interface | `src/quadruped_rl/algorithms/supermimic/agent.py` |
| Finding the reference repository, YAML → its `TrainConfig`, robot consistency check | `src/quadruped_rl/algorithms/supermimic/backend.py` |
| Settings | `configs/algorithms/supermimic.yaml`, `configs/tasks/motion_tracking.yaml` |
| Clip format and library view (numpy only) | `src/quadruped_rl/motions/` |

The reference repository is looked for at `backend.root` in the algorithm YAML, then
`$SUPERMIMIC_ROOT`, then `../supermimic-quadruped` next to this repository.

Because of the adapter, a SuperMimic run is an ordinary run directory:
`train.py`, `play.py`, `record.py`, `replay.py`, `evaluate.py` and `plot_training.py`
work on it as on a PPO run.

## Commands

```bash
python scripts/motions.py                                # the clip library: groups, counts
python scripts/train.py --algo supermimic                # 4096 envs x 50000 iterations
python scripts/train.py --algo supermimic --num-envs 1024 --max-iterations 3000   # ~15 min, ~4 GiB
python scripts/train.py --continue-run outputs/supermimic/motion_tracking/<run>/  # after an interruption
python scripts/play.py   outputs/supermimic/motion_tracking/<run>/                # viewer, holdout clips
python scripts/record.py outputs/supermimic/motion_tracking/<run>/                # video with reference ghost
python scripts/evaluate.py outputs/supermimic/motion_tracking/<run>/              # clip benchmark + smoothness
python scripts/plot_training.py outputs/supermimic/motion_tracking/<run>/         # training dashboard
```

Any field of the reference `TrainConfig` can be set from the command line, for example
`--algo-set policy_trainer.horizon=16 schedule.updates_per_env_step=0.2`.

## Reading a SuperMimic run

* **Iterations are not PPO iterations.** One iteration is one control step in every
  environment; a PPO iteration is 24. Plots default to environment steps so the two line up.
* **Return is normalized**: reward per step divided by the maximum, so 1 is perfect
  tracking. It is not comparable with a velocity task's return.
* **What to watch in `training_curves.png`**: world-model error (the 8-step error should
  fall well below 0.1), imagined survival to the horizon (should approach 1), episode
  length (longer means the robot follows clips for longer before straying), and the
  benchmark numbers from `evaluate.py` (clips failed, on training and holdout clips).
* **Checkpoints** are `ckpt_<iteration>.pt` (190 MB each: world model, policy, critic and
  three optimizers); `ckpt_latest.pt` is the same as the newest numbered one.

## The motion library

520 clips at 50 Hz (dog motion capture from two public sources, retargeted with dynamic
similarity scaling: lengths × 0.74, velocities × 0.86 from the source robot to the Go1).
Each clip was screened in simulation for whether the Go1 can physically follow it, then
assigned to a group in `manifest.csv`:

| Group | Clips | Use |
|---|---|---|
| train | 275 (32 min) | training |
| hard | 179 (20 min) | training; the screen found them difficult |
| holdout | 66 (6 min) | never trained on; used by play, record and evaluate |

Counts include mirrored copies. The data is licensed for non-commercial research
(CC BY-NC and a research-only licence), so it is not committed here; it is built by the
reference repository's scripts (`scripts/data/`, run by `hpc/30_supermimic_motions.slurm`):
`fetch_sources.py`, `build_dataset.py --robot go1`, `check_library.py --robot go1`,
`screen_clips.py --robot go1 --stiffness 76`, `assign_groups.py --robot go1`.

## Known differences and limits

* **Default pose.** `configs/robots/go1.yaml` stands with the hips at ±0.1 rad; the
  SuperMimic Go1 spec uses 0. Gains, torque limits, armature and action scale agree.
  Training prints this as a warning and `tests/test_compatibility.py` fails if any other
  difference appears. Each policy is consistent with the pose it trained on, but a shared
  hardware interface needs one of the two changed: changing the spec means recalibrating
  its state scales and retraining.
* **Contact sensing.** The tracking environment has no foot contact sensor, so recordings
  estimate contact from foot height (flat ground only).
* **Hangs at the first compile.** The trainers use `torch.compile`; the first compilation
  on a machine occasionally never finishes. `train.py` reports a stall after 5 minutes
  (stacks of every thread) and `--stall-timeout` restarts the run. Compilation can be
  switched off with `--algo-set policy_trainer.compile=null wm_trainer.compile=null`,
  at about a fifth of the speed.
* **Adding a robot.** Write its `RobotSpec` in the reference repository (its
  `docs/PORTING.md`), build and screen a motion library for it, and add the name to
  `backend.robot_specs` in `configs/algorithms/supermimic.yaml`.
