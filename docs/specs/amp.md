# Spec: AMP, adversarial motion priors (`--algo amp`)

Task P5.2 in `docs/orchestration.md`. Written 2026-11-16 against the working tree after
commit `c0abbe8`. It can be built alongside DeepMimic (P5.1) and MPC (P5.3): DeepMimic
subclasses `PPOAgent` without editing `learn()`, so the only shared edits are small ones
in `tests/test_compatibility.py` and `src/quadruped_rl/algorithms/checkpoints.py`.

## Goal

AMP trains a velocity-following policy whose gait looks like the reference clips. It is
PPO on the existing velocity tasks, with the reward mixed from the task reward and a
style reward given by a discriminator that tells policy motion from clip motion:
`r = w_task * r_task + w_style * r_style`. It gives the project a third way to use the
motion library, next to SuperMimic and DeepMimic, that does not track a clip frame by
frame.

## Decisions already taken

1. **Velocity tasks, not the tracking environment.** `amp/agent.py` already declares
   `task_kinds = ("velocity",)`. The environment is the one `envs/builder.py` builds for
   `flat_velocity` and `rough_velocity`, unchanged.
2. **`PPOAgent.learn()` gets hook methods; AMP overrides them.** `learn()` has no
   extension points today: reward handling, storage and logging are inline (about lines
   94 to 164). Refactor it so a subclass can change the reward after each step, add
   losses and optimiser steps to the update, add scalars to the log and add state to the
   checkpoint. The hooks default to doing nothing and PPO's behaviour must not change.
   This is preferred over a second copy of the loop.
3. **Clips come from the same library SuperMimic uses**, resolved the same way
   (`supermimic/backend.py` `find_root` and `resolve_path`, pattern `data/npz/{robot}`),
   with a filter in the config for which manifest groups or categories count as the
   style. `configs/algorithms/amp.yaml` points at `data/motions/retargeted`, which is
   empty; change it.
4. **Loss and reward as in the stub's docstrings**: least-squares GAN loss with a
   gradient penalty on clip samples, and
   `r_style = max(0, 1 - 0.25 * (D(s, s') - 1)^2)`. Config values already in `amp.yaml`:
   hidden sizes 1024 and 512, 2 frames, learning rate 1e-4, penalty 10, replay buffer of
   1,000,000 transitions, weights 0.5 and 0.5.

## What exists

- PPO: `src/quadruped_rl/algorithms/ppo/{agent,networks,storage}.py`. `RolloutStorage`
  and `_update(storage) -> dict` are separable. Logging goes through `RunLogger`
  (`src/quadruped_rl/utils/logging.py`); `progress.csv` has fixed columns (`CSV_COLUMNS`,
  `agent.py:33`) and other scalars go to TensorBoard only.
- Clip format: `src/quadruped_rl/motions/format.py`. One `.npz` per clip at 50 frames
  per second, which equals the control rate (physics step 0.005 s, decimation 4), so a
  clip transition and an environment transition span the same time. Arrays: `joint_pos`
  and `joint_vel` `[T, 12]` in the robot YAML's joint order, and `body_pos_w`,
  `body_quat_w` (wxyz), `body_lin_vel_w`, `body_ang_vel_w` for 13 bodies with the trunk
  at index 0. There are no foot bodies in the clips.
- Library: `src/quadruped_rl/motions/motion_lib.py` `MotionLibrary`, numpy only. The Go1
  manifest has columns including `group`, `category`, `keep` and `mirror_of`.
- Robot state in the environment: `env.scene["robot"].data`.
- Stubs to fill: `src/quadruped_rl/algorithms/amp/{agent,discriminator,motion_buffer}.py`.

## Requirements

- The discriminator's input features are computed by one function used for both clip
  frames and environment states, so the two cannot drift apart. Choose features that
  exist on both sides and do not reveal the absolute position or heading: joint
  positions and velocities, base linear and angular velocity in the base frame, the
  gravity direction in the base frame and base height are the usual set. A test must
  show that a clip frame written into the simulator gives the same features from both
  paths.
- A transition that crosses an environment reset never reaches the discriminator or the
  replay buffer, and gets no style reward computed from the post-reset state.
- Discriminator inputs are normalised, and the normaliser, the discriminator and its
  optimiser are saved and restored, so `--continue-run` continues the same training.
- The policy checkpoint still loads for `play.py`, `record.py` and `evaluate.py` through
  the ordinary velocity path, with no AMP-specific flags.
- Training curves show the discriminator loss, its mean output on clip and on policy
  samples, the mean style reward and the mean task reward.
- A checkpoint from another algorithm is refused with a clear message, and a PPO run
  refuses an AMP checkpoint (see `algorithms/checkpoints.py` `_SIGNATURES`).
- `tests/test_ppo.py` passes without edits after the refactor. `tests/test_compatibility.py`
  is updated. New CPU tests cover the feature function, the motion buffer's sampling and
  the reset masking.

## Acceptance checks

```
pytest -q
pytest -m sim -q
python scripts/train.py --algo ppo --num-envs 256 --max-iterations 20 --run-dir outputs/debug/p5/ppo_regress
python scripts/train.py --algo amp --num-envs 256 --max-iterations 30 --run-dir outputs/debug/p5/amp_smoke
python scripts/train.py --continue-run outputs/debug/p5/amp_smoke --max-hours 0.01
python scripts/record.py outputs/debug/p5/amp_smoke
python scripts/evaluate.py outputs/debug/p5/amp_smoke
```

Then a learning check: 300 iterations at 1024 envs on `flat_velocity`. Report the task
return, the style reward and the discriminator's mean output on clip and policy samples,
at the start and the end. A healthy run has the discriminator neither perfectly
separating the two nor unable to separate them at all, and a task return in the same
range as the plain PPO run at 300 iterations (22.8, in
`outputs/ppo/flat_velocity/20261114_220614_go1_s0`). If the run is not healthy, report
what you see and what you tried; do not adjust the weights until the numbers look right
without understanding why.

## Out of scope

Tracking tasks; changes to the motion library's file format; the `hpc/` pipeline and the
README method table (P5.4); committing.

## Deviations

Record here anything you had to do differently from this spec and why. If you hit a
choice the spec does not cover, take the conservative option, note it here and continue.

Implementation (worktree `agent-aa519201bbcd5026d`), 2026-11-16:

- **Style reward is scaled by the control step:** `r = w_task * r_task + w_style * dt * r_style`
  (dt = `env.step_dt` = 0.02 s), not `w_style * r_style`. mjlab already multiplies every
  task reward term by dt (`RewardManager.compute(dt=step_dt)`), so an unscaled style
  reward in [0, 1] per step would outweigh the task reward about 50 times and the 0.5 / 0.5
  weights would not mean an even mix. With the scaling both are per-second rates. The
  logged `AMP/mean_style_reward` is the raw [0, 1] value.
- **Discriminator schedule** (not in the spec): after each PPO update, as many
  discriminator steps as PPO took (`num_epochs * num_minibatches`), each on a
  PPO-minibatch-sized batch of clip windows and replay windows. The discriminator's input
  normaliser starts from 100,000 clip windows on a fresh run and is then updated with each
  rollout's windows plus as many clip windows.
- **Replay buffer is not checkpointed** (1,000,000 x 68 floats); it refills after
  `--continue-run`. The discriminator, its normaliser and its optimiser are saved.
- **Style clips:** manifest groups `train,hard`, categories walk, trot, turn,
  turn_in_place, backward, idle (428 Go1 clips, 147,662 frames), mirrored copies kept; clip
  sampling weight duration^0.5 (`amp.motions.clip_weight_alpha`).
- **Features:** joint positions and velocities, base linear and angular velocity in the
  base frame, gravity in the base frame, base height above the environment origin
  (34 numbers per state, 2 states per window). On rough terrain the height is measured
  from the environment origin, not the ground under the robot.
- **`evaluation/curves.py`:** four named AMP panels added to `training_curves.png`
  (discriminator loss, discriminator output on clips and policy, style reward, task
  reward), so the curves the spec asks for are in the dashboard, not only in TensorBoard
  and `progress_all.csv`.
- **Hooks:** `PPOAgent` gained `_on_learn_start`, `_process_rewards`, `_extra_update`,
  `_extra_scalars`, `_extra_state`, `_load_extra_state` (no-ops) and a `checkpoint_kind`
  class attribute used by `load()`. The episode return logged as `Train/mean_return` stays
  the environment's task return, not the mixed reward.
