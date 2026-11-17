# Spec: convex MPC baseline (`--algo mpc`)

Task P5.3 in `docs/orchestration.md`. Written 2026-11-16 against the working tree after
commit `c0abbe8`. Independent of DeepMimic and AMP.

## Goal

A model-based controller with no learning, to stand next to the learned policies in the
same evaluation: it takes the same velocity commands in the same simulated environment
and is scored by the same `scripts/evaluate.py`. The method is the convex MPC of
Di Carlo et al. (2018): the body is one rigid mass, the foot forces over a short horizon
are the unknowns of a quadratic programme with friction-cone limits, the stance legs
turn those forces into joint torques, and the swing legs follow foot trajectories to
footholds chosen by a simple heuristic.

## Decisions already taken

1. **No new packages.** The `mujoco` conda env has no QP solver (no osqp, qpsolvers,
   quadprog, cvxpy, casadi, proxsuite or clarabel), and Joseph wants to be asked before
   packages change. The problem is small (horizon 10, 12 force components per step) and
   has to be solved for every parallel environment each control step, so write a batched
   solver in torch; an ADMM solver of the kind OSQP uses fits. If that proves inadequate,
   stop and report the options with measurements instead of installing something.
2. **The controller core does not import the environment.** It maps a state (base pose
   and velocity, joint positions and velocities, foot positions, contact schedule,
   command) to joint-level outputs. A thin adapter reads that state from the mjlab
   environment. Phase 3 (sim2sim on plain CPU MuJoCo) will reuse the core with a
   different adapter.
3. **Torques are delivered through the position-target action.** The environment has no
   torque action: `make_action` (`src/quadruped_rl/envs/builder.py:98`) sets
   `q_target = q_default + scale * a`, and the actuator applies a PD law with the robot's
   `kp` and `kd`. The adapter inverts that law to obtain the wanted torque at the start
   of each control step. The control step is 0.02 s (4 physics steps), so the applied
   torque drifts within a step; measure how much this matters and report it.
4. **Nothing robot-specific is hard-coded.** Leg geometry, mass, inertia and foot
   Jacobians come from the MuJoCo model and `RobotConfig`
   (`src/quadruped_rl/robots/config.py:32`: `foot_sites`, `joint_names`,
   `default_joint_pos`, `actuator.kp/kd/torque_limit`), so the controller also runs on
   the team's robot once it is imported. It may assume four legs of three joints each.

## What exists

- Stubs with the design intent in their docstrings:
  `src/quadruped_rl/algorithms/mpc_baseline/{controller,centroidal_model}.py`. The
  registry already maps `mpc` to `mpc_baseline.controller:ConvexMPC`.
- Config: `configs/algorithms/mpc.yaml` (horizon 10, dt 0.03, trot with period 0.5 s and
  duty 0.5, friction 0.6, cost weights). It has no `max_iterations` or `num_envs`.
- Contract: `src/quadruped_rl/algorithms/base.py`. `make_policy(env, checkpoint)` returns
  `(viewer_env, policy)` where `policy(viewer_env.get_observations())` gives actions.
  The MPC policy needs simulator state that is not in the observation, so its closure
  reads the environment directly.
- State: `env.scene["robot"].data` (root pose and velocity, joints, `site_pos_w`,
  `site_lin_vel_w`). The `feet_contact` sensor exists only when a reward or observation
  term asks for it (`builder.py:131`, `226`).
- Evaluation: `src/quadruped_rl/evaluation/rollout.py` `load_run` (line 114) requires
  `config.yaml` and a checkpoint found through `checkpoints.ALL_PATTERNS`, and raises
  otherwise. Velocity rollouts use the command term `twist`, including its
  `is_standing_env` flag.
- `scripts/train.py` reads `max_iterations` from the config (line 164) and marks a run
  finished from it (line 62).

## Requirements

- `python scripts/train.py --algo mpc` creates a run directory without training. It is
  the "build the controller and record its settings" step, and it finishes in seconds.
- `scripts/play.py`, `scripts/record.py` and `scripts/evaluate.py` then work on that run
  directory with no special flags, and `evaluate.py` scores it in the same table as PPO
  runs. Choose the cleanest way to satisfy `load_run`'s need for a checkpoint and say
  what you chose.
- The robot stands still when the command is zero, and trots otherwise.
- Torque limits from `RobotConfig` are respected in the QP or by clipping, and the
  report says which.
- `tests/test_compatibility.py` is updated. New CPU tests cover: the QP solver against a
  problem with a known solution, including active constraints; the single-rigid-body
  model, by checking that a robot standing still gets foot forces summing to its weight;
  and the torque-to-position-target inversion.

## Acceptance checks

```
pytest -q
pytest -m sim -q
python scripts/train.py --list
python scripts/train.py --algo mpc --run-dir outputs/debug/p5/mpc_smoke
python scripts/record.py outputs/debug/p5/mpc_smoke
python scripts/evaluate.py outputs/debug/p5/mpc_smoke outputs/ppo/flat_velocity/20261114_220614_go1_s0
```

Targets on `flat_velocity` with the Go1 and 16 environments, over 10-second episodes:
no falls when standing, at 0.5 m/s forward, at 1.0 m/s forward, at 0.3 m/s sideways and
at 0.5 rad/s of yaw, with mean velocity error under 0.15 m/s (0.15 rad/s for yaw). Also
report the control-loop time per step for 16 environments. These are targets to report
against. If one is missed after a reasonable effort, report the measured value and the
cause you suspect; do not loosen the check or special-case the command.

## Out of scope

Rough terrain; gaits other than standing and trotting; the plain CPU MuJoCo backend
(`envs/backends/mujoco_cpu.py`, phase 3); the `hpc/` pipeline and the README method
table (P5.4); committing.

## Deviations

Record here anything you had to do differently from this spec and why. If you hit a
choice the spec does not cover, take the conservative option, note it here and continue.

### P5.3 implementation notes (2026-11-16, worktree agent-a2fa4d9835e8da89e)

- **Checkpoint for `load_run`:** `train()` writes `checkpoints/model_0.pt`, which holds the
  controller settings, the leg kinematics and the rigid-body parameters read from the
  MuJoCo model (signature `mpc` in `checkpoints._SIGNATURES`). `make_policy` rebuilds the
  controller from that file, so play/record/evaluate need no special flags. `train.py`
  is unchanged: with no `max_iterations` the target is 0 and `status.json` says finished.
  The run also prints a harmless "could not plot training curves" line (no TensorBoard
  events).
- **Torque limits:** enforced by clipping, not in the QP. Stance torques are clipped to
  `actuator.torque_limit`, then the position target is clipped so the PD torque at the
  start of the control step stays inside the limit. The QP only bounds each normal force
  (`max_foot_force_bw`).
- **Config changes beyond the original YAML:** `friction_coef` 0.6 -> 0.3 (the lowest foot
  friction in the Go1 domain randomization; with 0.6, low-friction robots slipped and fell:
  15 falls at 0.5 m/s with 16 envs, against 1 with 0.3). Position weights on x, y 0 -> 5,
  plus a target pose (commanded velocity and yaw rate integrated from where the robot
  was, capped at `max_position_error` 0.1 m / `max_yaw_error` 0.2 rad). Without position
  feedback, the randomized added mass and CoM offset made standing robots creep about
  1.5 cm/s relative to their feet until they fell (6/16 fell within 10 s).
- **The stance PD target leads by half a control step** (`q_des = q + 0.5 dt qd`) so the
  torque averaged over the 4 physics steps is close to the wanted torque. Measured drift:
  the RMS change in PD torque over one control step is 2.6-4.7 N m, against an RMS torque
  of 5.5-8 N m while trotting (0 when standing). It is large; the lead term only
  compensates its mean.
- **Measured targets so far** (CPU, `CUDA_VISIBLE_DEVICES=""`, flat_velocity, Go1, 16 envs,
  play-mode domain randomization on, 10 s; error = |mean velocity after 2 s - command|):
  stand 0 falls, 0.000 m/s; 0.5 m/s 1 fall, 0.012; 1.0 m/s 3 falls, 0.052; 0.3 m/s
  sideways 2 falls, 0.013; 0.5 rad/s yaw 0 falls, 0.009 rad/s. The velocity targets are
  met. The no-falls target is missed by 1-3 falls per 10 s at speed, all on the
  robots with the lowest friction (0.33 at seed 0) combined with the other randomization
  terms. With randomization off, 0.5 m/s gives 0 falls. Control loop for 16 envs on the
  CPU: about 20 ms per step (4 torch threads; it rises to 90 ms with 16 threads
  oversubscribed next to the CPU simulator). Still to measure: the GPU numbers.
- **`evaluate.py` on the CPU** (`--seeds 1 --num-envs 16`, default command set, 8 s each),
  MPC against the PPO run `20261114_220614_go1_s0` (copied into the worktree's outputs so
  nothing is written into the main checkout): MPC 49 falls, rmse xy 0.207, wz 0.409; PPO 1
  fall, 0.112 / 0.084. MPC per command: 0.5 m/s 1 fall, 1.0 m/s 0, 1.5 m/s 43 (beyond the
  spec's targets; the 0.5 s trot with 0.15 m foothold cap cannot keep up), 1.0 rad/s yaw 5,
  combined 0.5/0.3/0.5 0.
