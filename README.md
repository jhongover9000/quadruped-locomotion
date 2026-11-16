# Quadruped Locomotion: Comparing RL Methods from Sim to Real

Train and compare locomotion policies for a custom quadruped. Development starts on a
stand-in robot (Unitree Go1 from MuJoCo Menagerie); the team's robot is swapped in through
a single config file once its URDF is ready, then policies move to hardware.

Goals: rough-terrain traversal and basic locomotion, with **speed and smoothness**
(low jerk) measured explicitly rather than judged by eye.

## Methods compared

Each method is a choice of *objective* (what is rewarded) and *optimizer* (how it learns):

| Optimizer | Task reward (velocity) | Imitation reward (track a clip) | Style reward (match motion distribution) |
|---|---|---|---|
| PPO (model-free) | Baseline | DeepMimic | AMP |
| World model + BPTT | Extension | SuperMimic-style | - |
| None (control) | Convex MPC | - | - |

## Repository layout

```
configs/                 everything tunable, as YAML
  robots/                one file per robot (go1.yaml, _template.yaml for the team's robot)
  tasks/                 flat_velocity, rough_velocity (inherits flat), motion_tracking
  algorithms/            ppo, deepmimic, amp, supermimic, mpc
  eval/                  the shared evaluation protocol
  sweeps/                grids for Pareto fronts (smoothness penalties, body height)
src/quadruped_rl/
  robots/                robot config loading and validation
  envs/                  batched environment: observations, rewards, commands, terrain,
                         domain randomization, terminations
    backends/            physics backends (CPU MuJoCo, mjlab, MJX)
    tasks/               velocity tracking, motion tracking
  actuators/             ideal PD -> analytic model -> actuator network
  algorithms/            ppo/, deepmimic/, amp/, supermimic/, mpc_baseline/ (one interface)
  motions/               reference-motion format, retargeting, trajectory optimization
  sysid/                 mass-property formulas, swing tests, parameter fitting
  evaluation/            metrics, evaluator, Pareto fronts, plots
  deploy/                policy export, hardware observation builder, control loop, safety
  utils/                 paths, config loading, seeding, logging
scripts/                 entry points (train, evaluate, sweep, sim2sim, export, sysID, ...)
assets/robots/           stand_in/ (downloaded, not committed), custom/ (team URDF, meshes, MJCF)
data/                    reference motions and system-ID recordings (raw data not committed)
policies/released/       policies approved for hardware (Git LFS)
docs/                    measurement protocol, deployment checklist
tests/                   config, metric, and formula tests
notebooks/               exploration and analysis
outputs/                 training runs (created automatically, not committed)
```

## Setup

Training runs on an NVIDIA GPU under Linux; on Windows, use WSL2 + Ubuntu.
**Full walkthrough: [docs/setup_wsl2.md](docs/setup_wsl2.md).** In short:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mjlab,dev]"                    # PyTorch (CUDA), MuJoCo Warp, mjlab
git lfs install                                  # once per machine
pytest                                           # fast checks, incl. PPO on a toy task
pytest -m sim                                    # builds the simulated Go1
```

The stand-in Go1 model ships with mjlab, so nothing needs downloading.

## Training

```bash
python scripts/train.py --algo ppo                       # flat ground, 4096 envs
python scripts/train.py --algo ppo --task rough_velocity
python scripts/train.py --algo ppo --set rewards.action_rate=-0.02    # override any task entry
python scripts/train.py --algo ppo_rslrl                 # reference PPO (rsl_rl), same env
python scripts/benchmark.py                              # steps/s, GPU memory, run/sweep times
tensorboard --logdir outputs                             # learning curves
python scripts/play.py outputs/ppo/flat_velocity/<run>/  # watch the policy
python scripts/play.py --robot go1                       # no policy: does the robot stand?
```

`ppo_rslrl` runs rsl_rl's PPO (the trainer mjlab itself uses) on exactly our environment.
If it learns and `ppo` doesn't, the bug is in our PPO; if neither does, look at the
environment or rewards.

## Workflow

| Phase | Goal | Main files |
|---|---|---|
| 1. Infrastructure | PPO walks on flat ground; metrics logged (**built**) | `envs/`, `algorithms/ppo/`, `scripts/train.py` |
| 2. Terrain | Terrain curriculum and domain randomization (**built**), sim2sim | `envs/terrain.py`, `envs/domain_randomization.py`, `scripts/sim2sim.py` |
| 3. Reference motions | Trot/bound clips from trajectory optimization or retargeted mocap | `motions/`, `scripts/generate_trajopt_gait.py`, `scripts/retarget_motion.py` |
| 4. Imitation methods | DeepMimic, AMP, SuperMimic-style | `algorithms/deepmimic/`, `algorithms/amp/`, `algorithms/supermimic/` |
| 5. Comparison | Same evaluation for all; speed-vs-jerk Pareto fronts | `evaluation/`, `configs/eval/`, `configs/sweeps/`, `scripts/sweep.py` |
| 6. Team robot | Measure, system ID, actuator model, retrain, hardware | `sysid/`, `actuators/`, `deploy/`, `docs/` |

## Swapping in the team's robot

1. Put the URDF and meshes in `assets/robots/custom/urdf/` and `assets/robots/custom/meshes/`.
2. Run `python scripts/convert_urdf_to_mjcf.py ...` and work through the checklist it prints.
   Check it with `python scripts/play.py --robot <name>`: it should stand under PD control.
3. Copy `configs/robots/_template.yaml` to `configs/robots/<name>.yaml` and fill it in
   (see `docs/measurement_protocol.md`).
4. Train with `--robot <name>`. Expect to retune PD gains and reward scales, regenerate
   reference motions, and retrain every method.

## Large files: where things go

| What | Where | In git? |
|---|---|---|
| Training runs, checkpoints, TensorBoard/W&B logs, videos | `outputs/` | No (ignored) |
| Checkpoints / `.onnx` / `.pt` anywhere else | - | No (ignored) |
| Policies approved for hardware | `policies/released/` | Yes, Git LFS |
| Robot meshes (`.stl`, `.obj`, `.dae`), STEP exports | `assets/robots/custom/meshes/` | Yes, Git LFS |
| Native CAD (SolidWorks, Fusion, Inventor) | your CAD tool | No (ignored) |
| Stand-in robot model | bundled with mjlab | - |
| Raw mocap, motor logs, hardware recordings | `data/*/raw/`, `data/hardware_logs/` | No (share via team drive) |
| Processed reference motions (small `.npz`) | `data/motions/retargeted/`, `data/motions/trajopt/` | Yes |

Each run saves its fully resolved `config.yaml` next to its checkpoints, so results stay
reproducible even though the checkpoints themselves are not committed.
