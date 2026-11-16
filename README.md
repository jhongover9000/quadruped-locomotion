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

Built so far: **PPO** (ours, plus rsl_rl's as a reference) on the velocity tasks, and
**SuperMimic** on motion tracking. DeepMimic, AMP and MPC are planned; `train.py` refuses
them rather than training something else under their name. `python scripts/train.py --list`
prints which algorithm runs on which task.

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
  motions/               reference-motion clip format and library, retargeting, trajectory optimization
  sysid/                 mass-property formulas, swing tests, parameter fitting
  evaluation/            metrics, rollout recording, rendering, training curves, plots
  deploy/                policy export, hardware observation builder, control loop, safety
  utils/                 paths, config loading, seeding, logging, stall watchdog
scripts/                 entry points (train, play, record, replay, evaluate, plot_training, ...)
hpc/                     Slurm pipeline for the cluster (see hpc/README.md)
assets/robots/           stand_in/ (downloaded, not committed), custom/ (team URDF, meshes, MJCF)
data/                    reference motions and system-ID recordings (raw data not committed)
policies/released/       policies approved for hardware (Git LFS)
docs/                    setup, SuperMimic guide, measurement protocol, deployment checklist
tests/                   config, metric, and formula tests
notebooks/               exploration and analysis
outputs/                 training runs (created automatically, not committed)
```

## Setup

Training runs on an NVIDIA GPU under Linux; on Windows, use WSL2 + Ubuntu.
**Full walkthrough: [docs/setup_wsl2.md](docs/setup_wsl2.md).** In short:

```bash
conda create -n mujoco python=3.12 -y && conda activate mujoco
pip install -e ".[mjlab,dev]"                    # PyTorch (CUDA), MuJoCo Warp, mjlab
git lfs install                                  # once per machine
pytest                                           # fast checks, incl. PPO on a toy task
pytest -m sim                                    # builds the simulated Go1
```

The stand-in Go1 model ships with mjlab, so nothing needs downloading.

## Training

```bash
python scripts/train.py --list                           # which algorithm runs on which task
python scripts/train.py --algo ppo                       # flat ground, 4096 envs
python scripts/train.py --algo ppo --task rough_velocity
python scripts/train.py --algo ppo --set rewards.action_rate=-0.02    # override any task entry
python scripts/train.py --algo ppo_rslrl                 # reference PPO (rsl_rl), same env
python scripts/train.py --algo supermimic                # motion tracking (docs/supermimic.md)
python scripts/train.py --continue-run <run>/            # pick up an interrupted run
python scripts/benchmark.py                              # steps/s, GPU memory, run/sweep times
```

`ppo_rslrl` runs rsl_rl's PPO (the trainer mjlab itself uses) on exactly our environment.
If it learns and `ppo` doesn't, the bug is in our PPO; if neither does, look at the
environment or rewards.

Every run ends with `training_curves.png` and `progress_all.csv` in its directory. A run
that stops making progress prints the stack of every thread after 5 minutes, so a hang can
be told from a slow step (`kill -USR1 <pid>` prints it on demand; `--stall-timeout 900`
restarts instead of waiting).

## Looking at results

All of these take a run directory and work for every implemented algorithm:

```bash
tensorboard --logdir outputs                             # live curves
python scripts/plot_training.py <run>/                   # dashboard of everything logged
python scripts/plot_training.py <run A>/ <run B>/        # runs side by side, same axes
python scripts/play.py <run>/                            # watch the policy
python scripts/play.py <run>/ --command 1.0 0 0          # velocity runs: every robot forward
python scripts/play.py --robot go1                       # no policy: does the robot stand?
python scripts/record.py <run>/                          # video, plots, summary
python scripts/replay.py <run>/recordings/<recording>/   # rewatch offline: no GPU, no policy
python scripts/evaluate.py <run>/ ...                    # shared protocol -> one CSV table
```

Trainers count iterations differently (PPO: 24 steps in every environment; SuperMimic:
one), so plots use simulated environment steps on the x-axis.

## On the cluster

`hpc/` chains Slurm jobs: environment check, PPO, motion library, SuperMimic (in resumable
segments), evaluation and a report folder. See [hpc/README.md](hpc/README.md).

```bash
bash hpc/00_setup.sh                 # once, on the login node
SMOKE=1 hpc/submit_pipeline.sh       # dry run of every stage with tiny settings
hpc/submit_pipeline.sh               # the real thing
```

## Workflow

| Phase | Goal | Main files |
|---|---|---|
| 1. Infrastructure | PPO walks on flat ground; metrics logged (**built**) | `envs/`, `algorithms/ppo/`, `scripts/train.py` |
| 2. Terrain | Terrain curriculum and domain randomization (**built**), sim2sim | `envs/terrain.py`, `envs/domain_randomization.py`, `scripts/sim2sim.py` |
| 3. Reference motions | Retargeted mocap library (**built** with the SuperMimic repository's tools); trajectory-optimized gaits | `motions/`, `scripts/motions.py`, `scripts/generate_trajopt_gait.py` |
| 4. Imitation methods | SuperMimic (**built**, via the reference implementation); DeepMimic, AMP | `algorithms/supermimic/`, `algorithms/deepmimic/`, `algorithms/amp/` |
| 5. Comparison | Same evaluation for all (**started**: `evaluate.py`, `plot_training.py`); speed-vs-jerk Pareto fronts | `evaluation/`, `configs/eval/`, `configs/sweeps/`, `scripts/sweep.py` |
| 6. Team robot | Measure, system ID, actuator model, retrain, hardware | `sysid/`, `actuators/`, `deploy/`, `docs/` |

## Swapping in the team's robot

One command converts the robot's URDF (any names, joint frames, axis directions and zero
pose) into the model and config every script here trains on, applies bench measurements
(mass, centre of mass, joint friction ...) and writes the SuperMimic side too:

```bash
python scripts/import_robot.py path/to/robot.urdf --name mydog --kp 20 20 20 --kd 0.5 0.5 0.5 \
    --measurements data/sysid/processed/mydog.yaml
python scripts/train.py --algo ppo --robot mydog
```

[docs/robot_import.md](docs/robot_import.md) is the guide: what the URDF needs, the
measurements file, what to check in the import report, and the SuperMimic steps.
Expect to retune PD gains and reward scales, rebuild the reference motions, and retrain
every method.

## Large files: where things go

| What | Where | In git? |
|---|---|---|
| Training runs, checkpoints, TensorBoard/W&B logs, videos | `outputs/` | No (ignored) |
| Checkpoints / `.onnx` / `.pt` anywhere else | - | No (ignored) |
| Policies approved for hardware | `policies/released/` | Yes, Git LFS |
| Robot model, meshes (`.stl`, `.obj`), source URDF | `assets/robots/<name>/` (written by `scripts/import_robot.py`) | Yes, meshes in Git LFS |
| Native CAD (SolidWorks, Fusion, Inventor) | your CAD tool | No (ignored) |
| Stand-in robot model | bundled with mjlab | - |
| Raw mocap, motor logs, hardware recordings | `data/*/raw/`, `data/hardware_logs/` | No (share via team drive) |
| Processed reference motions (small `.npz`) | `data/motions/retargeted/`, `data/motions/trajopt/` | Yes |
| Retargeted mocap library for SuperMimic (non-commercial licences) | SuperMimic repository, `data/npz/<robot>/` | No (rebuild or copy) |

Each run saves its fully resolved `config.yaml` next to its checkpoints, so results stay
reproducible even though the checkpoints themselves are not committed.
