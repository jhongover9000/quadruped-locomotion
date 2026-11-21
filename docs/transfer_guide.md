# Transfer guide: from the team's robot to sim and back to hardware

The end-to-end path for (A) bringing the team's quadruped into this framework and taking a
trained policy to the real robot, and (B) setting up the reference motions that motion
tracking (SuperMimic) imitates. Each step is a short summary with a link to the detailed
doc; read those for options and procedures.

Not everything on this path exists yet. Every step below is marked **built** or **stub**
(the script exists but exits with "not implemented yet", or the module is only a
docstring and a TODO). In short: import, references,
training and checks in sim are built; system-ID fitting, sim2sim, export and the onboard
control code are not.

## 1. Overview

```
URDF + meshes ──────┐
bench measurements ─┴─> scripts/import_robot.py ─> configs/robots/<name>.yaml  (the one robot file)
                              │                    assets/robots/<name>/       (model, report)
                              └──────────────────> SuperMimic repo: retarget config + RobotSpec
                                                          │
dog mocap (SuperMimic repo data/raw/) ─> build / check / screen / assign groups
                                                          │
                                         data/npz/<name>/*.npz + manifest.csv  (references)
                                                          │
         scripts/train.py --algo ppo          (velocity: flat_velocity, rough_velocity)
         scripts/train.py --algo supermimic   (motion_tracking, uses the references)
                                                          │
         play.py / record.py / replay.py / evaluate.py / plot_training.py
                                                          │
         sim2sim.py [stub] ─> export_policy.py [stub] ─> deploy/ [stub] ─> robot
                                                          │
         data/hardware_logs/ ─> fit_sysid.py / fit_actuator_model.py [stubs] ─> back to the top
```

| Step | Command | Output | Detailed doc | Status |
|---|---|---|---|---|
| Environment | `conda activate mujoco && pip install -e ".[mjlab,dev,video,supermimic]"` (`video`: imageio for `record.py`; `supermimic`: SuperMimic dependencies) | the `mujoco` env | [setup_wsl2.md](setup_wsl2.md) | built |
| Measure | bench procedures | `data/sysid/processed/<name>.yaml` | [measurement_protocol.md](measurement_protocol.md) | procedures; fitting scripts are stubs |
| Import | `scripts/import_robot.py robot.urdf --name <name> ...` | robot YAML, MJCF, `import_report.md`, SuperMimic files | [robot_import.md](robot_import.md) | built |
| References | SuperMimic repo `scripts/data/*.py --robot <name>` | `data/npz/<name>/` | [robot_import.md](robot_import.md#6-supermimic), [supermimic.md](supermimic.md#the-motion-library) | built (in the SuperMimic repo) |
| Inspect references | `scripts/motions.py --robot <name>` | groups, counts, validation | [supermimic.md](supermimic.md#commands) | built |
| Train | `scripts/train.py --algo ppo\|supermimic --robot <name>` | `outputs/<algo>/<task>/<run>/` | [README.md](../README.md#training) | built |
| Check in sim | `record.py`, `evaluate.py`, `plot_training.py`, `replay.py` | videos, CSV, plots | [README.md](../README.md#looking-at-results) | built |
| Cluster | `ROBOT=<name> hpc/submit_pipeline.sh` | `outputs/pipelines/<id>/report/` | [hpc/README.md](../hpc/README.md) | built, Slurm side untested |
| Sim2sim | `scripts/sim2sim.py <run>/` | - | [deployment_checklist.md](deployment_checklist.md) | stub |
| Export | `scripts/export_policy.py <run>/ --name ...` | `policies/released/<release>/` | [policies/released/README.md](../policies/released/README.md) | stub |
| Hardware | `src/quadruped_rl/deploy/` | - | [deployment_checklist.md](deployment_checklist.md) | stub |

## 2. Prerequisites

The `mujoco` conda environment with this repository installed, and an NVIDIA GPU (WSL2 on
Windows): [setup_wsl2.md](setup_wsl2.md#3-install-miniconda), then the checks in
[section 4](setup_wsl2.md#4-verify-from-known-good-to-ours) there. Install with pip, not
conda.

For anything SuperMimic (references, motion tracking), the reference repository
`supermimic-quadruped` must be found, in this order: `backend.root` in
`configs/algorithms/supermimic.yaml`, `$SUPERMIMIC_ROOT`, `../supermimic-quadruped`
(`src/quadruped_rl/algorithms/supermimic/backend.py`). It is not on GitHub; copy it from a
team workstation. Its own docs write commands as `uv run python ...`; with the `mujoco`
environment active, plain `python` is what this repository's docs use.

Extras for later steps are declared in `pyproject.toml` but their code is not written yet:
`trajopt` (Pinocchio + CasADi), `sysid` (CMA-ES), `deploy` (ONNX, onnxruntime).

## 3. Part A1: Bringing the robot in

### URDF

Twelve revolute joints (hip abduction, thigh pitch, calf pitch per leg), root link x
forward / y left / z up, a fixed link at the centre of each foot, `<inertial>` on every
moving link, `<limit>` on every joint, `.stl`/`.obj` meshes, xacro expanded. Any names,
frames, axis signs and zero pose are fine. Full list and what is not supported:
[robot_import.md, section 1](robot_import.md#1-what-the-urdf-must-have).

Keep the CAD export in `assets/robots/custom/urdf/` and `assets/robots/custom/meshes/`
(meshes in Git LFS; see `assets/robots/custom/README.md`).

### Measurements

CAD mass properties miss wiring, fasteners and the battery. Measure total mass, COM, link
masses, inertia and foot friction following [measurement_protocol.md](measurement_protocol.md)
(accuracy targets and formulas), and write the results to
`data/sysid/processed/<name>.yaml`; the keys are listed in
[robot_import.md, section 2](robot_import.md#2-measurements). Raw readings go in
`data/sysid/raw/` (not committed).

- The formulas are implemented as functions in `src/quadruped_rl/sysid/mass_properties.py`
  (`com_from_scales`, `com_height_from_tilt`, `inertia_compound_pendulum`,
  `inertia_bifilar`, `friction_from_slip_angle`, `reflected_inertia`); call them from a
  notebook or a Python shell.
- `scripts/compute_mass_properties.py` (the command-line wrapper) is a **stub**.
- `scripts/fit_sysid.py` (armature, friction, damping from swing tests) is a **stub**.
  Until it exists, put datasheet rotor inertia and gear ratio under `joint:` and leave
  friction to domain randomization.

### Import

```bash
python scripts/import_robot.py assets/robots/custom/urdf/<robot>.urdf --name mydog \
    --kp 20 20 20 --kd 0.5 0.5 0.5 \
    --measurements data/sysid/processed/mydog.yaml
```

`--kp`/`--kd` must be the gains the motor drivers will run. The import refuses to write
anything unless the rebuilt model reproduces the URDF. It writes
`assets/robots/mydog/{mjcf,meshes,urdf}/`, `assets/robots/mydog/import_report.md`,
`configs/robots/mydog.yaml`, and, when the SuperMimic repository is found, the robot's
retarget config and generated `RobotSpec` there plus a `backend.robot_specs` entry in
`configs/algorithms/supermimic.yaml`. Options (`--stance`, `--knees`, `--foot-links`,
`--foot-radius`, `--no-supermimic`, `--force`, ...): [robot_import.md, section 3](robot_import.md#3-import).

`configs/robots/testdog.yaml` is a complete example of an imported config (from the
deliberately awkward URDF in `tests/test_import_robot.py`). `configs/robots/_template.yaml`
is only for a model that is already an MJCF in this project's layout; normally you do not
write a robot YAML by hand.

### The robot YAML

`configs/robots/<name>.yaml` is the single place robot numbers live. The parts that matter
for transfer:

| Key | Why it matters |
|---|---|
| `joint_names` | Policy action and observation order. Never reorder after training. |
| `default_joint_pos`, `base_height_nominal` | Standing pose; actions are offsets from it. |
| `actuator.kp`, `kd`, `torque_limit`, `armature`, `frictionloss`, `viscous_damping` | Simulated servo. Must match the drivers. |
| `domain_randomization` | Ranges for the uncertainty that remains (see [Part A2](#6-part-a2-sim-to-real-transfer)). |
| `hardware.sign`, `hardware.offset` | `q_urdf = offset + sign * q_sim` per joint: what deployment code must apply. |
| `supermimic.err_scale` / `out_scale` / `in_scale` | World-model scales once calibrated (commented out until then). |

After editing it, run `python scripts/import_robot.py --sync mydog` to regenerate the
SuperMimic files. If the URDF or mass measurements change, import again with `--force` and
copy hand edits back ([robot_import.md, section 5](robot_import.md#5-keep-one-description)).

### Checks after the import

1. Read `assets/robots/mydog/import_report.md` in the order given in
   [robot_import.md, section 4](robot_import.md#4-read-the-report): leg labels, joint
   convention (sign/offset), masses, standing test, self-collisions, warnings.
2. Does it stand with no policy? `python scripts/play.py --robot mydog`, or offscreen
   `python scripts/record.py --robot mydog --scenario stand`.
3. A few training iterations:
   `python scripts/train.py --algo ppo --robot mydog --num-envs 64 --max-iterations 5`.
4. `pytest` (config, import and compatibility tests) and `pytest -m sim`.

## 4. Part B: Training references

Reference motions are the clips that motion-tracking training imitates frame by frame.
Only the motion-tracking task uses them; the velocity tasks (PPO) do not.

### Where they come from

| Source | Status |
|---|---|
| Dog motion capture (AI4Animation, Tencent Labrador, CyberDog2 lists), downloaded by the SuperMimic repo's `scripts/data/fetch_sources.py` into its `data/raw/` | **built**; the only source in use |
| `scripts/retarget_motion.py` / `src/quadruped_rl/motions/retarget.py` (retargeting in this repository) | **stub**; retargeting is done by the SuperMimic repository |
| `scripts/generate_trajopt_gait.py` / `src/quadruped_rl/motions/trajopt_gait.py` (trajectory-optimized gaits) | **stub** |

Two of the three sources are restricted (AI4Animation, 354 clips, CC BY-NC; Tencent, 150 clips, research-only); only the 16 CyberDog2 clips are Apache-2.0. The library as a whole therefore cannot be used commercially without a licence check, so clips are not committed; they are
rebuilt or copied ([supermimic.md](supermimic.md#the-motion-library)). `data/motions/raw/`,
`retargeted/` and `trajopt/` in this repository are reserved for this repository's own
clips (see `data/README.md`) and are empty; no task reads them unless you point
`motions.dir` there.

### Building a library for the team's robot

The Go1 library (520 clips, `data/npz/go1/` in the SuperMimic repo) is retargeted to the
Go1's legs and cannot be reused for another robot. After the import, build one in the
SuperMimic repository:

```bash
cd ../supermimic-quadruped
python -m pytest tests/test_quadmotion_robot.py -q                  # kinematics of every robot config
python scripts/data/build_dataset.py --robot mydog                  # retarget, cut, mirror -> data/npz/mydog/
python scripts/data/check_library.py --robot mydog                  # limits, feet below ground, foot sliding
python scripts/data/screen_clips.py --robot mydog --stiffness 76    # open-loop: which clips are followable
python scripts/data/assign_groups.py --robot mydog                  # train / hard / holdout in manifest.csv
python scripts/calibrate_scales.py --robot mydog --npz-dir data/npz/mydog
```

Details and how to turn `calibrate_scales.py`'s output into `supermimic:` scales:
[robot_import.md, section 6](robot_import.md#6-supermimic). Notes:

- Retargeting needs the thigh axis to cross the hip axis; if not, the import report says
  "Not usable" and no SuperMimic files are written (PPO still works).
- `screen_clips.py` deliberately uses a stiffer servo than the robot's (its docstring
  explains why); 76 is the cluster default (`SM_STIFFNESS` in `hpc/config.sh`) used for the
  Go1. `import_robot.py --sync` prints the same steps with `--stiffness <kp>` left for you.
- `scripts/data/render_preview.py --robot mydog --sampler 3` (SuperMimic repo, needs
  `MUJOCO_GL=egl` headless) renders kinematic previews of the retargeted clips.
- On the cluster, stage `motions` (`hpc/30_supermimic_motions.slurm`) runs the first four
  library steps for `$ROBOT` if the library is missing ([hpc/README.md](../hpc/README.md)).

### Clip format, manifest and groups

One `.npz` per clip at 50 Hz (the control rate): `fps`, `joint_names`, `body_names`,
`joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w` (w, x, y, z), `body_lin_vel_w`,
`body_ang_vel_w`. The root pose is the root body's row. Spec and validation:
`src/quadruped_rl/motions/format.py` (numpy only).

A library is a directory of clips plus `manifest.csv` (`name`, `keep`, `group`, and
columns such as `category`, `source`, `mirror_of`, `replay_return`, `weight`). Groups:
`train`, `hard` (screen found them difficult; still trained on) and `holdout` (never
trained on, whole takes held out so nothing leaks). Without a manifest, every clip is one
group `all` (`src/quadruped_rl/motions/motion_lib.py`).

### How the task config points at them

`configs/tasks/motion_tracking.yaml`, section `motions:`:

```yaml
dir: data/npz/{robot}                       # {robot} = the robot's SuperMimic spec name
manifest: data/npz/{robot}/manifest.csv
train_groups: train,hard                    # trained on
eval_groups: holdout                        # played by play.py / record.py / evaluate.py
exclude: null                               # regex on clip names
clips: null                                 # or an explicit list
```

plus `future_frames`, `sampling` (mode, clip weighting) and `terminations`. Relative paths
are tried in this repository first, then in the SuperMimic repository
(`backend.resolve_path`); the spec name comes from `backend.robot_specs` in
`configs/algorithms/supermimic.yaml`. `backend.build_train_config` copies these into the
reference `TrainConfig` (`data.npz_dir`, `data.manifest`, `data.group`, ...); the
observations and tracking reward come from the robot's `RobotSpec`, not from this file. Any
entry can be overridden per run, for example `--set motions.limit=50`.

### Inspecting and validating

```bash
python scripts/motions.py --robot mydog                     # library path, groups, which are trained / evaluated
python scripts/motions.py --robot mydog --group holdout     # clips of a group: duration, mean speed
python scripts/motions.py --robot mydog --check 50          # validate files and joint order vs the robot YAML
python scripts/motions.py --robot mydog --clip <name>       # one clip
```

`--check` fails if a clip's joint order differs from `joint_names` in
`configs/robots/mydog.yaml`. To watch a policy follow a clip, with the reference drawn as a
blue ghost: `python scripts/record.py <supermimic run>/ --clips <name>`. This repository has
no kinematic clip viewer without a trained policy; use the SuperMimic repo's
`render_preview.py` for that.

## 5. Part C: Training and checking in sim

### Train

```bash
python scripts/train.py --list                                          # algorithm x task table
python scripts/train.py --algo ppo --robot mydog                        # flat_velocity
python scripts/train.py --algo ppo --robot mydog --task rough_velocity
python scripts/train.py --algo supermimic --robot mydog                 # motion_tracking
python scripts/train.py --algo supermimic --robot mydog --num-envs 1024 --max-iterations 3000   # short run
python scripts/train.py --continue-run outputs/<algo>/<task>/<run>/     # after an interruption
```

Overrides: `--set` for task entries (`rewards.action_rate=-0.02`), `--algo-set` for
algorithm entries (`policy_trainer.horizon=16`). Runs go to
`outputs/<algo>/<task>/<timestamp>_<robot>_s<seed>/` with the resolved `config.yaml`.
Expect to retune PD gains and reward scales for a new robot; sizes SuperMimic tunes per
robot are scaled from the Go1 by standing height, not tuned
([robot_import.md, section 6](robot_import.md#6-supermimic)). Only PPO (`ppo`, `ppo_rslrl`)
and SuperMimic are implemented; DeepMimic, AMP and MPC refuse to train.

### Check

All take a run directory and work for PPO and SuperMimic runs alike:

```bash
python scripts/plot_training.py <run>/                 # training dashboard; several runs side by side
python scripts/record.py <run>/                        # video, plots.png, summary.txt, replay files
python scripts/replay.py <run>/recordings/<recording>/ # offline: no GPU, no policy
python scripts/evaluate.py <run>/                      # configs/eval/default.yaml -> one CSV row
python scripts/play.py <run>/                          # interactive viewer
```

How to read a SuperMimic run (normalized return, world-model error, checkpoints):
[supermimic.md](supermimic.md#reading-a-supermimic-run). `evaluate.py` does not yet cover
`terrain_levels`, `max_speed_test` or `perturbations` from the eval config.

### On the cluster

After the import, copy both repositories to scratch (the robot's files live in both), then
`ROBOT=mydog hpc/submit_pipeline.sh` (try `SMOKE=1` first). Stages, sizing and resuming:
[hpc/README.md](../hpc/README.md), [robot_import.md, section 7](robot_import.md#7-on-the-cluster).

## 6. Part A2: Sim-to-real transfer

### Domain randomization

Ranges live in the robot YAML under `domain_randomization:` and are applied by
`src/quadruped_rl/envs/domain_randomization.py`; a key left out is not randomized. The
import fills them from the mass and foot friction:

| Key | What |
|---|---|
| `friction` | foot friction range (0.6x to 1.4x the measured `foot_friction`) |
| `added_base_mass`, `base_com_offset` | trunk mass and COM uncertainty |
| `kp_scale`, `kd_scale`, `motor_strength` | servo gain and torque-limit spread |
| `frictionloss_scale`, `armature_scale` | `[lo, hi]` factors on the joint dry friction (`actuator.frictionloss`) and reflected inertia (`actuator.armature`), one uniform draw per env and per joint (mjlab `dr.joint_friction`, `dr.joint_armature`) |
| `encoder_bias` | constant per-joint offset on the actor's joint positions |
| `action_delay_steps` | command latency, control steps |
| `push_interval_s`, `push_velocity` | random pushes |

Observation noise is per task, under `observations.noise` in
`configs/tasks/flat_velocity.yaml`. Set ranges from what the measurements leave uncertain
([measurement_protocol.md](measurement_protocol.md#accuracy-targets)).

For SuperMimic, `--sync` copies `friction`, `base_com_offset` and `encoder_bias` into the
generated `RobotSpec`; action delay is not ported (`action_delay_substeps=(0, 0)` in
`src/quadruped_rl/robots/supermimic_port.py`). Its own randomization switch is
`env.domain_randomization` in `configs/algorithms/supermimic.yaml`.

### Actuator model

Training simulates one of two PD servos (`src/quadruped_rl/envs/builder.py`), both with
the YAML's `kp`, `kd`, `torque_limit`, `armature`, friction and damping:

- `model: ideal_pd`: MuJoCo's built-in position actuator, torque clipped at `torque_limit`.
- `model: dc_motor`: the same PD law computed in torch every physics step (mjlab
  `DcMotorActuator`), clipped to a DC motor's torque-speed line. With `actuator.params`
  `saturation_effort` (N m, voltage-limited stall torque at the joint) and `no_load_speed`
  (rad/s at the joint), the torque available in the direction of motion at joint speed w
  is `min(torque_limit, saturation_effort * (1 - w / no_load_speed))`, the mirror image
  for the other direction. Above `no_load_speed` mjlab's line goes negative: back-EMF then
  forces a braking torque (down to `-torque_limit`). Both params take a scalar, a
  `[hip, thigh, calf]` triple or 12 values. The PD damping is explicit here (the built-in
  actuator's is implicit), so very small armature with large `kd` can be stiff. Note:
  mjlab keeps the MuJoCo `ctrlrange` at `torque_limit`, so with `dc_motor` a
  `motor_strength` factor above 1 is capped at `torque_limit`.

`actuator.velocity_limit` is not applied by either. `analytic` and `actuator_net`
(`src/quadruped_rl/actuators/`) are **stubs**, are only used outside the GPU simulator,
and do not change training (`make_actuator` there does not know `dc_motor`).
`scripts/fit_actuator_model.py` is a **stub**. Until measurements exist, the actuator gap
is covered by `kp_scale`, `kd_scale`, `motor_strength`, `frictionloss_scale`,
`armature_scale` and `action_delay_steps`.

### Sim2sim

`scripts/sim2sim.py` (run the policy in plain CPU MuJoCo) and the CPU backend it needs
(`src/quadruped_rl/envs/backends/mujoco_cpu.py`) are **stubs**. The only plain-MuJoCo check
today is the import's standing test.

### Export

`scripts/export_policy.py` and `src/quadruped_rl/deploy/export.py` are **stubs**. When
written, a release goes to `policies/released/<algo>_<task>_<robot>_v<N>/` with
`policy.onnx`, `config.yaml` and `notes.md` (the `.onnx` and `.pt` files are Git LFS; see
[policies/released/README.md](../policies/released/README.md)). What an exported PPO
policy must carry, from the current training code:

- **Observation normalization.** PPO's actor normalizes inputs with running statistics
  (`EmpiricalNormalization` in `src/quadruped_rl/algorithms/ppo/networks.py`), stored in
  the checkpoint (`<run>/checkpoints/model_<N>.pt`). Bake it into the exported graph.
- **Observation layout** (`observations.actor` in the task YAML, terms in
  `src/quadruped_rl/envs/observations.py`): `base_ang_vel`, `projected_gravity`,
  `velocity_command`, `joint_pos_rel` (q minus the standing pose), `joint_vel`,
  `last_action`, all in the body frame, joints in `joint_names` order.
- **Action**: `q_target = default_joint_pos + scale_j * action`, with
  `scale_j = scale * torque_limit_j / kp_j` for `scale_mode: effort_relative`
  (`action:` in `configs/tasks/flat_velocity.yaml`).
- **Rate**: 50 Hz (`physics_dt: 0.005`, `decimation: 4`).
- **Joint convention**: the policy works in this project's convention; convert to motor
  angles with `hardware.sign` / `hardware.offset` from the robot YAML.

A SuperMimic policy's observations (defined by its `RobotSpec`) include `future_frames`
reference frames, so running one on the robot also needs the reference clips onboard.
Nothing for this exists yet.

### Onboard code and the deployment checklist

`src/quadruped_rl/deploy/obs_builder.py` (sensor readings to observation),
`policy_runner.py` (50 Hz loop) and `safety.py` (clamps, tilt cutoff, watchdog, e-stop)
are **stubs** that describe what each must do. There is no hardware driver interface in
this repository.

Before the robot moves, go through [deployment_checklist.md](deployment_checklist.md)
(interface checks, then staged testing: hanging, standing, gantry, free walking). The
single most important check is the joint sign/offset on the bench, described in
[robot_import.md, section 4](robot_import.md#4-read-the-report).

### Closing the loop

Log every hardware run to `data/hardware_logs/` (not committed; no logging code exists
yet) and motor and swing-test logs to `data/sysid/raw/`. Then:

1. Compare a sim rollout and a hardware rollout observation by observation (checklist).
2. Update `data/sysid/processed/<name>.yaml` and re-import with `--force`, or edit
   `configs/robots/<name>.yaml` (gains, friction, randomization ranges) and `--sync`.
3. Retrain. Changing the standing pose, gains, action scale or joint order invalidates
   trained policies ([robot_import.md, section 5](robot_import.md#5-keep-one-description)).

Once `fit_sysid.py` and `fit_actuator_model.py` exist, they are the step between 1 and 2.

## 7. Troubleshooting

- **`Robot 'mydog' has no SuperMimic RobotSpec`.** The import did not write the SuperMimic
  side (repository not found, `--no-supermimic`, or unsupported leg geometry; see the
  import report). Fix the cause and run `python scripts/import_robot.py --sync mydog`.
- **`The SuperMimic reference implementation was not found`.** Put it at
  `../supermimic-quadruped` or set `SUPERMIMIC_ROOT` ([section 2](#2-prerequisites)).
- **`No motion library at .../data/npz/mydog`** (from `scripts/motions.py`). The library
  has not been built for this robot ([Part B](#building-a-library-for-the-teams-robot)),
  or `motions.dir` points elsewhere.
- **Joint order errors from `motions.py --check`.** The clips were built for another model
  or joint order. Rebuild the library after any re-import.
- **`base_height_nominal ... does not match the standing pose's height`** on `--sync`. The
  YAML's standing pose and height disagree after a hand edit; fix one of them.
- **Robot falls in the import's standing test, or sags.** Gains too soft or torque limits
  too low for the weight ([robot_import.md, section 4](robot_import.md#4-read-the-report)).
- **Policies trained before a change to pose, gains, action scale or joint order** no
  longer match the robot; retrain.
- **Go1 only: default-pose warning when training SuperMimic.** The Go1 YAML and its
  SuperMimic spec disagree on the hip angle; imported robots generate the spec from the
  YAML and do not have this ([supermimic.md](supermimic.md#known-differences-and-limits)).
- **Training stalls at the first SuperMimic compile.** See
  [supermimic.md](supermimic.md#known-differences-and-limits) and
  `--stall-timeout`; general hangs and WSL issues:
  [setup_wsl2.md](setup_wsl2.md#troubleshooting).
