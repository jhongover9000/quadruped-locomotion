# Bringing your own quadruped in

How to go from the robot on the bench (a URDF from CAD plus measurements) to a robot that
every script here trains on, PPO and SuperMimic alike. One command does the conversion;
the rest of this guide is what to give it, what to check afterwards, and how to keep the
model honest as measurements improve.

```
URDF + meshes ─┐
               ├─> scripts/import_robot.py ─> assets/robots/<name>/   model, meshes, report
measurements ──┘                              configs/robots/<name>.yaml   <- the one file you edit
                                              SuperMimic repo: retarget config + training spec
```

`configs/robots/<name>.yaml` is the single place robot numbers live. PPO reads it
directly; the SuperMimic files are generated from it, so the two can never describe
different robots (the Go1 stand-in has that problem: its two descriptions were written
separately and disagree on the hip angle of the standing pose).

## 1. What the URDF must have

- **12 revolute joints**: per leg a hip abduction joint, then thigh pitch, then calf
  pitch. Make every other joint `fixed`, and remove a floating-base joint if the exporter
  added one.
- **Root link with x forward, y left, z up** (the ROS convention). The legs are told
  apart by where their hips are.
- **A link at each foot**, attached to the calf with a fixed joint, at the centre of the
  foot's ball. If it has a sphere collision, its radius is used; otherwise pass
  `--foot-radius`.
- **Mass and inertia on every moving link** (`<inertial>`).
- **`<limit lower upper effort velocity>`** on every joint, or pass `--torque-limit` and
  `--velocity-limit`.
- Meshes as `.stl` or `.obj`. `package://` paths are fine; ASCII STLs are converted.
  `.dae` files cannot be read and are dropped with a warning (visual only, no effect on
  training).
- A xacro file must be expanded first: `xacro robot.urdf.xacro > robot.urdf`.

What it may have: any link and joint names, any joint frames, either axis direction, any
zero pose, joints that are offset from each other. The importer rebuilds the robot in
this project's convention and records how the two relate (step 4).

Not supported: more or fewer than 12 joints, prismatic joints, legs whose hip does not
turn about the forward axis or whose thigh and calf do not turn about the sideways axis.
Motion retargeting (SuperMimic) additionally needs the thigh axis to cross the hip axis;
if it does not, the importer says so, still writes the model, and PPO still works.

## 2. Measurements

CAD mass properties are usually wrong by more than a policy tolerates: wiring, fasteners,
battery and glue are missing. Measure what you can ([measurement_protocol.md](measurement_protocol.md)
has the procedures and the accuracy to aim for) and put the results in
`data/sysid/processed/<name>.yaml`. Every key is optional; what is absent stays at the
CAD value.

```yaml
total_mass: 12.4            # kg, the assembled robot on a scale
com: [0.008, 0.0, null]     # m, whole-robot centre of mass in the root link's frame, standing
                            #    pose: x, y from the four-scale test, z from the tilt test
                            #    (null = keep CAD)
link_mass:                  # kg, one link of each kind (the calf includes the foot)
  hip: 0.62
  thigh: 1.05
  calf: 0.20
trunk_inertia: [0.07, 0.25, 0.28]   # kg m^2 about the trunk's COM, x y z (pendulum tests)
joint:                      # each: 3 values [hip, thigh, calf] or 12 (FR, FL, RR, RL x 3)
  rotor_inertia: [1.1e-4, 1.1e-4, 1.1e-4]   # kg m^2, motor datasheet
  gear_ratio: [6, 6, 9]                     # -> armature = ratio^2 x rotor inertia
  # armature: [0.004, 0.004, 0.009]         # or give the armature directly (swing-test fit)
  frictionloss: [0.15, 0.15, 0.2]           # N m, dry friction at the joint
  damping: [0.03, 0.03, 0.03]               # N m s/rad, viscous
foot_friction: 0.8          # slip-angle test; sets the randomisation range to 0.6x .. 1.4x
```

How each is applied:

| Key | Effect on the model |
|---|---|
| `link_mass` | That link's mass on all four legs; its inertia scales with the mass. |
| `total_mass` | The trunk takes whatever the legs leave, so the robot weighs this. |
| `com` | The trunk's centre of mass moves until the whole robot's, in the standing pose, is here. |
| `trunk_inertia` | Replaces the trunk's inertia. |
| `joint` | Armature, dry friction and viscous damping of every joint, in the config and the model. |
| `foot_friction` | Friction randomisation range in the config. |

Joint friction and armature matter most on geared motors and are the hardest to get from
CAD. Until the swing-test fit exists (`scripts/fit_sysid.py` is still a stub), use the
datasheet rotor inertia and gear ratio and leave friction to domain randomisation.

## 3. Import

```bash
python scripts/import_robot.py path/to/robot.urdf --name mydog \
    --kp 20 20 20 --kd 0.5 0.5 0.5 \
    --measurements data/sysid/processed/mydog.yaml
```

`--kp` / `--kd` are the position-loop gains the motor drivers will run, at the joint
(hip, thigh, calf; or 12 values). Simulation and robot must use the same gains, so decide
them here. Without them, and with the armature known, gains are derived by the rule mjlab
uses for the Go1 (10 Hz natural frequency on the reflected inertia); treat that as a
starting point.

Other options (`--help` lists all):

| Option | When |
|---|---|
| `--stance THIGH CALF` | Your own standing pose (rad, this project's convention). Default: feet under the hips at 62% of the leg length. |
| `--knees front` | Knees bend forwards in the default stance. |
| `--foot-links A B C D` | The feet are not found by name (`foot`, `toe`, `paw`). |
| `--foot-radius R` | The URDF feet are not spheres. |
| `--torque-limit`, `--velocity-limit`, `--armature` | Override the URDF or the measurements. |
| `--mesh-dir DIR` | Meshes are not near the URDF. |
| `--no-supermimic` | Only the PPO side. |
| `--force` | Replace an existing `configs/robots/<name>.yaml` (hand edits are lost). |

Nothing is written unless the rebuilt model reproduces the URDF: at 20 random poses the
foot positions and centre of mass must agree to a micrometre, and the mass exactly.

## 4. Read the report

`assets/robots/<name>/import_report.md`. Check, in this order:

1. **Leg labels.** FR must be the front-right leg seen from behind the robot. If not, the
   root link's axes are not x forward / y left.
2. **Joint convention.** The model's joint angles follow the Go1 convention: hip 0 = leg
   vertical, thigh 0 = straight down, calf 0 = in line with the thigh, positive hip
   rotates about forward, positive thigh and calf swing the leg backwards. The table gives,
   per joint, `sign` and `offset` in

   ```
   q_urdf = offset + sign * q_sim
   ```

   The same numbers are in the config under `hardware:`. This is the single most
   important thing to verify on the robot: command the standing pose's "URDF angle"
   column on the bench, with the robot hanging, and confirm the legs take the pose the
   simulator shows. A wrong sign here is the classic first-deployment failure.
3. **Mass table.** Which URDF links were merged into which body, and the masses.
4. **Standing test.** The robot is set down in plain MuJoCo with the config's gains for
   two seconds. It must stay up. Look at the joint sag and the holding torque: a sag of
   several tenths of a radian means the gains are soft for the robot's weight, and
   holding torque near 100% of the limit means the motors cannot carry it.
5. **Self-collisions** in the standing pose. Collision shapes of links that are not
   neighbours must not overlap, or the robot fights itself from the first step. Shrink the
   URDF's collision geometry and import again.
6. **Warnings**: dropped meshes, mesh collisions replaced by their bounding boxes, joints
   without limits.

Then look at it and train for a few iterations:

```bash
python scripts/train.py --algo ppo --robot mydog --num-envs 64 --max-iterations 5
python scripts/record.py outputs/ppo/flat_velocity/<run>/ --duration 4
```

## 5. Keep one description

Edit `configs/robots/<name>.yaml` as you learn more (gains after tuning the drivers,
friction after swing tests, randomisation ranges), then

```bash
python scripts/import_robot.py --sync mydog
```

regenerates the SuperMimic files from it. `--sync` does not touch the model; if the URDF
or the mass measurements change, import again with `--force` (and copy over any hand
edits to the config).

Policies depend on the standing pose, gains, action scale and joint order. Changing any
of them invalidates trained policies, so settle them before long runs.

## 6. SuperMimic

The import already wrote the robot into the SuperMimic repository (when it is at
`$SUPERMIMIC_ROOT` or `../supermimic-quadruped`):

| File there | What |
|---|---|
| `src/supermimic/robots/assets/<name>/` | Copy of the model and meshes |
| `configs/robots/<name>.yaml` | Retargeting: legs, standing pose |
| `src/supermimic/robots/<name>.py` | Training spec, generated: do not edit |
| `src/supermimic/robots/__init__.py` | One registry line |

Build the robot's motion library there (dog motion capture retargeted to its legs), which
takes the longest of these steps:

```bash
cd ../supermimic-quadruped
python -m pytest tests/test_quadmotion_robot.py -q                  # kinematics of every robot config
python scripts/data/build_dataset.py --robot mydog                  # retarget every capture
python scripts/data/check_library.py --robot mydog                  # joint limits, ground contact
python scripts/data/screen_clips.py --robot mydog --stiffness 76    # which clips it can follow
python scripts/data/assign_groups.py --robot mydog                  # train / hard / holdout
python scripts/calibrate_scales.py --robot mydog --npz-dir data/npz/mydog
```

`calibrate_scales.py` prints statistics of this robot's motion (rms and 95th percentile
of the per-step changes, integration residuals and state magnitudes, for reference replay
and for a random policy). The world-model scales are chosen from them, as the comments in
the Go1 spec (`src/supermimic/robots/go1.py`, `state_scales`) describe: each `err_scale`
and `out_scale` entry at about the 95th percentile of its quantity, each `in_scale` entry
at about its rms. Until you do this the spec uses the Go1's scales adjusted for standing
height, which is enough to start training. Put the chosen ones in the config and sync:

```yaml
supermimic:
  #           root pos, root rotvec, root lin vel, root ang vel, joint pos, joint vel
  err_scale: [0.01, 0.05, 0.09, 1.0, 0.07, 2.5]
  #           dv, dw, dqd, dp, dth, dq
  out_scale: [0.09, 1.0, 2.5, 0.0007, 0.006, 0.015]
  #           root z, rot6d, lin vel, ang vel, q, qd
  in_scale: [0.26, 1.0, 0.3, 1.0, 1.2, 2.0]
```

Then train from this repository:

```bash
python scripts/train.py --algo supermimic --robot mydog
```

Sizes that SuperMimic tunes per robot (reward widths, reset noise, push speeds,
termination height) are scaled from the Go1 spec by standing height. They are reasonable
defaults, not tuned values.

## 7. On the cluster

`ROBOT=mydog hpc/submit_pipeline.sh` runs the whole pipeline for the robot. Both
repositories must be copied to scratch after the import, since the robot's files live in
both; stage `check` fails within minutes if the SuperMimic spec is missing, and stage
`motions` builds the library if it was not copied.

## Where this sits on the way to the real robot

| Step | Status |
|---|---|
| Real to sim: model from URDF and measurements | This guide. |
| Real to sim: armature / friction fit from swing tests (`scripts/fit_sysid.py`) | Not written yet. Enter datasheet or hand-fitted values under `joint:`. |
| Real to sim: actuator model from logged torque (`scripts/fit_actuator_model.py`) | Not written yet. The config uses an ideal PD servo. |
| Sim to sim: the policy in plain MuJoCo (`scripts/sim2sim.py`) | Not written yet. The import's standing test is the only plain-MuJoCo check so far. |
| Sim to real: export and deployment ([deployment_checklist.md](deployment_checklist.md)) | Not written yet (`scripts/export_policy.py` is a stub). The deployment code must apply `hardware:` (sign, offset) between the policy and the motors. |
