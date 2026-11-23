# What it takes to make the team's quadruped move

Everything that has to exist before the real robot walks under a trained policy: what
the team supplies, the hardware, the numbers the robot and the policy must agree on, the
moteus settings, and the code that is still missing. Written 2026-11-17; the team's
answers of 2026-11-18 about their robot (Astra) are worked in.

This is the list of requirements. The procedures are in the guides and are not repeated
here: [transfer_guide.md](transfer_guide.md) (the whole path),
[robot_import.md](robot_import.md), [measurement_protocol.md](measurement_protocol.md),
[deployment_checklist.md](deployment_checklist.md).

How far to trust each statement:

- **Repo** facts were read from the code and configs on 2026-11-17.
- **moteus and pi3hat** facts were read from the mjbots documentation on 2026-11-17
  ([sources](#sources)). Where a point is marked *unconfirmed*, the page did not state
  it and it comes from general knowledge; check it before relying on it.
- **Team** facts about Astra were given by the team on 2026-11-18 and are marked
  *team, 11-18*. They are reported, not measured here.
- **Proposals** are marked as such. Nothing here has been tried on hardware.

## Where things stand

The tools for bringing the robot into simulation are built. The robot itself has not
been brought in, and nothing between a trained policy and the motors exists yet.

| Step | State |
|---|---|
| Importer (`scripts/import_robot.py`), its guide and the transfer guide | Built. Tested on a synthetic robot (`testdog`) only. |
| The team robot's URDF, gains and measurements | URDF and meshes received (Astra, `assets/robots/custom/`); gains, limits, friction and motor data from the team (11-18, below). No bench measurements in `data/sysid/` yet. |
| Robot imported, standing in simulation | Done 2026-11-18: `configs/robots/astra.yaml`, report in `assets/robots/astra/import_report.md`. Stands under the PD gains; a flat-ground PPO policy walks in simulation. |
| Training in simulation (PPO, SuperMimic, and now DeepMimic, AMP, MPC) | Built. Runs on the Go1 stand-in. GPU validation of the last three is pending ([gpu_validation.md](gpu_validation.md)). |
| Fitting joint friction and the actuator model from bench data | Stubs. |
| Sim2sim check in plain MuJoCo | Stub. |
| Policy export for the Raspberry Pi | Stub. |
| Onboard code: observation builder, control loop, safety layer | Stubs (a docstring each). |
| Driver for the moteus controllers, hardware logging | Do not exist, not even as stubs. |

## 1. What the team supplies

### Robot description

| Item | Requirement |
|---|---|
| URDF | 12 revolute joints (hip abduction, thigh pitch, calf pitch per leg); root link x forward, y left, z up; a fixed link at the centre of each foot; `<inertial>` on every moving link; `<limit lower upper effort velocity>` on every joint; xacro expanded. Any names, frames, axis directions and zero pose. Details: [robot_import.md, section 1](robot_import.md#1-what-the-urdf-must-have). |
| Meshes | `.stl` or `.obj` (`.dae` is dropped). Into `assets/robots/custom/meshes/`, in Git LFS. |
| PD gains | `kp` (N m/rad) and `kd` (N m s/rad) for hip, thigh and calf: the gains the controllers will actually run. They are fixed at import; changing them later invalidates trained policies. Astra: kp 50, 50, 80 and kd 2.0, 2.0, 3.2, run on the controllers as scales of a fixed ceiling ([section 4](#4-moteus-settings)). |
| Torque and speed limits | Per joint, at the output: the peak torque the actuator can give (N m), the torque it can hold continuously, and its top speed (rad/s). Astra (*team, 11-18*): peak 30 N m measured; the true peak may be about 40 N m but is unmeasured. Continuous about 5 N m per joint without cooling, limited by the driver's temperature; fans are being added to the knee drivers. The config carries both: `torque_limit: [30, 30, 30]` and `continuous_torque: [5, 5, 5]`, which `scripts/record.py` reports RMS torques against. Simulated walking needs 9 to 14 N m RMS at the knee, so the continuous rating is the binding limit. |
| Joint ranges | Astra (*team, 11-18*): hip pitch 0 to 90 deg preferred, -20 to 100 deg possible if needed (the URDF has 0 to 90). Knee: no mechanical stop, it turns a full 360 deg; the range 0.3927 to 2.9671 rad (URDF angles) is set at import and has to be enforced in software on the controllers ([section 4](#4-moteus-settings)). |
| Standing pose | The joint angles of the nominal stance, if the team has one in mind; otherwise the importer's `--stance` options choose it. |

### Actuator data (per joint type)

| Item | Why |
|---|---|
| Motor model and datasheet | Rotor inertia, torque constant, current limit. |
| Gear ratio | Sets armature (ratio squared times rotor inertia) in simulation and `motor_position.rotor_to_output_ratio` on the controller. |
| Which way each motor turns and where its zero is | The sign and offset per joint. The importer derives them for the URDF; the bench check confirms them for the motors. |
| Whether there is a sensor on the output side of the gearbox | Decides how the joints are zeroed at power-on ([section 4](#4-moteus-settings)). |

Astra (*team, 11-18*): the same motor on all 12 joints, an EaglePower 8308, 90 KV, with a
10:1 cycloidal reducer; the knee adds a 1:1 belt, so every joint is 10:1. The encoder is
on the motor side only; there is no output-side sensor. Rotor inertia is not measured
yet. The armature in the config, 0.015 kg m^2, is our estimate: a rotor of about
1.5e-4 kg m^2 times 10^2.

### Measurements

Into `data/sysid/processed/<name>.yaml`; every key is optional and what is missing stays
at the CAD value. Procedures and formulas: [measurement_protocol.md](measurement_protocol.md).

| Quantity | Accuracy to aim for | How |
|---|---|---|
| Total mass | 1 to 2% | Scale |
| Centre of mass, x and y | 5 to 10 mm | One scale under each foot |
| Centre of mass height | 1 to 2 cm | Tilt test |
| Link masses (hip, thigh, calf with foot) | 5 to 10% | Scale, one leg. Astra (*team, 11-18*): the 6.6 kg trunk includes the battery, computer, wiring and the four abduction actuators with their drivers. |
| Trunk inertia | 20 to 30% | Pendulum tests |
| Joint zero offsets | about 1 degree | Alignment jig |
| Armature, dry friction, damping | From swing tests | Needs `scripts/fit_sysid.py` (stub); until then datasheet rotor inertia and gear ratio. Astra friction (*team, 11-18*): knee 2.0 to 2.5 N m, hip pitch 0.2 N m measured; abduction 0.2 N m assumed. Rotor inertia not measured (armature 0.015 kg m^2 estimated). |
| Continuous torque | The torque at which the driver derates | Bench test, Astra: hold 8, 10 and 12 N m on one joint and log the driver temperature until it derates; with and without the knee fans |
| Foot friction | Rough | Slip angle on a tilted board, per floor surface |

### Answers still needed

1. How the Pi reaches the controllers: mjbots pi3hat, fdcanusb, or another CAN-FD adapter.
2. Which IMU, where it is mounted and how it is oriented.
3. Battery voltage and chemistry.
4. How velocity commands reach the robot (gamepad, keyboard over Wi-Fi, fixed script).
5. How the emergency stop is wired (cuts motor power, or only tells the software).
6. The units of the gain ceiling (kp 318, kd 7.16; [section 4](#4-moteus-settings)):
   N m/rad as stated, or moteus's own N m per revolution.
7. The true peak torque (about 40 N m?) and the rotor inertia, both unmeasured.

Answered by the team on 2026-11-18: motors and gear ratios, and whether there are
output-side encoders (EaglePower 8308, 10:1, motor-side encoder only; section 1).

## 2. Hardware

| Part | What is known | What it has to do |
|---|---|---|
| Onboard computer | Raspberry Pi 4 | Run the policy and the control loop at 50 Hz on its CPU (no GPU). 64-bit OS, so that `onnxruntime` installs (*unconfirmed*: check the wheel for the OS chosen). |
| Motors | 12 x EaglePower 8308, 90 KV, 10:1 cycloidal reducer, knee through an extra 1:1 belt (*team, 11-18*) | Continuous torque about 5 N m per joint without cooling, set by the driver temperature; fans are being added to the knee drivers. |
| Motor controllers | 12 x moteus r4.11 | Position control with the simulated gains. Supply 10 V minimum, 24 V nominal, 44 V absolute maximum (10S or less). Peak output 100 A. Power rating 900 W at 30 V or less, 400 W at 38 V or more. |
| Link to the controllers | Open | CAN-FD at 1 Mbit standard and 5 Mbit data rate. The Pi has no CAN-FD port. The pi3hat gives four 5 Mbps CAN-FD ports, one 125 kbps CAN port and a 1 kHz IMU with attitude reference, and works on a Pi 4. Four buses fit one leg (three controllers) per bus (*proposal*). |
| IMU | Open | Body angular velocity and the direction of gravity, at 50 Hz or faster, in a known frame. The pi3hat's IMU would do; its frame configuration was not on the page read. |
| Battery | Open | Inside the controllers' 10 to 44 V range with margin below 44 V for regeneration; the controller faults when the bus reaches `servo.max_voltage`. |
| Emergency stop | Open | In hand whenever the robot is powered. Cutting motor power is the safest form. |
| Bench equipment | For the team | Four scales, two supports for the tilt test, a pendulum rig and stopwatch, an alignment jig for the joint zeros, a tilting board, a harness or gantry that carries the robot with its feet off the ground, and a current-limited bench supply for first power-up (*proposal*). |

## 3. What the policy and the robot must agree on

These numbers are fixed at training time. The onboard code has to reproduce them exactly;
most failed first deployments are a mismatch here, not a training problem. Values are for
the velocity tasks (`configs/tasks/flat_velocity.yaml`) and the in-house PPO
(`configs/algorithms/ppo.yaml`).

**Rate.** 50 Hz (physics step 0.005 s, policy every 4 steps). Training randomises the
command delay between 0 and 1 control step, so the time from reading the sensors to the
controllers acting on the new target has to stay under 20 ms, and a few ms is better.

**Network.** Input 45 numbers, hidden layers 512, 256, 128 with ELU, output 12. The input
is normalised with running statistics stored in the checkpoint; the export has to carry
them. Inference should take well under a millisecond on the Pi (*estimate*: measure it).

**Observation**, in this order (confirm order and length against a checkpoint when the
export is written):

| Index | Term | Meaning | Unit and frame | Source on the robot | Noise trained with |
|---|---|---|---|---|---|
| 0 to 2 | `base_ang_vel` | Angular velocity of the trunk | rad/s, body frame | IMU gyro | ±0.2 |
| 3 to 5 | `projected_gravity` | Direction of gravity; (0, 0, -1) when level | unit vector, body frame | IMU attitude | ±0.05 |
| 6 to 8 | `velocity_command` | Forward, sideways, yaw | m/s, m/s, rad/s | Operator | none |
| 9 to 20 | `joint_pos_rel` | Joint angle minus the standing pose | rad, `joint_names` order | Controllers | ±0.01 |
| 21 to 32 | `joint_vel` | Joint velocity | rad/s, `joint_names` order | Controllers | ±1.5 |
| 33 to 44 | `last_action` | The network's previous output, unscaled | none | Stored by the loop | none |

There are no extra scale factors on the terms. Commands were trained within ±1.0 m/s
forward, ±0.5 m/s sideways and ±1.0 rad/s yaw; clamp the operator's input to that.

**Action.** `q_target = default_joint_pos + scale_j * action`, with
`scale_j = 0.25 * torque_limit_j / kp_j`. For the Go1 stand-in that is 0.373 rad on hip
and thigh and 0.249 rad on the calf. The simulated servo is an ideal PD loop:
`torque = kp * (q_target - q) - kd * qd`, clipped to `torque_limit`, with no integral term.

**Joint order and convention.** The policy works in the project's convention and the
order of `joint_names` in `configs/robots/<name>.yaml` (FR, FL, RR, RL; hip, thigh, calf).
The robot YAML's `hardware:` block gives the map to the URDF's joints:
`q_urdf = offset + sign * q_sim`, with `urdf_joint_names` for the order.

**Motion tracking policies** (SuperMimic, DeepMimic) also need the reference clips on the
robot, because their observation includes future reference frames. Nothing exists for
that. The first policy on hardware should be a velocity policy.

## 4. moteus settings

moteus works in revolutions at the output shaft, not radians. Every quantity crossing
the boundary needs a conversion, and a wrong one moves a joint six times too far.

| Quantity | Simulation | moteus | Conversion |
|---|---|---|---|
| Position | rad | revolutions | `rev = (offset + sign * q_sim) / (2 pi)`, plus the controller's own zero |
| Velocity | rad/s | revolutions/s | `rad/s = sign * 2 pi * rev/s` |
| Torque | N m at the joint | N m at the output shaft | none, once the gear ratio is set |
| `kp` | N m/rad | `servo.pid_position.kp`, N m per rotation | times 2 pi (Go1 example: 15.895 becomes 99.9) |
| `kd` | N m s/rad | `servo.pid_position.kd`, N m per rotation/s | times 2 pi (Go1 example: 1.012 becomes 6.36) |
| Gear ratio N:1 | armature | `motor_position.rotor_to_output_ratio` | `1 / N` (6:1 is 0.1667; Astra's 10:1 on every joint, belt included, is 0.1) |

Settings on each controller:

| Setting | Value | Why |
|---|---|---|
| `id.id` | 1 to 12 in `joint_names` order (*proposal*) | One address per joint; IDs are 7 bits |
| `motor_position.rotor_to_output_ratio` | `1 / N` | Positions, velocities and torques are then at the output |
| `servo.pid_position.kp`, `kd` | The simulated gains times 2 pi; on Astra a fixed ceiling, see below | Same servo as in training |
| `servo.pid_position.ki` | 0 | The simulated servo has no integral term |
| Default velocity and acceleration limits | Disabled (*unconfirmed* names: `servo.default_velocity_limit`, `servo.default_accel_limit`) | If set, the controller follows a trajectory to the target, which is not the PD loop the policy was trained on |
| `servopos.position_min`, `position_max` | The joint limits in revolutions, in the motor's sign. Astra knee: 0.0625 and 0.4722 rev (URDF 0.3927 to 2.9671 rad divided by 2 pi), given homing that makes the output position equal the URDF angle / 2 pi | Fault 39 when control starts outside them. On Astra's knee, which has no mechanical stop, this is the only thing that keeps the joint in range |
| `servo.max_current_A` | From the motor's limit | Caps the torque whatever the software asks |
| `servo.max_voltage` | Above the full battery, below 44 V | Regeneration |
| `servo.default_timeout_s` | About 0.1 s (*proposal*; default not found) | If the Pi stops sending, the controller enters its timeout state and needs a stop command to resume |
| `servo.timeout_mode`, `servo.timeout_max_torque_Nm` | 12 (zero velocity, i.e. damping) with a low torque cap (*proposal*; try it in the harness) | What the legs do when the Pi dies. The choices are 0 stopped, 10 decelerate and hold, 12 zero velocity, 15 brake. |

**Astra's gains** (*team, 11-18*). The PD loop runs on the controller with a ceiling of
kp 318 N m/rad and kd 7.16 N m s/rad, which a command can only scale down
(`kp_scale`, `kd_scale` between 0 and 1). The simulated gains as scales:

| Joint | sim kp (N m/rad) | `kp_scale` = kp / 318 | sim kd (N m s/rad) | `kd_scale` = kd / 7.16 |
|---|---|---|---|---|
| Hip abduction, hip pitch | 50 | 0.157 | 2.0 | 0.279 |
| Knee | 80 | 0.252 | 3.2 | 0.447 |

Check before use: 318 / 2 pi = 50.6 and 7.16 / 2 pi = 1.14. If the ceiling is in moteus's
own units (N m per revolution) rather than N m/rad, it is only 50.6 N m/rad: hip and thigh
kp 50 would need a scale of 0.99, and the knee's 80 would be above the ceiling.

Each cycle the loop sends every controller a position command with: position (the
target in revolutions), velocity 0, feedforward torque 0, `kp_scale` and `kd_scale` 1
(on Astra, the scales above), and maximum torque equal to the simulated `torque_limit`. The reply carries mode,
position, velocity, torque, voltage, temperature and fault code; position and velocity
feed the observation and all of it goes to the log.

**Zeroing at power-on.** The onboard encoder senses the rotor. Behind a gearbox the
output angle is therefore known only up to a multiple of one rotor turn divided by the
gear ratio when the controller starts. The documented ways to resolve it are an
output-side sensor or a homing switch, configured as
`motor_position.output.reference_source`. Without one, the usual practice is to power
the robot on in a known pose (the alignment jig) and tell each controller to take the
nearest matching position (*unconfirmed*: the "set output nearest" command exists, its
exact behaviour was not on the pages read). Whichever way is chosen, it sets the joint
zero offsets, which need about 1 degree.

Astra has no output-side sensor (*team, 11-18*), so this applies to every joint: with
10:1 reducers, the output angle at power-on is ambiguous by 36 deg per motor turn. The
robot has to start in a known pose, for example lying down in the team's all-zero pose,
or in a jig, and each controller is then told its output position (the team names
moteus's "set output nearest" and the diagnostic command `d exact`; neither name was
checked against the documentation from this repository). Set it so that the output
position in revolutions equals the URDF angle / 2 pi; the knee limits above assume
that. A knee homed one motor turn off is not caught by a hard stop, because there is
none.

Fault codes worth recognising in the log: 33 motor driver, 34 over voltage, 35 encoder,
38 over temperature, 39 outside limit, 40 under voltage, 43 position invalid.

## 5. Code still to write

In the order it is needed. Paths marked *new* are proposals; the others are existing stubs.

| Piece | Where | What it must do | Done when |
|---|---|---|---|
| Sim2sim | `scripts/sim2sim.py`, `src/quadruped_rl/envs/backends/mujoco_cpu.py` | Run a trained policy in plain CPU MuJoCo, with the observation built by code that shares nothing with the training environment | Tracking error and fall rate are within a stated tolerance of the training simulator |
| Export | `scripts/export_policy.py`, `src/quadruped_rl/deploy/export.py` | Write the actor with its normalisation as ONNX or TorchScript, plus a config holding joint order, standing pose, action scales, gains and rate | The exported file gives the same actions as the checkpoint on recorded observations, on the laptop and on the Pi |
| Observation builder | `src/quadruped_rl/deploy/obs_builder.py` | Turn IMU and controller readings into the 45 numbers of section 3: sign, offset, revolutions to radians, IMU frame, previous action | Fed with a simulated rollout's raw sensor values, it reproduces that rollout's observations element by element |
| moteus driver | `src/quadruped_rl/deploy/moteus_io.py` (*new*) | Send 12 position commands and read 12 replies in one cycle, with the conversions of section 4; stop command; fault reporting | A measured round trip for 12 controllers that fits the 20 ms budget with room to spare |
| Control loop | `src/quadruped_rl/deploy/policy_runner.py` | 50 Hz: read, build observation, run the policy, send targets. Start by standing: ramp from the measured pose to the standing pose before the policy takes over | Loop period and jitter measured on the Pi |
| Safety layer | `src/quadruped_rl/deploy/safety.py` | Clamp targets to the joint limits and to a maximum step per cycle, cut out on tilt, on a stale reading, on a controller fault and on the emergency stop; always on | Each cutout shown to trigger in the harness |
| Hardware logging | *new* | Write every cycle's observation, action, targets and controller replies to `data/hardware_logs/` | A hardware run can be compared with a simulated one, observation by observation |
| Mass properties wrapper | `scripts/compute_mass_properties.py` | Command line over the formulas already in `src/quadruped_rl/sysid/mass_properties.py` | Writes `data/sysid/processed/<name>.yaml` from raw readings |
| Joint fit | `scripts/fit_sysid.py` and `src/quadruped_rl/sysid/` | Armature, dry friction and damping from swing tests | Reproduces a recorded swing |
| Actuator model | `scripts/fit_actuator_model.py`, `src/quadruped_rl/actuators/` | Fit the real servo's response from controller logs and use it in training | Simulated and logged torque agree on a held-out log |

The last three improve the match between simulation and robot. They are not needed for
the first steps; domain randomisation covers the gap until then.

## 6. Shortest path to first steps

1. The team delivers the URDF, meshes, gains, limits and gear ratios (section 1).
2. Import, read the import report, check that the robot stands in simulation with no policy.
3. Train a flat-ground PPO velocity policy; look at the recording.
4. Write sim2sim and pass it.
5. Write the export, the observation builder, the driver, the loop, the safety layer and
   the logging (section 5). The driver and the conversions can be developed against one
   controller on the bench before the robot is assembled.
6. Configure the controllers (section 4) and check each joint's sign, zero and limits on
   the bench, one joint at a time, at low torque.
7. Go through [deployment_checklist.md](deployment_checklist.md): hanging in the harness,
   then standing, then the gantry, then free walking with slow commands.
8. Compare the hardware logs with simulation and feed the differences back into the
   robot YAML and the randomisation ranges.

Measurements (mass, centre of mass) should be in before step 3 if they are available,
because re-importing with new masses means retraining. SuperMimic, DeepMimic and AMP on
the real robot come after a velocity policy walks.

## Sources

Team answers about Astra (motors, gains, homing, torque, friction, masses), 2026-11-18.

Read 2026-11-17:

- moteus documentation, <https://mjbots.github.io/moteus/>: configuration reference,
  control modes guide, hardware reference (electrical limits), register list, encoder
  reference, CAN protocol, Python reference.
- pi3hat README, <https://github.com/mjbots/pi3hat>.

Not found on those pages, so not stated above as fact: continuous current of the r4.11,
default values of the watchdog timeout and of the limits, CAN termination and bus
length, devices per bus, the pi3hat's IMU frame configuration and input voltage, the
Python package names and whether they need root, and an achievable update rate for 12
controllers from Python (the protocol page says only that over 1 kHz is possible on the
bus).
