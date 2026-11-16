# Import report: testdog

Source: `/tmp/claude-1000/-home-aimlab-code-quadruped-locomotion/eab039e1-a7bc-4c85-88f3-d3f25078aed9/scratchpad/testdog_src/awkward.urdf`, imported 2026-11-16 11:16.

## Legs

| Leg | URDF joints (hip, thigh, calf) | hip to thigh, sideways (m) | thigh (m) | calf to foot centre (m) |
|---|---|---|---|---|
| FR | rf_haa, rf_hfe, rf_kfe | -0.0600 | 0.2000 | 0.2100 |
| FL | lf_haa, lf_hfe, lf_kfe | +0.0600 | 0.2000 | 0.2100 |
| RR | rh_haa, rh_hfe, rh_kfe | -0.0600 | 0.2000 | 0.2100 |
| RL | lh_haa, lh_hfe, lh_kfe | +0.0600 | 0.2000 | 0.2100 |

Foot radius 0.0200 m. Check the leg labels: FR is front right seen from behind the robot.

## Joint convention

`q_urdf = offset + sign * q_sim`. A sign of -1 means the joint turns the other way than in the URDF; a non-zero offset means the URDF's zero is not the straight-down leg.

| Joint | URDF joint | sign | offset (rad) | range here (rad) | standing pose here | standing pose, URDF angle |
|---|---|---|---|---|---|---|
| FR_hip_joint | rf_haa | -1 | +0.0000 | -0.800 .. +0.800 | +0.000 | +0.000 |
| FR_thigh_joint | rf_hfe | -1 | +0.4000 | -2.600 .. +3.400 | +0.933 | -0.533 |
| FR_calf_joint | rf_kfe | -1 | -0.9000 | -3.900 .. +2.100 | -1.805 | +0.905 |
| FL_hip_joint | lf_haa | +1 | +0.0000 | -0.800 .. +0.800 | +0.000 | +0.000 |
| FL_thigh_joint | lf_hfe | +1 | -0.4000 | -2.600 .. +3.400 | +0.933 | +0.533 |
| FL_calf_joint | lf_kfe | +1 | +0.9000 | -3.900 .. +2.100 | -1.805 | -0.905 |
| RR_hip_joint | rh_haa | +1 | +0.0000 | -0.800 .. +0.800 | +0.000 | +0.000 |
| RR_thigh_joint | rh_hfe | +1 | +0.4000 | -3.400 .. +2.600 | +0.933 | +1.333 |
| RR_calf_joint | rh_kfe | +1 | -0.9000 | -2.100 .. +3.900 | -1.805 | -2.705 |
| RL_hip_joint | lh_haa | -1 | +0.0000 | -0.800 .. +0.800 | +0.000 | +0.000 |
| RL_thigh_joint | lh_hfe | -1 | -0.4000 | -3.400 .. +2.600 | +0.933 | -1.333 |
| RL_calf_joint | lh_kfe | -1 | +0.9000 | -2.100 .. +3.900 | -1.805 | +2.705 |

## Mass

| Body | mass (kg) | COM in the trunk frame, legs straight down (m) | URDF links merged |
|---|---|---|---|
| trunk | 6.4000 | [0.0377, 0.0, 0.0102] | base_link, imu_link |
| FR_hip | 0.5000 | [0.2, -0.06, 0.0] | rf_hip |
| FR_thigh | 0.8000 | [0.23, -0.11, -0.08] | rf_thigh |
| FR_calf | 0.2000 | [0.23, -0.1117, -0.3183] | rf_calf, rf_foot |
| FL_hip | 0.5000 | [0.2, 0.06, 0.0] | lf_hip |
| FL_thigh | 0.8000 | [0.23, 0.11, -0.08] | lf_thigh |
| FL_calf | 0.2000 | [0.23, 0.1117, -0.3183] | lf_calf, lf_foot |
| RR_hip | 0.5000 | [-0.2, -0.06, 0.0] | rh_hip |
| RR_thigh | 0.8000 | [-0.17, -0.11, -0.08] | rh_thigh |
| RR_calf | 0.2000 | [-0.17, -0.1117, -0.3183] | rh_calf, rh_foot |
| RL_hip | 0.5000 | [-0.2, 0.06, 0.0] | lh_hip |
| RL_thigh | 0.8000 | [-0.17, 0.11, -0.08] | lh_thigh |
| RL_calf | 0.2000 | [-0.17, 0.1117, -0.3183] | lh_calf, lh_foot |

Total 12.4000 kg. Standing height 0.2742 m.

## Actuators

- torque limits: URDF <limit effort>
- velocity limits: URDF <limit velocity>
- PD gains: from the armature, natural frequency 10.0 Hz, damping ratio 2.0 (the rule mjlab uses for the Go1). A starting point: set --kp / --kd to what the motor drivers will run
- kp [15.633, 15.633, 35.175], kd [0.99526, 0.99526, 2.2393], torque limit [25, 25, 25] N m

## Measurements applied

- calf links: mass set to 0.2000 kg each (inertia scaled with it)
- trunk mass 6.0500 -> 6.4000 kg so the robot weighs 12.4000 kg
- joint armature set from measurements
- joint frictionloss set from measurements
- joint damping set from measurements
- whole-robot COM in the standing pose [-0.0061, 0.001, -0.0196] -> [0.008, 0.0, -0.0196] m (trunk COM moved by [0.0273, -0.002, 0.0])

## Checks

- Rebuilt model against the URDF at 20 random poses (foot positions, centre of mass, mass): largest difference 4.17e-11 (m, kg).
- Standing for 2 s on a floor with these gains: stays up; trunk 0.274 -> 0.207 m, tilt 9.2 deg, largest joint sag 0.295 rad, holding torque up to 37% of the limit.
- Self-collisions in the standing pose: none.

## Motion retargeting (SuperMimic)

- Leg geometry is supported.
- SuperMimic retarget config and training spec written.

## Warnings

- mesh trunk.stl: ASCII STL converted to binary (MuJoCo reads only binary)
- link 'lf_thigh': visual mesh 'package://awkward_description/meshes/thigh.dae' dropped: not found (pass --mesh-dir)
- link 'rf_thigh': visual mesh 'package://awkward_description/meshes/thigh.dae' dropped: not found (pass --mesh-dir)
- link 'lh_thigh': visual mesh 'package://awkward_description/meshes/thigh.dae' dropped: not found (pass --mesh-dir)
- link 'rh_thigh': visual mesh 'package://awkward_description/meshes/thigh.dae' dropped: not found (pass --mesh-dir)
- no readable visual geometry for FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf: drawn with their collision shapes
