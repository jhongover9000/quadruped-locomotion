# Import report: astra

Source: `/home/aimlab/code/quadruped-locomotion/assets/robots/custom/urdf/astra.urdf`, imported 2026-11-18 11:14.

## Legs

| Leg | URDF joints (hip, thigh, calf) | hip to thigh, sideways (m) | thigh (m) | calf to foot centre (m) |
|---|---|---|---|---|
| FR | FR_j1, FR_j2, FR_j3 | -0.1510 | 0.2500 | 0.2500 |
| FL | FL_j1, FL_j2, FL_j3 | +0.1510 | 0.2500 | 0.2500 |
| RR | RR_j1, RR_j2, RR_j3 | -0.1510 | 0.2500 | 0.2500 |
| RL | RL_j1, RL_j2, RL_j3 | +0.1510 | 0.2500 | 0.2500 |

Foot radius 0.0350 m. Check the leg labels: FR is front right seen from behind the robot.

## Joint convention

`q_urdf = offset + sign * q_sim`. A sign of -1 means the joint turns the other way than in the URDF; a non-zero offset means the URDF's zero is not the straight-down leg.

| Joint | URDF joint | sign | offset (rad) | range here (rad) | standing pose here | standing pose, URDF angle |
|---|---|---|---|---|---|---|
| FR_hip_joint | FR_j1 | +1 | +0.0000 | -0.454 .. +0.454 | +0.000 | +0.000 |
| FR_thigh_joint | FR_j2 | -1 | +1.5708 | -0.000 .. +1.571 | +0.785 | +0.785 |
| FR_calf_joint | FR_j3 | +1 | +3.1416 | -2.749 .. -0.174 | -1.571 | +1.571 |
| FL_hip_joint | FL_j1 | -1 | +0.0000 | -0.454 .. +0.454 | +0.000 | +0.000 |
| FL_thigh_joint | FL_j2 | -1 | +1.5708 | -0.000 .. +1.571 | +0.785 | +0.785 |
| FL_calf_joint | FL_j3 | +1 | +3.1416 | -2.749 .. -0.174 | -1.571 | +1.571 |
| RR_hip_joint | RR_j1 | +1 | +0.0000 | -0.454 .. +0.454 | +0.000 | +0.000 |
| RR_thigh_joint | RR_j2 | -1 | +1.5708 | -0.000 .. +1.571 | +0.785 | +0.785 |
| RR_calf_joint | RR_j3 | +1 | +3.1416 | -2.749 .. -0.174 | -1.571 | +1.571 |
| RL_hip_joint | RL_j1 | -1 | +0.0000 | -0.454 .. +0.454 | +0.000 | +0.000 |
| RL_thigh_joint | RL_j2 | -1 | +1.5708 | -0.000 .. +1.571 | +0.785 | +0.785 |
| RL_calf_joint | RL_j3 | +1 | +3.1416 | -2.749 .. -0.174 | -1.571 | +1.571 |

Joint ranges set at import, not taken from the URDF:

- calf joints (FR_j3, FL_j3, RL_j3, RR_j3): range set to [0.3927, 2.9671] rad in URDF angles by --joint-range (URDF: continuous)

## Mass

| Body | mass (kg) | COM in the trunk frame, legs straight down (m) | URDF links merged |
|---|---|---|---|
| trunk | 6.6000 | [-0.0061, -0.0036, -0.0046] | base |
| FR_hip | 1.0000 | [0.2568, -0.0598, 0.0] | FR_hip |
| FR_thigh | 1.3147 | [0.2626, -0.1552, -0.021] | FR_thigh |
| FR_calf | 0.2853 | [0.2626, -0.2176, -0.3601] | FR_calf, FR_foot |
| FL_hip | 1.0000 | [0.2568, 0.0599, 0.0002] | FL_hip |
| FL_thigh | 1.3147 | [0.2624, 0.1553, -0.0208] | FL_thigh |
| FL_calf | 0.2853 | [0.2626, 0.2177, -0.3595] | FL_calf, FL_foot |
| RR_hip | 1.0000 | [-0.2568, -0.0598, 0.0003] | RR_hip |
| RR_thigh | 1.3147 | [-0.2625, -0.1553, -0.021] | RR_thigh |
| RR_calf | 0.2853 | [-0.2625, -0.2176, -0.3601] | RR_calf, RR_foot |
| RL_hip | 1.0000 | [-0.2568, 0.0599, 0.0002] | RL_hip |
| RL_thigh | 1.3147 | [-0.2627, 0.1553, -0.0208] | RL_thigh |
| RL_calf | 0.2853 | [-0.2624, 0.2177, -0.3595] | RL_calf, RL_foot |

Total 17.0000 kg. Standing height 0.3886 m.

## Actuators

- torque limits: URDF <limit effort>
- velocity limits: URDF <limit velocity>
- PD gains: --kp / --kd
- armature: 0 (unknown). Geared motors need it: put rotor_inertia and gear_ratio, or armature, under joint: in the measurements file
- kp [50, 50, 80], kd [2, 2, 3.2], torque limit [30, 30, 30] N m

## Checks

- Rebuilt model against the URDF at 20 random poses (foot positions, centre of mass, mass): largest difference 3.55e-15 (m, kg).
- Standing for 2 s on a floor with these gains: stays up; trunk 0.389 -> 0.351 m, tilt 2.1 deg, largest joint sag 0.129 rad, holding torque up to 34% of the limit.
- Self-collisions in the standing pose: none.

## Motion retargeting (SuperMimic)

- Leg geometry is supported.
- SuperMimic files not written (--no-supermimic).

## Warnings

- link 'FR_calf': collision mesh at the foot replaced by the foot sphere (r 0.035 m)
- link 'FL_calf': collision mesh at the foot replaced by the foot sphere (r 0.035 m)
- link 'RL_calf': collision mesh at the foot replaced by the foot sphere (r 0.035 m)
- link 'RR_calf': collision mesh at the foot replaced by the foot sphere (r 0.035 m)
