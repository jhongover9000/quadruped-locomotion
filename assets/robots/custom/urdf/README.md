# Astra – URDF

## Contents
- `astra.urdf` – robot model (SI units)
- `meshes/` – visual meshes and foot collision meshes

## Frames
- Base: x forward, y left, z up. Origin at the body center, at the height of the hip pitch axes.
- Legs: `FR`, `FL`, `RL`, `RR`. Links per leg: `hip`, `thigh`, `calf`, plus a `foot` frame at the foot center.
- Joints per leg: `j1` hip abduction, `j2` hip pitch, `j3` knee (e.g. `FR_j1`).
- The base is not fixed: add a floating base in the simulator.

## Joint conventions
Same meaning on all four legs:
- `j1`: 0 = leg vertical, positive = foot toward the body center.
- `j2`: 0 = thigh horizontal pointing backward, positive = knee down.
- `j3`: knee inner angle, 0 = calf folded onto the thigh.

All joints at 0 = lying pose. Nominal standing pose: `j1` 0°, `j2` 45°, `j3` 90°.

## Limits

| Joint | Range | Effort | Velocity |
|---|---|---|---|
| j1 | ±26° | 30 Nm | 20 rad/s |
| j2 | 0° to 90° | 30 Nm | 20 rad/s |
| j3 | continuous | 30 Nm | 20 rad/s |

Some limits can be widened if needed.

## Notes
- Joint friction is set in `<dynamics>`. Some simulators ignore it on import.
- Collisions use simplified shapes; feet use a convex mesh.
- MuJoCo strips mesh paths when importing a URDF: set `meshdir` to `meshes/`.
