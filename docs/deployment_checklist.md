# Deployment checklist

Most failed first deployments are interface mismatches, not RL problems.

## Before the robot moves

- [ ] Policy passes sim2sim (`scripts/sim2sim.py`) in plain MuJoCo
- [ ] Joint order matches `joint_names` in the robot YAML
- [ ] Joint sign conventions and zero offsets verified with the alignment jig
- [ ] IMU frame matches the simulated base frame (check projected gravity while tilting by hand)
- [ ] Default pose, action scale, PD gains, and control rate match the training config
- [ ] Observation normalization is baked into the exported policy
- [ ] Log one sim rollout and one hardware rollout; compare observation vectors element by element
- [ ] Safety layer active: joint/torque clamps, tilt cutoff, watchdog, e-stop in hand

## Staged testing

1. Hanging in the harness (legs should make a walking motion in the air)
2. Standing
3. Walking on the gantry
4. Free walking, slow commands first

Log every hardware run to `data/hardware_logs/` and compare against sim to refine the
actuator model and randomization ranges.
