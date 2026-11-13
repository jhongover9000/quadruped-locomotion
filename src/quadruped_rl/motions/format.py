"""Reference-motion file format (.npz), consumed by every imitation method.

    fps        scalar
    root_pos   [T, 3]  m, world frame
    root_quat  [T, 4]  w, x, y, z
    joint_pos  [T, 12] rad, in the robot config's joint order
    foot_pos   [T, 4, 3] m, world frame (FR, FL, RR, RL)

Velocities are derived by finite differences when loaded.
TODO(phase 3): save/load helpers and validation."""
