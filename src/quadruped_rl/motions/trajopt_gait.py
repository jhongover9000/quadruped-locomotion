"""Generates reference gaits by trajectory optimization on the robot model.

Decision variables: joint trajectories and contact forces over one gait cycle with a fixed
contact schedule (trot, bound, pace). Cost includes integral of ||q_ddd||^2 for smoothness,
torque, and velocity tracking; constraints are dynamics, friction cones, and periodicity.
Uses the 'trajopt' extra (Pinocchio + CasADi).

TODO(phase 3)."""
