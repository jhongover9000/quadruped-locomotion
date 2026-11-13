"""Observation terms, one function per name used in the task YAML.

base_ang_vel        IMU gyro, body frame
projected_gravity   g_b = R^T [0, 0, -1]: gravity in the body frame (encodes tilt)
velocity_command    (vx*, vy*, wz*)
joint_pos_rel       q - q_default
joint_vel           qd
last_action         a_{t-1}
foot_contacts       binary per foot (critic only)
friction            sampled ground friction (critic only)
base_lin_vel        true base velocity (critic only; hard to measure on hardware)
height_scan         terrain heights around the base (critic only, rough terrain)
motion_phase        phase of the reference clip in [0, 1) as (sin, cos)
ref_future_frames   next K reference frames relative to the current state

TODO(phase 1)."""
