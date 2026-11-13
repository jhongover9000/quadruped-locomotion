"""Reward terms, one function per name used in the task YAML.

Total reward per control step: r = dt * sum_i w_i * r_i.

Task terms:
  track_lin_vel_xy    exp(-||v*_xy - v_xy||^2 / sigma)
  track_ang_vel_z     exp(-(w*_z - w_z)^2 / sigma)
  lin_vel_z           v_z^2                  (penalty: bouncing)
  ang_vel_xy          ||w_xy||^2             (penalty: wobbling)
  flat_orientation    ||g_b,xy||^2           (penalty: body tilt)
  torques             ||tau||^2
  joint_acc           ||qdd||^2
  action_rate         ||a_t - a_{t-1}||^2
  feet_air_time       sum_feet (t_air - 0.5) on touchdown (rewards real steps)
  undesired_contacts  thigh/calf/base contacts

Imitation terms (DeepMimic), each exp(-k * error^2) with k = 'scale' in the YAML:
  imitate_joint_pos, imitate_joint_vel, imitate_end_effector,
  imitate_root_pose, imitate_root_vel

TODO(phase 1: task terms; phase 4: imitation terms)."""
