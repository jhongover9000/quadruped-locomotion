"""Reward terms. Each returns an unweighted [num_envs] tensor; the task YAML gives weights.

Total reward per control step (mjlab multiplies by dt so totals don't depend on the
control rate):

    r_t = dt * sum_i  w_i * r_i

Tracking terms are positive and bounded in (0, 1]. Penalty terms return a magnitude
(always >= 0) and get a negative weight in the YAML, so the sign convention is visible
in one place.

    track_lin_vel_xy    exp(-||v*_xy - v_xy||^2 / sigma)       follow the velocity command
    track_ang_vel_z     exp(-(w*_z - w_z)^2 / sigma)           follow the turn-rate command
    lin_vel_z           v_z^2                                   bouncing
    ang_vel_xy          ||w_xy||^2                              rolling / pitching
    flat_orientation    ||g_b,xy||^2                            body tilt
    torques             ||tau||^2                               effort (heat, battery)
    joint_acc           ||qdd||^2                               jerky joint motion
    action_rate         ||a_t - a_{t-1}||^2                     jittery commands
    joint_pos_limits    distance past the soft joint limits
    feet_air_time       sum_feet (t_air - T) at touchdown        take real steps
    undesired_contacts  number of thigh/calf contacts
    base_height         (h - h*)^2                              hold a body height
"""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

ROBOT = SceneEntityCfg("robot")


def _robot(env, asset_cfg: SceneEntityCfg = ROBOT):
    return env.scene[asset_cfg.name].data


# ---- task ---------------------------------------------------------------------------

def track_lin_vel_xy(env, sigma: float, command_name: str = "twist") -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    v = _robot(env).root_link_lin_vel_b
    error = torch.sum(torch.square(cmd[:, :2] - v[:, :2]), dim=1)
    return torch.exp(-error / sigma)


def track_ang_vel_z(env, sigma: float, command_name: str = "twist") -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    w = _robot(env).root_link_ang_vel_b
    error = torch.square(cmd[:, 2] - w[:, 2])
    return torch.exp(-error / sigma)


# ---- base motion --------------------------------------------------------------------

def lin_vel_z(env) -> torch.Tensor:
    return torch.square(_robot(env).root_link_lin_vel_b[:, 2])


def ang_vel_xy(env) -> torch.Tensor:
    return torch.sum(torch.square(_robot(env).root_link_ang_vel_b[:, :2]), dim=1)


def flat_orientation(env) -> torch.Tensor:
    return torch.sum(torch.square(_robot(env).projected_gravity_b[:, :2]), dim=1)


def base_height(env, target: float) -> torch.Tensor:
    return torch.square(_robot(env).root_link_pos_w[:, 2] - target)


# ---- joints and actions -------------------------------------------------------------

def torques(env) -> torch.Tensor:
    return torch.sum(torch.square(_robot(env).actuator_force), dim=1)


def joint_acc(env) -> torch.Tensor:
    return torch.sum(torch.square(_robot(env).joint_acc), dim=1)


def action_rate(env) -> torch.Tensor:
    am = env.action_manager
    return torch.sum(torch.square(am.action - am.prev_action), dim=1)


def joint_pos_limits(env) -> torch.Tensor:
    data = _robot(env)
    lower = data.soft_joint_pos_limits[..., 0]
    upper = data.soft_joint_pos_limits[..., 1]
    below = (lower - data.joint_pos).clamp(min=0.0)
    above = (data.joint_pos - upper).clamp(min=0.0)
    return torch.sum(below + above, dim=1)


# ---- feet and contacts --------------------------------------------------------------

def feet_air_time(env, target: float, command_name: str = "twist",
                  sensor_name: str = "feet_contact", min_command: float = 0.1) -> torch.Tensor:
    """At each touchdown, reward (air time - target). Only while asked to move."""
    sensor = env.scene[sensor_name]
    touchdown = sensor.compute_first_contact(env.step_dt)            # [N, 4] bool
    air_time = sensor.data.last_air_time                             # [N, 4] s
    reward = torch.sum((air_time - target) * touchdown.float(), dim=1)
    cmd = env.command_manager.get_command(command_name)
    moving = torch.norm(cmd[:, :2], dim=1) > min_command
    return reward * moving.float()


def undesired_contacts(env, sensor_name: str = "undesired_contact") -> torch.Tensor:
    return torch.sum((env.scene[sensor_name].data.found > 0).float(), dim=1)


# Terms that need a sensor in the scene; the builder adds the sensor when one is used.
NEEDS_SENSOR = {"feet_air_time": "feet_contact", "undesired_contacts": "undesired_contact"}
