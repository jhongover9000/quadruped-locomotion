"""Observation terms. Each maps the environment state to a [num_envs, d] tensor.

The task YAML lists terms by name under observations.actor / observations.critic, and
envs/builder.py wires them into mjlab. All quantities are in the base (body) frame
unless noted, which is what the robot's own sensors report.

    base_ang_vel        w_b                 IMU gyro
    projected_gravity   g_b = R^T [0,0,-1]  gravity direction in the body frame (tilt)
    velocity_command    (vx*, vy*, wz*)     what the robot is asked to do
    joint_pos_rel       q - q_default       encoders, relative to the standing pose
    joint_vel           qd
    last_action         a_{t-1}             lets the policy know what it just commanded
    base_lin_vel        v_b                 privileged: hard to measure on hardware
    foot_contacts       1 if foot touching  privileged here (no foot sensors assumed)
    height_scan         terrain heights     privileged on rough terrain
"""

from __future__ import annotations

import torch
from mjlab.envs import mdp as mjlab_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg

ROBOT = SceneEntityCfg("robot")


def base_lin_vel(env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_lin_vel_b


def base_ang_vel(env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_ang_vel_b


def projected_gravity(env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.projected_gravity_b


def velocity_command(env, command_name: str = "twist") -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def joint_pos_rel(env, biased: bool = False, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    """q - q_default. ``biased`` adds the randomized encoder offset (actor only)."""
    data = env.scene[asset_cfg.name].data
    q = data.joint_pos_biased if biased else data.joint_pos
    return q - data.default_joint_pos


def joint_vel(env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_vel


def last_action(env) -> torch.Tensor:
    return env.action_manager.action


def foot_contacts(env, sensor_name: str = "feet_contact") -> torch.Tensor:
    return (env.scene[sensor_name].data.found > 0).float()


def height_scan(env, sensor_name: str = "terrain_scan") -> torch.Tensor:
    return mjlab_mdp.height_scan(env, sensor_name=sensor_name)


# Terms that need a sensor in the scene; the builder adds the sensor when one is used.
NEEDS_SENSOR = {"foot_contacts": "feet_contact", "height_scan": "terrain_scan"}
