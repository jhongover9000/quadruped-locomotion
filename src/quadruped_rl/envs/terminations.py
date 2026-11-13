"""Episode terminations. Each returns a [num_envs] bool tensor.

    time_out         episode reached episode_length_s (a truncation: the critic
                     bootstraps through it, since the robot didn't fail)
    base_contact     a base_contact_geoms geom touches the ground (a fall)
    bad_orientation  tilt angle acos(-g_b,z) beyond the limit
"""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

ROBOT = SceneEntityCfg("robot")


def time_out(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def base_contact(env, sensor_name: str = "base_contact") -> torch.Tensor:
    return torch.any(env.scene[sensor_name].data.found > 0, dim=1)


def bad_orientation(env, limit_rad: float, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    g_z = env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]
    return torch.acos(torch.clamp(-g_z, -1.0, 1.0)) > limit_rad
