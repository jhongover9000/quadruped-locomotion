"""Velocity commands: what the policy is asked to do.

Each environment samples (vx*, vy*, wz*) uniformly from the task YAML ranges, holds it
for resample_interval_s, then samples again. A fraction of environments is told to
stand still so the policy also learns a stable stance.
"""

from __future__ import annotations

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

COMMAND_NAME = "twist"


def make_velocity_command(cmd_cfg: dict) -> UniformVelocityCommandCfg:
    interval = float(cmd_cfg.get("resample_interval_s", 10.0))
    return UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(interval, interval),
        rel_standing_envs=float(cmd_cfg.get("standing_fraction", 0.0)),
        heading_command=False,
        rel_heading_envs=0.0,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=tuple(cmd_cfg["lin_vel_x"]),
            lin_vel_y=tuple(cmd_cfg["lin_vel_y"]),
            ang_vel_z=tuple(cmd_cfg["ang_vel_z"]),
        ),
    )
