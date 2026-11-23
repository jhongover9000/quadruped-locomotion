"""Domain randomization: resets, pushes, and per-environment physics variation.

Ranges come from the robot YAML (domain_randomization). Physics parameters are
sampled once per environment at startup; with thousands of environments, the policy
trains across the whole range at once. Pushes happen at random intervals during
episodes. Any key left out of the YAML is simply not randomized.
"""

from __future__ import annotations

import math

from mjlab.envs import mdp as mjlab_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


def make_events(robot, play: bool = False) -> dict[str, EventTermCfg]:
    rand = robot.domain_randomization
    events: dict[str, EventTermCfg] = {
        # Spawn at a random position and heading.
        "reset_base": EventTermCfg(
            func=mjlab_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.02),
                               "yaw": (-math.pi, math.pi)},
                "velocity_range": {},
            },
        ),
        # Start each episode near, not exactly at, the default pose.
        "reset_joints": EventTermCfg(
            func=mjlab_mdp.reset_joints_by_offset,
            mode="reset",
            params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
    }

    if "push_velocity" in rand and not play:
        v = float(rand["push_velocity"])
        events["push"] = EventTermCfg(
            func=mjlab_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=tuple(rand.get("push_interval_s", (5.0, 10.0))),
            params={"velocity_range": {"x": (-v, v), "y": (-v, v)}},
        )
    if "friction" in rand:
        events["foot_friction"] = EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", geom_names=tuple(robot.foot_geoms)),
                    "operation": "abs", "axes": [0], "ranges": tuple(rand["friction"]),
                    "shared_random": True},
        )
    if "added_base_mass" in rand:
        events["base_mass"] = EventTermCfg(
            func=dr.body_mass,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(robot.base_body,)),
                    "operation": "add", "ranges": tuple(rand["added_base_mass"])},
        )
    if "base_com_offset" in rand:
        r = tuple(rand["base_com_offset"])
        events["base_com"] = EventTermCfg(
            func=dr.body_com_offset,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(robot.base_body,)),
                    "operation": "add", "ranges": {0: r, 1: r, 2: r}},
        )
    if "kp_scale" in rand or "kd_scale" in rand:
        events["pd_gains"] = EventTermCfg(
            func=dr.pd_gains,
            mode="startup",
            params={"kp_range": tuple(rand.get("kp_scale", (1.0, 1.0))),
                    "kd_range": tuple(rand.get("kd_scale", (1.0, 1.0))),
                    "operation": "scale"},
        )
    if "motor_strength" in rand:
        events["motor_strength"] = EventTermCfg(
            func=dr.effort_limits,
            mode="startup",
            params={"effort_limit_range": tuple(rand["motor_strength"]), "operation": "scale"},
        )
    # Joint dry friction and reflected rotor inertia: one draw per env and per joint,
    # multiplying the configured value (actuator.frictionloss / armature, else the model's).
    joints = SceneEntityCfg("robot", joint_names=(".*",))
    if "frictionloss_scale" in rand:
        events["joint_frictionloss"] = EventTermCfg(
            func=dr.joint_friction,
            mode="startup",
            params={"asset_cfg": joints, "operation": "scale",
                    "ranges": tuple(rand["frictionloss_scale"])},
        )
    if "armature_scale" in rand:
        events["joint_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="startup",
            params={"asset_cfg": joints, "operation": "scale",
                    "ranges": tuple(rand["armature_scale"])},
        )
    if "encoder_bias" in rand:
        b = float(rand["encoder_bias"])
        events["encoder_bias"] = EventTermCfg(
            func=dr.encoder_bias, mode="startup", params={"bias_range": (-b, b)},
        )
    return events


def action_delay_lag(robot, decimation: int) -> tuple[int, int]:
    """Command latency in physics steps, from action_delay_steps in control steps."""
    lo, hi = robot.domain_randomization.get("action_delay_steps", (0, 0))
    return int(lo) * decimation, int(hi) * decimation
