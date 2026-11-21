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

Gait-shaping terms (all off by default; see configs/experiments/ for the presets):

    foot_clearance      exp(-sum_feet (h_f - H)^2 tanh(k |v_f,xy|) / std)   lift swing feet
                        h_f = foot's lowest point above where it last touched down
                        (terrain-agnostic); 0 while the command is ~zero
    feet_slip           sum_feet contact_f * |v_f,xy|           sliding stance feet
    feet_stumble        1 if any foot has |F_xy| > ratio * |F_z|  kicking obstacles
    base_height_band    (max(0, lo - h) + max(0, h - hi))^2      body height in a band,
                        h = base z - mean lowest z of the feet in contact (all feet if none)
    knee_bend           sum_calf max(0, b - |q_calf|)           straight-legged walking
    torques_rated       sum_j (tau_j / tau_ref,j)^2             effort against each motor's
                        continuous (thermal) rating, else its torque limit
    torque_rate         sum_j ((tau_j,t - tau_j,t-1) / tau_ref,j)^2   torque jitter, in
                        the same units as torques_rated; 0 on the first step of an episode
    feet_impact         sum_feet (max(0, |F_z| - k m g / 4) / (m g))^2   hard footfalls,
                        m = the robot's total mass

Foot positions and velocities come from the foot collision geoms (the sphere centres),
ordered as the feet_contact sensor orders them; a foot's lowest point is its centre
minus the geom's size[0] (the sphere radius).

Reward weight ramps (task key ``reward_ramps``) scale a term's weight linearly from
``start_scale`` to 1 between env steps ``start_steps`` and ``end_steps``; see
``reward_weight_ramp`` below.
"""

from __future__ import annotations

import re

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

ROBOT = SceneEntityCfg("robot")


def _robot(env, asset_cfg: SceneEntityCfg = ROBOT):
    return env.scene[asset_cfg.name].data


def _cached(env, key, build):
    """Per-environment cache for lookups that never change (ids, constant tensors)."""
    cache = env.__dict__.setdefault("_reward_term_cache", {})
    if key not in cache:
        cache[key] = build()
    return cache[key]


def _feet(env, sensor_name: str):
    """(contact sensor, foot geom ids in the robot entity, foot radii), ordered like the
    sensor's primaries so per-foot tensors line up with ``sensor.data.found``."""
    sensor = env.scene[sensor_name]

    def build():
        entity = env.scene[ROBOT.name]
        local, _ = entity.find_geoms([re.escape(n) for n in sensor.primary_names],
                                     preserve_order=True)
        global_ids = entity.indexing.geom_ids[local].cpu().numpy()
        radius = env.sim.mj_model.geom_size[global_ids, 0]
        return (torch.tensor(local, dtype=torch.long, device=env.device),
                torch.tensor(radius, dtype=torch.float, device=env.device))

    ids, radius = _cached(env, ("feet", sensor_name), build)
    return sensor, ids, radius


def _foot_low_z(env, ids: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """[N, F] world z of each foot's lowest point."""
    return _robot(env).geom_pos_w[:, ids, 2] - radius


def _foot_speed_xy(env, ids: torch.Tensor) -> torch.Tensor:
    """[N, F] horizontal speed of each foot."""
    return torch.norm(_robot(env).geom_lin_vel_w[:, ids, :2], dim=-1)


def _moving(env, command_name: str, min_command: float) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.norm(cmd[:, :2], dim=1) > min_command).float()


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


def band_penalty(h: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """(max(0, low - h) + max(0, h - high))^2: zero inside [low, high]."""
    return torch.square((low - h).clamp(min=0.0) + (h - high).clamp(min=0.0))


def height_above_feet(base_z: torch.Tensor, foot_z: torch.Tensor,
                      contact: torch.Tensor) -> torch.Tensor:
    """base z minus the mean lowest z of the feet in contact (all feet when none are)."""
    c = contact.float()
    n = c.sum(dim=1)
    in_contact = (foot_z * c).sum(dim=1) / n.clamp(min=1.0)
    ground = torch.where(n > 0, in_contact, foot_z.mean(dim=1))
    return base_z - ground


def base_height_band(env, low: float, high: float,
                     sensor_name: str = "feet_contact") -> torch.Tensor:
    sensor, ids, radius = _feet(env, sensor_name)
    h = height_above_feet(_robot(env).root_link_pos_w[:, 2], _foot_low_z(env, ids, radius),
                          sensor.data.found > 0)
    return band_penalty(h, low, high)


# ---- joints and actions -------------------------------------------------------------

def torques(env) -> torch.Tensor:
    return torch.sum(torch.square(_robot(env).actuator_force), dim=1)


def rated_torque(robot) -> list[float]:
    """Per-joint reference torque for torques_rated: the continuous (thermal) rating if the
    robot config gives one, otherwise the torque limit."""
    act = robot.actuator
    return list(act.continuous_torque if act.continuous_torque is not None else act.torque_limit)


def torques_rated(env, tau_ref: list[float]) -> torch.Tensor:
    tau = _robot(env).actuator_force
    ref = _cached(env, ("tau_ref", tuple(tau_ref), str(tau.device)),
                  lambda: torch.tensor(tau_ref, dtype=tau.dtype, device=tau.device))
    return torch.sum(torch.square(tau / ref), dim=1)


class torque_rate:
    """Change in actuator torque between policy steps, relative to each joint's reference
    torque (as torques_rated). Keeps the last torque per env; after an episode reset the
    first step compares against itself, so it gives 0 rather than a spike."""

    def __init__(self, cfg, env):
        self._last: torch.Tensor | None = None
        self._fresh: torch.Tensor | None = None     # [N] bool: no previous torque yet

    def reset(self, env_ids=None) -> None:
        if self._fresh is not None:
            self._fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env, tau_ref: list[float]) -> torch.Tensor:
        tau = _robot(env).actuator_force
        ref = _cached(env, ("tau_ref", tuple(tau_ref), str(tau.device)),
                      lambda: torch.tensor(tau_ref, dtype=tau.dtype, device=tau.device))
        if self._last is None:
            self._last = tau.clone()
            self._fresh = torch.ones(tau.shape[0], dtype=torch.bool, device=tau.device)
        last = torch.where(self._fresh[:, None], tau, self._last)
        value = torch.sum(torch.square((tau - last) / ref), dim=1)
        self._last = tau.clone()
        self._fresh[:] = False
        return value


def calf_joint_ids(joint_names) -> list[int]:
    """Indices of the knee (calf) joints, found by name."""
    ids = [i for i, n in enumerate(joint_names) if "calf" in n]
    if not ids:
        raise ValueError(f"knee_bend: no joint name contains 'calf' in {list(joint_names)}")
    return ids


def knee_bend(env, joint_ids: list[int], min_bend: float) -> torch.Tensor:
    """Hinge on knee flexion: penalize calves bent less than min_bend (|q_calf| < b)."""
    q = _robot(env).joint_pos[:, joint_ids]
    return torch.sum((min_bend - torch.abs(q)).clamp(min=0.0), dim=1)


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


def clearance_kernel(h: torch.Tensor, speed_xy: torch.Tensor, target: float, std: float,
                     tanh_mult: float) -> torch.Tensor:
    """exp(-sum_feet (h - target)^2 * tanh(tanh_mult * speed) / std), Isaac Lab's Spot form."""
    err = torch.square(h - target) * torch.tanh(tanh_mult * speed_xy)
    return torch.exp(-torch.sum(err, dim=1) / std)


class foot_clearance:
    """Lift swing feet to ``target`` m above where they last touched down.

    Keeps a per-env, per-foot touchdown height, refreshed every step the foot is in
    contact and cleared on episode reset (the first evaluation after a reset takes the
    foot's current height), so the clearance is measured from the local ground on any
    terrain. Feet that move slowly (stance) barely count: tanh(k |v_xy|) ~ 0.
    """

    def __init__(self, cfg, env):
        self._touchdown_z: torch.Tensor | None = None

    def reset(self, env_ids=None) -> None:
        if self._touchdown_z is not None:
            self._touchdown_z[slice(None) if env_ids is None else env_ids] = float("nan")

    def __call__(self, env, target: float, std: float, tanh_mult: float,
                 command_name: str = "twist", sensor_name: str = "feet_contact",
                 min_command: float = 0.1) -> torch.Tensor:
        sensor, ids, radius = _feet(env, sensor_name)
        z = _foot_low_z(env, ids, radius)
        if self._touchdown_z is None:
            self._touchdown_z = torch.full_like(z, float("nan"))
        refresh = (sensor.data.found > 0) | torch.isnan(self._touchdown_z)
        self._touchdown_z = torch.where(refresh, z, self._touchdown_z)
        reward = clearance_kernel(z - self._touchdown_z, _foot_speed_xy(env, ids),
                                  target, std, tanh_mult)
        return reward * _moving(env, command_name, min_command)


def feet_slip(env, sensor_name: str = "feet_contact") -> torch.Tensor:
    sensor, ids, _ = _feet(env, sensor_name)
    contact = (sensor.data.found > 0).float()
    return torch.sum(contact * _foot_speed_xy(env, ids), dim=1)


def stumble(force: torch.Tensor, ratio: float) -> torch.Tensor:
    """1 where any foot's horizontal force exceeds ratio x its vertical force. [N, F, 3]."""
    horizontal = torch.norm(force[..., :2], dim=-1)
    return torch.any(horizontal > ratio * torch.abs(force[..., 2]), dim=1).float()


def feet_stumble(env, ratio: float = 2.0, sensor_name: str = "feet_contact") -> torch.Tensor:
    # feet_contact reduces with "netforce", so force is the net contact force per foot in
    # the world frame (zero when the foot is off the ground).
    return stumble(env.scene[sensor_name].data.force, ratio)


def impact(force_z: torch.Tensor, weight: float, k: float) -> torch.Tensor:
    """sum_feet (max(0, |F_z| - k * weight / 4) / weight)^2, weight = m g. [N, F]."""
    excess = (torch.abs(force_z) - k * weight / 4.0).clamp(min=0.0)
    return torch.sum(torch.square(excess / weight), dim=1)


def _robot_weight(env) -> float:
    """m g of the whole robot (subtree mass of its root body), from the model."""
    def build():
        entity = env.scene[ROBOT.name]
        root = int(entity.indexing.root_body_id)
        model = env.sim.mj_model
        return float(model.body_subtreemass[root]) * float(-model.opt.gravity[2])
    return _cached(env, "robot_weight", build)


def feet_impact(env, k: float = 1.5, sensor_name: str = "feet_contact") -> torch.Tensor:
    # |F_z|: the sign of the netforce z component depends on mjlab's primary/secondary
    # convention, and only the magnitude of the vertical load matters here.
    return impact(env.scene[sensor_name].data.force[..., 2], _robot_weight(env), k)


# Terms that need a sensor in the scene; the builder adds the sensor when one is used.
NEEDS_SENSOR = {
    "feet_air_time": "feet_contact",
    "undesired_contacts": "undesired_contact",
    "foot_clearance": "feet_contact",
    "feet_slip": "feet_contact",
    "feet_stumble": "feet_contact",
    "base_height_band": "feet_contact",
    "feet_impact": "feet_contact",
}


# ---- reward weight ramps (curriculum terms, not rewards) ---------------------------

def ramp_scale(step: int, start_steps: int, end_steps: int, start_scale: float = 0.0) -> float:
    """start_scale before start_steps, 1 after end_steps, linear in between."""
    if step <= start_steps:
        return float(start_scale)
    if step >= end_steps:
        return 1.0
    frac = (step - start_steps) / float(end_steps - start_steps)
    return float(start_scale + (1.0 - start_scale) * frac)


class reward_weight_ramp:
    """mjlab curriculum term: weight = full weight * ramp_scale(env.common_step_counter).

    Same mechanism as mjlab's ``reward_curriculum`` (which only supports step changes):
    mjlab evaluates curriculum terms whenever environments reset, so the weight follows
    the counter at every reset event (several per step with thousands of envs).
    """

    def __init__(self, cfg, env):
        self._term_cfg = env.reward_manager.get_term_cfg(cfg.params["reward_name"])
        self._weight = float(self._term_cfg.weight)
        if int(cfg.params["end_steps"]) < int(cfg.params["start_steps"]):
            raise ValueError(f"reward_ramps.{cfg.params['reward_name']}: end_steps < start_steps")

    def __call__(self, env, env_ids, reward_name: str, start_steps: int, end_steps: int,
                 start_scale: float = 0.0) -> dict[str, torch.Tensor]:
        scale = ramp_scale(int(env.common_step_counter), int(start_steps), int(end_steps),
                           float(start_scale))
        self._term_cfg.weight = self._weight * scale
        return {"scale": torch.tensor(scale), "weight": torch.tensor(self._term_cfg.weight)}
