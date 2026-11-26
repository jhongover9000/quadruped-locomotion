"""Run velocity-command policies on the evaluation courses (envs/courses.py) and score them.

A shared high-level controller turns each robot's world pose and its goal into a
velocity command (vx, vy, wz); the policy under test only does locomotion, so
differences between policies come from the policy alone.

- gauntlet (and each ladder): lane following. vx = speed, vy = -k_y * lateral error,
  wz = k_yaw * heading error, clipped to the gains' limits and to the policy's trained
  command range.
- directional gauntlet (sections with a ``mode``): the robot still travels +x but holds
  the body yaw of its section's mode. World velocity v_w = (speed, -k_y * lateral error),
  wz = k_yaw * wrap(mode yaw - yaw), body (vx, vy) = R(-yaw) v_w, scaled (not clipped per
  axis) into the trained range. Travel slows while the yaw error is large; a spin section
  turns on the spot by spin_deg before going on.
- beacon: homing. e = bearing to the goal - yaw; wz = k_yaw * e; vx = v_max * max(0, cos e),
  slowing linearly inside slow_radius_m; vy = 0.

Evaluation conditions are fixed: no domain randomization (``eval_robot``), no
observation noise and deterministic actions (play mode), robots reset exactly at their
start (``course_task``). A robot's attempt ends at its first reset (a fall, or leaving
the terrain); everything after it is masked out of the metrics.

Used by scripts/course.py. The controllers and the metrics are plain numpy.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import math
from pathlib import Path

import numpy as np

from quadruped_rl.envs import courses as C
from quadruped_rl.evaluation import metrics

VELOCITY_ALGOS = ("ppo", "ppo_rslrl", "amp", "mpc")
JOINT_KINDS = ("hip", "thigh", "calf")
RUNNING = "running"
FALL_STATUSES = ("fell", "out_of_bounds")     # counted as falls in the section tables
STOP_STATUSES = ("timeout", "stuck")          # counted as t/o
# directional gauntlet: default travel speed per mode, m/s (forward: the controller's speed)
MODE_SPEEDS = {"backward": 0.5, "left": 0.35, "right": 0.35,
               "diag_fl": 0.5, "diag_fr": 0.5, "diag_bl": 0.5, "diag_br": 0.5}
TRANSITION_DEG = 15.0                         # yaw error above which travel slows


# ---- evaluation conditions ----------------------------------------------------------

def check_compatible(algo_name: str | None, task: dict) -> None:
    """Courses need a policy that follows a velocity command."""
    from quadruped_rl.algorithms.base import task_kind

    if task_kind(task) != "velocity" or (algo_name and algo_name not in VELOCITY_ALGOS):
        raise ValueError(
            f"The evaluation courses drive a policy with velocity commands; this run "
            f"({algo_name or '?'}, {task_kind(task)} task '{task.get('name')}') tracks motion "
            f"clips and has no velocity command. Supported: {', '.join(VELOCITY_ALGOS)}.")


def action_delay_steps(robot) -> int:
    """The fixed command latency used on the courses: the middle of the training range,
    rounded down (control steps)."""
    lo, hi = robot.domain_randomization.get("action_delay_steps", (0, 0))
    return (int(lo) + int(hi)) // 2


def eval_robot(robot):
    """The robot config with every domain randomization removed (friction, mass, CoM,
    gains, motor strength, joint friction, armature, encoder bias, pushes) and the
    action delay fixed at ``action_delay_steps``."""
    delay = action_delay_steps(robot)
    dr = {"action_delay_steps": [delay, delay]} if delay else {}
    return dataclasses.replace(robot, domain_randomization=dr)


def course_task(task: dict, course: dict, num_envs: int) -> dict:
    """The task config with the course as its terrain and exact, fixed resets."""
    task = copy.deepcopy(task)
    task["terrain"] = {"type": "course", "course": copy.deepcopy(course), "lanes": num_envs}
    task["fixed_reset"] = True
    return task


def command_ranges(task: dict) -> dict[str, tuple[float, float]]:
    c = task["commands"]
    return {k: (float(c[k][0]), float(c[k][1])) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")}


def eval_conditions(robot) -> dict:
    return {"domain_randomization": "none", "observation_noise": "none",
            "actions": "deterministic", "pushes": "none",
            "action_delay_steps": action_delay_steps(robot),
            "reset": "exact start pose, default joint pose",
            "falls": "an attempt ends at its first reset; later steps are masked out"}


# ---- controllers --------------------------------------------------------------------

def wrap_pi(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def yaw_from_quat(q: np.ndarray) -> np.ndarray:
    """Yaw of (w, x, y, z) quaternions [..., 4]."""
    w, x, y, z = (q[..., i] for i in range(4))
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _clip_to_ranges(cmd: np.ndarray, ranges: dict) -> np.ndarray:
    for i, key in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z")):
        lo, hi = ranges[key]
        cmd[..., i] = np.clip(cmd[..., i], lo, hi)
    return cmd


def lane_command(y, yaw, lane_y, *, speed: float, k_y: float, vy_max: float, k_yaw: float,
                 wz_max: float, ranges: dict) -> np.ndarray:
    """[N, 3] commands that keep robots on their lane centre line, heading +x."""
    lateral = np.asarray(y, float) - np.asarray(lane_y, float)          # + = left of the line
    heading = wrap_pi(0.0 - np.asarray(yaw, float))
    cmd = np.stack([np.full(lateral.shape, float(speed)),
                    np.clip(-k_y * lateral, -vy_max, vy_max),
                    np.clip(k_yaw * heading, -wz_max, wz_max)], axis=-1)
    return _clip_to_ranges(cmd, ranges)


def beacon_command(xy, yaw, goal, *, v_max: float, k_yaw: float, wz_max: float,
                   slow_radius_m: float, ranges: dict) -> np.ndarray:
    """[N, 3] homing commands towards ``goal``."""
    d = np.asarray(goal, float) - np.asarray(xy, float)
    dist = np.linalg.norm(d, axis=-1)
    e = wrap_pi(np.arctan2(d[..., 1], d[..., 0]) - np.asarray(yaw, float))
    slow = np.clip(dist / slow_radius_m, 0.0, 1.0) if slow_radius_m > 0 else np.ones_like(dist)
    cmd = np.stack([v_max * np.maximum(0.0, np.cos(e)) * slow, np.zeros_like(dist),
                    np.clip(k_yaw * e, -wz_max, wz_max)], axis=-1)
    return _clip_to_ranges(cmd, ranges)


def mode_yaw(mode: str) -> float:
    """Body yaw (rad) a directional section's mode holds while travelling +x."""
    return math.radians(C.MODE_YAW_DEG[mode])


def clip_planar(cmd: np.ndarray, ranges: dict) -> np.ndarray:
    """[..., 3] body commands with (vx, vy) scaled down together until both lie in the
    trained ranges (the direction is kept) and wz clipped."""
    cmd = np.array(cmd, dtype=float)
    scale = np.ones(cmd.shape[:-1])
    for i, key in enumerate(("lin_vel_x", "lin_vel_y")):
        lo, hi = ranges[key]
        v = cmd[..., i]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(v > hi, hi / v, np.where(v < lo, lo / v, 1.0))
        scale = np.minimum(scale, np.clip(s, 0.0, 1.0))
    cmd[..., :2] *= scale[..., None]
    lo, hi = ranges["ang_vel_z"]
    cmd[..., 2] = np.clip(cmd[..., 2], lo, hi)
    return cmd


def directional_command(y, yaw, lane_y, speed, yaw_des, *, k_y: float, vy_max: float,
                        k_yaw: float, wz_max: float, ranges: dict,
                        transition_deg: float = TRANSITION_DEG,
                        transition_speed_factor: float = 0.3) -> np.ndarray:
    """[N, 3] body commands that move robots along +x at ``speed`` (per robot) on their
    lane centre line while holding body yaw ``yaw_des`` (per robot). While the yaw error
    exceeds ``transition_deg`` the travel speed is scaled by ``transition_speed_factor``."""
    yaw = np.asarray(yaw, float)
    lateral = np.asarray(y, float) - np.asarray(lane_y, float)          # + = left of the line
    err = wrap_pi(np.asarray(yaw_des, float) - yaw)
    speed = np.broadcast_to(np.asarray(speed, float), yaw.shape)
    speed = np.where(np.abs(err) > math.radians(transition_deg), speed * transition_speed_factor,
                     speed)
    vwx, vwy = speed, np.clip(-k_y * lateral, -vy_max, vy_max)        # world frame
    c, s = np.cos(yaw), np.sin(yaw)
    cmd = np.stack([c * vwx + s * vwy, -s * vwx + c * vwy,             # R(-yaw) v_w
                    np.clip(k_yaw * err, -wz_max, wz_max)], axis=-1)
    return clip_planar(cmd, ranges)


def goal_reached(xy, goal, radius: float) -> np.ndarray:
    return np.linalg.norm(np.asarray(xy, float) - np.asarray(goal, float), axis=-1) <= radius


class _Attempts:
    """Per robot: status ("running" until the attempt ends) and end step (exclusive:
    steps [0, end) count). The first reset (a fall, or leaving the terrain) ends an attempt."""

    def __init__(self, n: int, limit_steps: int):
        self.status = np.array([RUNNING] * n, dtype=object)
        self.end = np.full(n, limit_steps, dtype=int)
        self.limit = limit_steps

    @property
    def running(self) -> np.ndarray:
        return self.status == RUNNING

    @property
    def done(self) -> bool:
        return not self.running.any()

    def _end(self, mask: np.ndarray, status: str, end: int) -> None:
        mask = mask & self.running
        self.status[mask] = status
        self.end[mask] = end

    def resets(self, t: int, terminated, time_out) -> None:
        # the step that reset a robot already shows it at its start: it does not count
        self._end(np.asarray(terminated, bool), "fell", t)
        self._end(np.asarray(time_out, bool), "out_of_bounds", t)

    def time_limit(self, t: int, status: str = "timeout") -> None:
        if t + 1 >= self.limit:
            self._end(np.ones_like(self.running), status, t + 1)


class GauntletController:
    """Lane following on the gauntlet (and each ladder). Positions in world coordinates.

    A course whose sections set a ``mode`` or ``speed_m_s`` (C.is_directional) is driven
    with ``directional_command`` instead of ``lane_command``: each robot holds its current
    section's mode yaw at the section's speed, and spin sections turn on the spot first.
    Other gauntlets keep the plain lane-following controller, so their results stay
    comparable with earlier runs.

    With ``outcome: {stuck_window_s, stuck_progress_m}`` in the course (ladders), an
    attempt whose furthest x grows less than stuck_progress_m in the window ends "stuck".
    """

    def __init__(self, course: dict, num_envs: int, ranges: dict, dt: float,
                 speed: float | None = None, time_limit_s: float | None = None):
        self.course, self.ranges, self.dt = course, ranges, dt
        frame = C.course_frame(course, num_envs)
        self.x0, self.lane_y = frame["x0"], np.array(frame["lane_y"])
        self.gains = dict(course["controller"])
        if speed is not None:
            self.gains["speed"] = float(speed)
        self.sections = C.gauntlet_sections(course)
        self._ends = np.array([s.x1 for s in self.sections])
        self.length = self.sections[-1].x1
        self.finish_x = self.length - float(course.get("finish_margin_m", 1.0))
        self.directional = C.is_directional(course)
        self.transition_deg = float(self.gains.get("transition_deg", TRANSITION_DEG))
        self.yaw_des = np.array([mode_yaw(s.mode) for s in self.sections])
        self.speeds = np.array([self.section_speed(s) for s in self.sections])
        lo, hi = ranges["ang_vel_z"]
        self.spin = {}                  # section index -> per-robot spin state
        for s in self.sections:
            if s.mode != "spin":
                continue
            deg = float(s.params["spin_deg"])
            self.spin[s.index] = {
                "target": math.radians(abs(deg)), "sign": 1.0 if deg >= 0 else -1.0,
                "rate": float(np.clip(math.copysign(float(self.gains["wz_max"]), deg), lo, hi)),
                "start": np.full(num_envs, -1), "end": np.full(num_envs, -1),
                "turned": np.zeros(num_envs), "prev": np.zeros(num_envs),
                "start_xy": np.full((num_envs, 2), np.nan), "end_xy": np.full((num_envs, 2), np.nan)}
        if time_limit_s is None:
            factor = float(course.get("time_limit_factor", 3.0))
            if self.directional:
                travel = sum(s.length / v for s, v in zip(self.sections, self.speeds))
                travel += sum(st["target"] / max(abs(st["rate"]), 1e-6) for st in self.spin.values())
                time_limit_s = factor * travel
            else:
                time_limit_s = factor * self.length / self.gains["speed"]
        self.time_limit_s = time_limit_s
        self.attempts = _Attempts(num_envs, max(int(round(self.time_limit_s / dt)), 1))
        out = course.get("outcome") or {}
        self.stuck_steps = (int(round(float(out["stuck_window_s"]) / dt))
                            if "stuck_window_s" in out else None)
        self.stuck_progress = float(out.get("stuck_progress_m", 0.3))
        self._xmax_log: list[np.ndarray] = []
        self._calls = 0                 # command() calls = the rollout step being commanded

    def section_speed(self, s: C.Section) -> float:
        """Travel speed along +x in section ``s``: its speed_m_s, else the mode's default
        (forward and spin: the controller's speed; others: controller.mode_speeds or
        MODE_SPEEDS)."""
        if "speed_m_s" in s.params:
            return float(s.params["speed_m_s"])
        if s.mode in ("forward", "spin"):
            return float(self.gains["speed"])
        return float((self.gains.get("mode_speeds") or {}).get(s.mode, MODE_SPEEDS[s.mode]))

    def section_at(self, x) -> np.ndarray:
        """Index of the section at course coordinate x (clipped to the course)."""
        return np.clip(np.searchsorted(self._ends, np.asarray(x, float), side="right"),
                       0, len(self.sections) - 1)

    def start_pose(self):
        return None                     # mjlab's lane origins, facing +x

    def _track_spins(self, pos: np.ndarray, yaw: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Advance every spin section's state; returns which robots are spinning now. A
        spin starts when the robot is first in the section and ends once the accumulated
        (unwrapped) yaw has turned spin_deg in the commanded direction."""
        spinning = np.zeros(len(yaw), bool)
        for k, st in self.spin.items():
            new = (idx == k) & (st["start"] < 0) & self.attempts.running
            st["start"][new] = self._calls
            st["start_xy"][new] = pos[new, :2]
            st["prev"][new] = yaw[new]
            active = (st["start"] >= 0) & (st["end"] < 0)
            st["turned"][active] += wrap_pi(yaw[active] - st["prev"][active])
            st["prev"][active] = yaw[active]
            done = active & (st["sign"] * st["turned"] >= st["target"])
            st["end"][done] = self._calls
            st["end_xy"][done] = pos[done, :2]
            spinning |= active & ~done
        return spinning

    def command(self, pos: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        g = self.gains
        if not self.directional:
            cmd = lane_command(pos[:, 1], yaw, self.lane_y, speed=g["speed"], k_y=g["k_y"],
                               vy_max=g["vy_max"], k_yaw=g["k_yaw"], wz_max=g["wz_max"],
                               ranges=self.ranges)
        else:
            idx = self.section_at(pos[:, 0] - self.x0)
            spinning = self._track_spins(pos, yaw, idx)
            cmd = directional_command(
                pos[:, 1], yaw, self.lane_y, self.speeds[idx], self.yaw_des[idx], k_y=g["k_y"],
                vy_max=g["vy_max"], k_yaw=g["k_yaw"], wz_max=g["wz_max"], ranges=self.ranges,
                transition_deg=self.transition_deg,
                transition_speed_factor=float(g.get("transition_speed_factor", 0.3)))
            for k, st in self.spin.items():
                m = spinning & (st["start"] >= 0) & (st["end"] < 0)
                cmd[m] = [0.0, 0.0, st["rate"]]
        self._calls += 1
        cmd[~self.attempts.running] = 0.0
        return cmd

    def update(self, t: int, pos: np.ndarray, yaw: np.ndarray, terminated, time_out) -> bool:
        a = self.attempts
        a.resets(t, terminated, time_out)
        x = pos[:, 0] - self.x0
        a._end(x >= self.finish_x, "finished", t + 1)
        if self.stuck_steps is not None:
            xmax = np.maximum(self._xmax_log[-1], x) if self._xmax_log else np.asarray(x, float)
            self._xmax_log.append(xmax)
            if len(self._xmax_log) > self.stuck_steps:
                progress = xmax - self._xmax_log[-1 - self.stuck_steps]
                a._end(progress < self.stuck_progress, "stuck", t + 1)
        a.time_limit(t)
        return a.done


class BeaconController:
    """Homing on the beacon field. Positions in world coordinates."""

    def __init__(self, course: dict, starts_w: np.ndarray, goals_w: np.ndarray, ranges: dict,
                 dt: float, time_limit_s: float | None = None):
        self.course, self.ranges, self.dt = course, ranges, dt
        self.starts, self.goals = np.asarray(starts_w, float), np.asarray(goals_w, float)
        self.gains = dict(course["controller"])
        out = course.get("outcome") or {}
        self.goal_radius = float(out.get("goal_radius_m", 0.5))
        dist = float(np.linalg.norm(self.goals[:, :2] - self.starts[:, :2], axis=1).max())
        self.time_limit_s = (time_limit_s if time_limit_s is not None else
                             float(out.get("time_limit_factor", 3.0)) * dist / self.gains["v_max"]
                             + float(out.get("time_limit_extra_s", 10.0)))
        self.stuck_steps = int(round(float(out.get("stuck_window_s", 10.0)) / dt))
        self.stuck_progress = float(out.get("stuck_progress_m", 0.5))
        self.attempts = _Attempts(len(self.starts), max(int(round(self.time_limit_s / dt)), 1))
        self.dist_log: list[np.ndarray] = []

    def start_pose(self) -> np.ndarray:
        """[N, 7] world root poses (position z = ground; the caller adds the robot's
        nominal base height): facing the goal if ``start_facing_goal``."""
        d = self.goals[:, :2] - self.starts[:, :2]
        face = (self.course.get("episodes") or {}).get("start_facing_goal", True)
        yaw = np.arctan2(d[:, 1], d[:, 0]) if face else np.zeros(len(d))
        quat = np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], 1)
        return np.concatenate([self.starts, quat], axis=1)

    def command(self, pos: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        g = self.gains
        cmd = beacon_command(pos[:, :2], yaw, self.goals[:, :2], v_max=g["v_max"], k_yaw=g["k_yaw"],
                             wz_max=g["wz_max"], slow_radius_m=g["slow_radius_m"], ranges=self.ranges)
        cmd[~self.attempts.running] = 0.0
        return cmd

    def update(self, t: int, pos: np.ndarray, yaw: np.ndarray, terminated, time_out) -> bool:
        a = self.attempts
        a.resets(t, terminated, time_out)
        dist = np.linalg.norm(pos[:, :2] - self.goals[:, :2], axis=1)
        self.dist_log.append(dist)
        a._end(dist <= self.goal_radius, "success", t + 1)
        if t >= self.stuck_steps:
            progress = self.dist_log[t - self.stuck_steps] - dist
            a._end(progress < self.stuck_progress, "stuck", t + 1)
        a.time_limit(t)
        return a.done


class EnvDriver:
    """Connects a controller to record_rollout (evaluation/rollout.py): writes start
    poses after the reset, commands before each step, and reports when every attempt
    has ended."""

    def __init__(self, controller, base_height: float):
        self.ctrl, self.base_height = controller, base_height

    def _pose(self, env):
        robot = env.scene["robot"]
        pos = robot.data.root_link_pos_w.detach().cpu().numpy().astype(float)
        yaw = robot.data.heading_w.detach().cpu().numpy().astype(float)
        return pos, yaw

    def reset(self, env) -> None:
        import torch

        pose = self.ctrl.start_pose()
        if pose is None:
            return
        pose = np.array(pose, dtype=float)
        pose[:, 2] += self.base_height
        robot = env.scene["robot"]
        robot.write_root_link_pose_to_sim(torch.tensor(pose, dtype=torch.float, device=env.device))
        robot.write_root_link_velocity_to_sim(torch.zeros(len(pose), 6, device=env.device))
        env.sim.forward()

    def command(self, env):
        import torch

        pos, yaw = self._pose(env)
        return torch.tensor(self.ctrl.command(pos, yaw), dtype=torch.float, device=env.device)

    def after_step(self, env, t: int) -> bool:
        pos, yaw = self._pose(env)
        return self.ctrl.update(t, pos, yaw, env.reset_terminated.detach().cpu().numpy(),
                                env.reset_time_outs.detach().cpu().numpy())


# ---- metrics ------------------------------------------------------------------------

def _effort(rollout: dict, n: int, a: int, b: int, dt: float, mass: float, legs,
            continuous) -> dict:
    """Energy, cost of transport, jerk, RMS torque per joint kind and foot slip of robot
    n over steps [a, b)."""
    from quadruped_rl.evaluation.render import torque_stats

    out = {"energy_J": None, "cot": None, "jerk": None, "foot_slip": None}
    out.update({f"{k}_rms_Nm": None for k in JOINT_KINDS})
    out.update({f"{k}_over_continuous": None for k in JOINT_KINDS})
    if b - a < 1:
        return out
    tau = rollout["torques"][a:b, n].astype(float)
    qd = rollout["joint_vel"][a:b, n].astype(float)
    out["energy_J"] = metrics.mechanical_energy(tau, qd, dt)
    cot = metrics.cost_of_transport(tau, qd, rollout["base_lin_vel_b"][a:b, n, :2], mass)
    out["cot"] = cot if math.isfinite(cot) else None
    if b - a >= 3:
        out["jerk"] = metrics.joint_jerk(qd, dt)
    out["foot_slip"] = metrics.foot_slip(rollout["foot_vel_w"][a:b, n, :, :2],
                                         rollout["foot_contact"][a:b, n])
    stats = torque_stats(rollout["torques"][a:b, n:n + 1], np.ones((b - a, 1), bool), list(legs),
                         continuous)
    if stats:
        for k in JOINT_KINDS:
            out[f"{k}_rms_Nm"] = stats[k]["rms"]
            out[f"{k}_over_continuous"] = stats[k].get("over_continuous")
    return out


def _first(mask: np.ndarray, start: int = 0) -> int | None:
    idx = np.flatnonzero(mask[start:])
    return int(idx[0]) + start if idx.size else None


def _heading_and_tracking(rollout: dict, n: int, a: int, b: int, err_deg: np.ndarray,
                          valid: np.ndarray, dt: float, threshold_deg: float) -> dict:
    """Heading error RMS (deg) and settle time (s with |error| > threshold) over the valid
    steps of [a, b), and the body-frame command tracking RMSE (planar m/s, yaw rad/s)."""
    out = {"heading_rms_deg": None, "settle_s": None, "track_planar_rmse": None,
           "track_yaw_rmse": None}
    if b - a < 1:
        return out
    e = err_deg[a:b][valid[a:b]]
    if e.size:
        out["heading_rms_deg"] = float(np.sqrt(np.mean(e ** 2)))
        out["settle_s"] = float(np.sum(e > threshold_deg) * dt)
    if "command" in rollout:
        cmd = rollout["command"][a:b, n].astype(float)
        v = rollout["base_lin_vel_b"][a:b, n, :2].astype(float)
        out["track_planar_rmse"] = float(np.sqrt(np.mean(np.sum((v - cmd[:, :2]) ** 2, axis=-1))))
        if "base_ang_vel_b" in rollout:
            w = rollout["base_ang_vel_b"][a:b, n, 2].astype(float)
            out["track_yaw_rmse"] = float(np.sqrt(np.mean((w - cmd[:, 2]) ** 2)))
    return out


def gauntlet_results(rollout: dict, ctrl: GauntletController, mass: float, legs,
                     continuous=None) -> tuple[list[dict], list[dict]]:
    """(section rows: one per robot and section, robot rows: one per robot).

    An attempt ends in the furthest section it had entered (its base's furthest x), not
    where its base was at the last step: a robot that climbs onto a curb and then tips
    back off it ends (and falls) at the curb. Heading error is measured against the yaw
    the controller asks for in the section the robot is in (0 outside directional
    courses), leaving out the time a spin section spends turning on the spot.
    """
    dt = float(rollout["dt"])
    sections = C.gauntlet_sections(ctrl.course)
    ends = np.array([s.x1 for s in sections])
    yaw_des = np.array([mode_yaw(s.mode) for s in sections])
    threshold = float(getattr(ctrl, "transition_deg", TRANSITION_DEG))
    spin = getattr(ctrl, "spin", {})
    status, end = ctrl.attempts.status, ctrl.attempts.end
    rows, robots = [], []
    for n in range(rollout["qpos"].shape[1]):
        T = int(min(end[n], rollout["qpos"].shape[0]))
        x = rollout["qpos"][:T, n, 0].astype(float) - ctrl.x0
        y = rollout["qpos"][:T, n, 1].astype(float) - ctrl.lane_y[n]
        yaw = yaw_from_quat(rollout["qpos"][:T, n, 3:7].astype(float))
        idx = np.clip(np.searchsorted(ends, x, side="right"), 0, len(sections) - 1)
        err = np.degrees(np.abs(wrap_pi(yaw - yaw_des[idx])))
        valid = np.ones(T, bool)
        spun = {}
        for k, st in spin.items():
            a, b = int(st["start"][n]), int(st["end"][n])
            if a >= 0:
                valid[a:(b if b >= 0 else T)] = False
            done = a >= 0 and b >= 0
            spun[k] = {"spin_done": done, "spin_time_s": (b - a) * dt if done else None,
                       "spin_drift_m": float(np.linalg.norm(st["end_xy"][n] - st["start_xy"][n]))
                       if done else None}
        fell = status[n] in FALL_STATUSES
        # the furthest section entered (not the one under the base at the last step)
        here = sections[min(int(np.searchsorted(ends, x.max(), side="right")),
                            len(sections) - 1)] if T else None
        robot = {"robot": n, "status": status[n], "finished": status[n] == "finished",
                 "total_time_s": T * dt if status[n] == "finished" else None,
                 "x_reached_m": float(x.max()) if T else 0.0, "fell": fell,
                 "end_section": here.label if here else None,
                 "fall_section": here.label if fell and here else None,
                 "fall_x_m": float(x[-1]) if fell and T else None,
                 "fall_time_s": T * dt if fell else None}
        robots.append(robot)
        for s in sections:
            b_eff = min(s.x1, ctrl.finish_x) if s is sections[-1] else s.x1
            enter = _first(x >= s.x0) if T else None
            exit_ = _first(x >= b_eff, enter) if enter is not None else None
            row = {"robot": n, "section": s.index, "label": s.label, "type": s.type,
                   "friction": s.friction, "mode": s.mode, "entered": enter is not None,
                   "exited": exit_ is not None, "time_s": None, "mean_speed": None,
                   "lateral_rms_m": None}
            stop = exit_ if exit_ is not None else T
            if enter is not None and stop > enter:
                row["time_s"] = (stop - enter) * dt
                if stop - enter >= 2:
                    row["mean_speed"] = float((x[stop - 1] - x[enter]) / ((stop - 1 - enter) * dt))
                row["lateral_rms_m"] = float(np.sqrt(np.mean(y[enter:stop] ** 2)))
                row.update(_heading_and_tracking(rollout, n, enter, stop, err, valid, dt, threshold))
                row.update(_effort(rollout, n, enter, stop, dt, mass, legs, continuous))
            else:
                row.update(_heading_and_tracking(rollout, n, 0, 0, err, valid, dt, threshold))
                row.update(_effort(rollout, n, 0, 0, dt, mass, legs, continuous))
            if s.index in spun:
                row.update(spun[s.index])
            row.update({f"robot_{k}": v for k, v in robot.items() if k != "robot"})
            rows.append(row)
    return rows, robots


def beacon_results(rollout: dict, ctrl: BeaconController, fld: C.BeaconField, frame: dict,
                   mass: float, legs, continuous=None) -> list[dict]:
    """One row per attempt."""
    dt = float(rollout["dt"])
    corner = np.array([frame["x0"], frame["y0"]])
    rows = []
    for n in range(rollout["qpos"].shape[1]):
        T = int(min(ctrl.attempts.end[n], rollout["qpos"].shape[0]))
        start, goal = ctrl.starts[n, :2] - corner, ctrl.goals[n, :2] - corner
        xy = rollout["qpos"][:T, n, :2].astype(float) - corner
        path = np.concatenate([start[None], xy]) if T else start[None]
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        moved = float(np.linalg.norm(path[-1] - path[0]))
        fric, slope = fld.friction_at(path[:, 0], path[:, 1]), fld.slope_deg(path[:, 0], path[:, 1])
        status = ctrl.attempts.status[n]
        row = {"robot": n, "start_x": float(start[0]), "start_y": float(start[1]),
               "goal_x": float(goal[0]), "goal_y": float(goal[1]),
               "goal_distance_m": float(np.linalg.norm(goal - start)), "outcome": status,
               "time_to_goal_s": T * dt if status == "success" else None,
               "duration_s": T * dt, "path_length_m": length,
               "path_efficiency": moved / length if length > 1e-6 else None,
               "mean_speed": length / (T * dt) if T else None,
               "min_goal_distance_m": float(np.linalg.norm(path - goal, axis=1).min()),
               "friction_mean": float(fric.mean()), "friction_max": float(fric.max()),
               "slope_mean_deg": float(slope.mean()), "slope_max_deg": float(slope.max())}
        row.update(_effort(rollout, n, 0, T, dt, mass, legs, continuous))
        rows.append(row)
    return rows


# ---- aggregation and summary --------------------------------------------------------

def _vals(rows, key):
    return [r[key] for r in rows if r.get(key) is not None]


def _mean(v):
    return float(np.mean(v)) if len(v) else None


def _median(v):
    return float(np.median(v)) if len(v) else None


def attempt_end_sections(rows: list[dict]) -> dict[tuple, tuple[int, str]]:
    """Per attempt ((seed, robot)): (index of the furthest section it entered, its status).
    This is the section the attempt ended in, so for every section
    entered = done + falls + t/o (attempts that ended there by timeout or getting stuck)."""
    out: dict[tuple, tuple[int, str]] = {}
    for r in rows:
        if r["entered"]:
            key = (r.get("seed"), r["robot"])
            status = r.get("robot_status") or ("fell" if r.get("robot_fell") else "")
            out[key] = (max(out.get(key, (-1, ""))[0], r["section"]), status)
    return out


def aggregate_gauntlet(rows: list[dict]) -> list[dict]:
    """Per section, over robots (and seeds): completion rate of those that entered,
    median time of those that got through, mean speed, cost of transport, the falls and
    timeouts that ended there (``attempt_end_sections``), and, for directional courses,
    heading error, command tracking and settle time."""
    ends = attempt_end_sections(rows)
    out = []
    for idx in sorted({r["section"] for r in rows}):
        sec = [r for r in rows if r["section"] == idx]
        entered = [r for r in sec if r["entered"]]
        done = [r for r in entered if r["exited"]]
        ended = [status for i, status in ends.values() if i == idx]
        spins = [r for r in sec if r.get("spin_done") is not None]
        out.append({"section": idx, "label": sec[0]["label"], "friction": sec[0]["friction"],
                    "mode": sec[0].get("mode") or "forward",
                    "entered": len(entered), "completed": len(done),
                    "completion": len(done) / len(entered) if entered else None,
                    "median_time_s": _median(_vals(done, "time_s")),
                    "mean_speed": _mean(_vals(entered, "mean_speed")),
                    "cot": _mean(_vals(done, "cot")),
                    "falls": sum(st in FALL_STATUSES for st in ended),
                    "timeouts": sum(st in STOP_STATUSES for st in ended),
                    "heading_rms_deg": _mean(_vals(entered, "heading_rms_deg")),
                    "track_planar_rmse": _mean(_vals(entered, "track_planar_rmse")),
                    "track_yaw_rmse": _mean(_vals(entered, "track_yaw_rmse")),
                    "settle_s": _mean(_vals(entered, "settle_s")),
                    "spins": len(spins), "spins_done": sum(bool(r["spin_done"]) for r in spins),
                    "spin_time_s": _median(_vals(spins, "spin_time_s")),
                    "spin_drift_m": _median(_vals(spins, "spin_drift_m"))})
    return out


def furthest_levels(rows: list[dict]) -> list[int]:
    """Per attempt, the highest level k of the ``L<k> ...`` sections it entered (0: none);
    empty when the course has no levelled sections."""
    if not any(C.section_level(r["label"]) for r in rows):
        return []
    best: dict[tuple, int] = {}
    for r in rows:
        key = (r.get("seed"), r["robot"])
        k = C.section_level(r["label"]) or 0
        best[key] = max(best.get(key, 0), k if r["entered"] else 0)
    return list(best.values())


def lower_median(v) -> int | float | None:
    """The lower of the two middle values (a value that occurs, for an even count)."""
    v = sorted(v)
    return v[(len(v) - 1) // 2] if v else None


def aggregate_beacon(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["outcome"] == "success"]
    counts = {k: sum(r["outcome"] == k for r in rows)
              for k in ("fell", "timeout", "stuck", "out_of_bounds")}
    return {"attempts": len(rows), "success_rate": len(ok) / len(rows) if rows else None,
            "median_time_to_goal_s": _median(_vals(ok, "time_to_goal_s")),
            "mean_path_efficiency": _mean(_vals(ok, "path_efficiency")),
            "mean_cot": _mean(_vals(ok, "cot")), **counts}


def _f(v, fmt: str, width: int) -> str:
    return f"{'-':>{width}}" if v is None else f"{v:>{width}{fmt}}"


END_RULE = ("falls / t/o: attempts that ended in this section by a fall (or leaving the "
            "terrain) / by timeout or getting stuck. An attempt ends in the furthest section it "
            "had entered, so entered = done + falls + t/o.")


def format_gauntlet(run_label: str, rows: list[dict], robots: list[dict]) -> str:
    agg = aggregate_gauntlet(rows)
    finished = sum(r["finished"] for r in robots)
    times = [r["total_time_s"] for r in robots if r["total_time_s"] is not None]
    directional = any(a["mode"] != "forward" for a in agg)
    lines = [f"Run: {run_label}",
             (f"{len(robots)} attempts; finished {finished}; median time "
              f"{_f(_median(times), '.1f', 0)} s; falls {sum(r['fell'] for r in robots)}")]
    levels = furthest_levels(rows)
    if levels:
        top = max(C.section_level(r["label"]) or 0 for r in rows)
        lines.append(f"furthest level reached: median L{lower_median(levels)} of L{top} "
                     f"(per attempt: {', '.join(f'L{k}' for k in sorted(levels))})")
    lines.append("")
    if directional:
        lines.append(f"{'section':<20}{'mode':>9}{'entered':>9}{'done':>6}{'rate':>7}"
                     f"{'median s':>10}{'speed':>7}{'hdg deg':>9}{'v rmse':>8}{'wz rmse':>9}"
                     f"{'settle s':>10}{'falls':>7}{'t/o':>5}")
        for a in agg:
            lines.append(f"{a['label']:<20}{a['mode']:>9}{a['entered']:>9}{a['completed']:>6}"
                         f"{_f(a['completion'], '.0%', 7)}{_f(a['median_time_s'], '.1f', 10)}"
                         f"{_f(a['mean_speed'], '.2f', 7)}{_f(a['heading_rms_deg'], '.1f', 9)}"
                         f"{_f(a['track_planar_rmse'], '.2f', 8)}{_f(a['track_yaw_rmse'], '.2f', 9)}"
                         f"{_f(a['settle_s'], '.1f', 10)}{a['falls']:>7}{a['timeouts']:>5}")
        for a in agg:
            if a["spins"]:
                lines.append(f"spin '{a['label']}': completed {a['spins_done']}/{a['spins']}; "
                             f"median spin time {_f(a['spin_time_s'], '.1f', 0)} s, median drift "
                             f"{_f(a['spin_drift_m'], '.2f', 0)} m (spin start to spin end)")
    else:
        lines.append(f"{'section':<20}{'mu':>5}{'entered':>9}{'done':>6}{'rate':>7}{'median s':>10}"
                     f"{'speed':>7}{'CoT':>7}{'falls':>7}{'t/o':>5}")
        for a in agg:
            lines.append(f"{a['label']:<20}{a['friction']:>5.2f}{a['entered']:>9}{a['completed']:>6}"
                         f"{_f(a['completion'], '.0%', 7)}{_f(a['median_time_s'], '.1f', 10)}"
                         f"{_f(a['mean_speed'], '.2f', 7)}{_f(a['cot'], '.2f', 7)}{a['falls']:>7}"
                         f"{a['timeouts']:>5}")
    lines += ["", ("rate: completed / entered;  median s: time in the section of those that got "
                   "through;  speed: mean forward m/s;  " +
                   ("hdg deg: RMS error to the mode's body yaw (spins excluded);  v rmse / wz rmse: "
                    "body-frame command tracking RMSE, m/s and rad/s;  settle s: mean time with "
                    "heading error above 15 deg." if directional else
                    "CoT: mechanical cost of transport.")),
              END_RULE]
    return "\n".join(lines) + "\n"


def format_beacon(run_label: str, rows: list[dict]) -> str:
    a = aggregate_beacon(rows)
    lines = [f"Run: {run_label}",
             f"  attempts               {a['attempts']}",
             f"  success rate           {_f(a['success_rate'], '.0%', 0)}",
             f"  median time to goal    {_f(a['median_time_to_goal_s'], '.1f', 0)} s (successes)",
             f"  mean path efficiency   {_f(a['mean_path_efficiency'], '.3f', 0)} (successes)",
             f"  mean cost of transport {_f(a['mean_cot'], '.2f', 0)} (successes)",
             f"  failures               fell {a['fell']}, timeout {a['timeout']}, "
             f"stuck {a['stuck']}" + (f", left the terrain {a['out_of_bounds']}"
                                      if a["out_of_bounds"] else "")]
    return "\n".join(lines) + "\n"


def compare_gauntlet(per_run: dict[str, list[dict]]) -> str:
    names = list(per_run)
    aggs = {k: aggregate_gauntlet(v) for k, v in per_run.items()}
    w = max(16, *(len(n) + 2 for n in names)) if names else 16
    lines = ["Comparison: completion rate / median seconds per section", f"{'section':<20}"
             + "".join(f"{n:>{w}}" for n in names)]
    for i, sec in enumerate(aggs[names[0]]):
        cells = []
        for n in names:
            a = aggs[n][i]
            rate = "-" if a["completion"] is None else f"{a['completion']:.0%}"
            t = "-" if a["median_time_s"] is None else f"{a['median_time_s']:.1f}"
            cells.append(f"{rate + ' / ' + t:>{w}}")
        lines.append(f"{sec['label']:<20}" + "".join(cells))
    return "\n".join(lines) + "\n"


def compare_beacon(per_run: dict[str, list[dict]]) -> str:
    names = list(per_run)
    aggs = {k: aggregate_beacon(v) for k, v in per_run.items()}
    w = max(14, *(len(n) + 2 for n in names)) if names else 14
    items = [("success rate", "success_rate", ".0%"), ("median time s", "median_time_to_goal_s", ".1f"),
             ("path efficiency", "mean_path_efficiency", ".3f"), ("cost of transport", "mean_cot", ".2f"),
             ("fell", "fell", "d"), ("timeout", "timeout", "d"), ("stuck", "stuck", "d")]
    lines = ["Comparison", f"{'':<20}" + "".join(f"{n:>{w}}" for n in names)]
    for label, key, fmt in items:
        lines.append(f"{label:<20}" + "".join(_f(aggs[n][key], fmt, w) for n in names))
    return "\n".join(lines) + "\n"


# ---- ladders -----------------------------------------------------------------------

def format_value(value, unit: str) -> str:
    """A ladder level's value in physical units: 15 deg, 0.10 m, mu 0.35."""
    if value is None:
        return "-"
    if unit == "deg":
        return f"{value:g} deg"
    if unit == "m":
        return f"{value:.2f} m"
    if unit == "mu":
        return f"mu {value:g}"
    return f"{value:g}{(' ' + unit) if unit else ''}"


def ladder_scores(rows: list[dict], ladder_course: dict) -> list[dict]:
    """One row per attempt on one ladder (``rows`` from gauntlet_results on
    ``C.ladder_course``): the score is the highest level k whose sections (feature parts
    and rest) it fully exited, 0 if it did not get through L1; plus the section the
    attempt ended in and how (fell, out_of_bounds, timeout, stuck, finished)."""
    lad = ladder_course["ladder"]
    levels, top = list(lad["levels"]), len(lad["levels"])
    by: dict[tuple, list[dict]] = {}
    for r in rows:
        by.setdefault((r.get("seed"), r["robot"]), []).append(r)
    out = []
    for (seed, robot), rs in by.items():
        exited: dict[int, bool] = {}
        for r in rs:
            k = C.section_level(r["label"])
            if k:
                exited[k] = exited.get(k, True) and bool(r["exited"])
        score = 0
        for k in range(1, top + 1):
            if not exited.get(k):
                break
            score = k
        out.append({"seed": seed, "robot": robot, "ladder": lad["name"], "score": score,
                    "top_level": top, "value": levels[score - 1] if score else None,
                    "unit": lad["unit"], "end_section": rs[0].get("robot_end_section"),
                    "outcome": rs[0].get("robot_status")})
    return out


def aggregate_ladder(scores: list[dict], levels: list[float], unit: str) -> dict:
    """Over the attempts on one ladder: median level (the lower median, so a level at least
    half the robots completed) and its value, interquartile range, the fraction completing
    each level, and the most common section where attempts ended short of the finish."""
    s = np.array([r["score"] for r in scores], float)
    med = lower_median([int(v) for v in s])
    fails = [r for r in scores if r["outcome"] != "finished"]
    common, how = None, None
    if fails:
        counts: dict[str, int] = {}
        for r in fails:
            counts[r["end_section"]] = counts.get(r["end_section"], 0) + 1
        common = max(counts, key=counts.get)          # first seen wins a tie
        kinds: dict[str, int] = {}
        for r in fails:
            if r["end_section"] == common:
                kinds[r["outcome"]] = kinds.get(r["outcome"], 0) + 1
        how = ", ".join(f"{k} {v}" for k, v in kinds.items())
    return {"attempts": len(scores), "top_level": len(levels), "median": med,
            "median_value": levels[med - 1] if med else None, "unit": unit,
            "q25": float(np.percentile(s, 25)) if s.size else None,
            "q75": float(np.percentile(s, 75)) if s.size else None,
            "completing": [float(np.mean(s >= k)) if s.size else 0.0
                           for k in range(1, len(levels) + 1)],
            "finished": sum(r["outcome"] == "finished" for r in scores),
            "common_failure": common, "common_failure_how": how}


def ladder_profile(per_run: dict[str, list[dict]], course: dict) -> dict[str, dict[str, dict]]:
    """{run: {ladder: aggregate_ladder}} for the ladders each run was scored on."""
    out = {}
    for run, scores in per_run.items():
        out[run] = {}
        for name in C.ladder_names(course):
            sc = [r for r in scores if r["ladder"] == name]
            if sc:
                lad = course["ladders"][name]
                out[run][name] = aggregate_ladder(sc, [float(v) for v in lad["levels"]],
                                                  str(lad.get("unit", "")))
    return out


def format_ladders(per_run: dict[str, list[dict]], course: dict) -> str:
    """The capability profile (runs x ladders: median level and its value), then one block
    per run with each ladder's IQR, completion per level and most common failure."""
    prof = ladder_profile(per_run, course)
    names = [n for n in C.ladder_names(course) if any(n in p for p in prof.values())]
    w = max(18, *(len(n) + 2 for n in names))
    rw = max(12, *(len(r) + 2 for r in prof))
    lines = ["Capability profile: median level (its value); an attempt's score is the highest "
             "level it fully exited",
             f"{'run':<{rw}}" + "".join(f"{n:>{w}}" for n in names)]
    for run, p in prof.items():
        cells = []
        for n in names:
            a = p.get(n)
            cell = "-" if a is None else \
                f"{a['median']} ({format_value(a['median_value'], a['unit'])})"
            cells.append(f"{cell:>{w}}")
        lines.append(f"{run:<{rw}}" + "".join(cells))
    lines.append("")
    for run, p in prof.items():
        lines += [f"Run: {run}",
                  f"{'ladder':<18}{'robots':>7}{'median':>9}{'value':>11}{'IQR':>10}"
                  f"{'done':>6}  {'completing L1, L2, ...':<36}most common end short of the finish"]
        for n in names:
            a = p.get(n)
            if a is None:
                continue
            iqr = f"{a['q25']:.1f}-{a['q75']:.1f}"
            comp = " ".join(f"{c:.0%}" for c in a["completing"])
            fail = (f"{a['common_failure']} ({a['common_failure_how']})" if a["common_failure"]
                    else "-")
            med = f"{a['median']}/{a['top_level']}"
            lines.append(f"{n:<18}{a['attempts']:>7}{med:>9}"
                         f"{format_value(a['median_value'], a['unit']):>11}{iqr:>10}"
                         f"{a['finished']:>6}  {comp:<36}{fail}")
        lines.append("")
    lines.append("median: lower median of the scores (a level at least half the attempts "
                 "completed) / top level; IQR: 25th-75th percentile of the scores; done: "
                 "attempts that finished the whole ladder.")
    return "\n".join(lines) + "\n"


# Okabe-Ito order (colourblind-safe), one colour per run, in a fixed order
RUN_COLOURS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442",
               "#000000")


def plot_ladder_profile(per_run: dict[str, list[dict]], course: dict, path: Path) -> Path:
    """profile.png: one horizontal bar group per ladder, one bar per run, the median level
    as a fraction of the ladder's top level, the IQR as a whisker, the median level and
    its physical value as a label."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prof = ladder_profile(per_run, course)
    runs = list(prof)
    names = [n for n in C.ladder_names(course) if any(n in p for p in prof.values())]
    bar = 0.8 / max(len(runs), 1)
    fig, ax = plt.subplots(figsize=(11, 1.6 + 0.4 * len(names) * max(len(runs), 1)))
    for i, run in enumerate(runs):
        colour = RUN_COLOURS[i % len(RUN_COLOURS)]
        for j, n in enumerate(names):
            a = prof[run].get(n)
            if a is None:
                continue
            top = a["top_level"]
            yv = j + (i - (len(runs) - 1) / 2) * bar
            frac = a["median"] / top
            ax.barh(yv, frac, height=bar * 0.9, color=colour, label=run if j == 0 else None,
                    edgecolor="white", linewidth=1)
            ax.errorbar(frac, yv, xerr=[[frac - a["q25"] / top], [a["q75"] / top - frac]],
                        fmt="none", ecolor="#333333", elinewidth=1, capsize=3)
            ax.text(max(frac, a["q75"] / top) + 0.015, yv,
                    f"L{a['median']}/{top}  {format_value(a['median_value'], a['unit'])}",
                    va="center", fontsize=8, color="#222222")
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "top level"])
    ax.set_xlabel("median level completed, as a fraction of the ladder's top level "
                  "(whisker: interquartile range)")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Capability profile on the ladders")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False,
              title="run", title_fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_csv(rows: list[dict], path: Path) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else
                            (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k])) for k in keys})


# ---- preview ------------------------------------------------------------------------

SHORT_SECTION_M = 1.0          # previews: shorter sections are numbered and listed in a key


def _section_tags(course: dict, sections: list) -> tuple[list[str], list[str]]:
    """Preview labels: the label of each section (with its friction when not 1 and, on
    directional courses, its mode), or "[k]" for sections shorter than SHORT_SECTION_M,
    whose full labels go into the returned key."""
    directional = C.is_directional(course)
    tags, key = [], []
    for s in sections:
        text = s.label
        if s.friction != 1.0:
            text += f" \u00b7 mu {s.friction:g}"
        if directional:
            text += f" \u00b7 {s.mode}" + (f" {float(s.params['spin_deg']):g} deg"
                                           if s.mode == "spin" else "")
        if s.length < SHORT_SECTION_M:
            key.append(f"[{len(key) + 1}] {text}")
            tags.append(f"[{len(key)}]")
        else:
            tags.append(text)
    return tags, key


def _key_text(key: list[str], per_line: int = 6) -> str:
    lines = ["   ".join(key[i:i + per_line]) for i in range(0, len(key), per_line)]
    return ("short sections: " + "\n".join(lines)) if key else ""


def preview_gauntlet(course: dict, out_dir: Path, num_lanes: int = 4) -> list[Path]:
    """Top view (lanes, sections coloured by friction, blocks and rough shaded by height)
    and side elevation (centre-line height) of the gauntlet."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    sections = C.gauntlet_sections(course)
    prims = C.lane_primitives(course)
    w = float(course.get("lane_width_m", 2.0))
    length = C.course_length(course)
    finish = length - float(course.get("finish_margin_m", 1.0))
    xs = np.arange(0.0, length, 0.02)
    ys = np.linspace(-w / 2 + 0.01, w / 2 - 0.01, 60)
    X, Y = np.meshgrid(xs, ys)
    Z = C.surface_height(prims, X.ravel(), Y.ravel()).reshape(X.shape)
    detail = Z - C.nominal_height(sections, X.ravel()).reshape(X.shape)
    paths = []
    tags, key = _section_tags(course, sections)
    key_lines = _key_text(key).count("\n") + 1 if key else 0

    # layout in inches: title, label band (labels at 60 deg), lanes (about equal aspect),
    # x axis, key of the short sections
    width = float(np.clip(0.16 * length, 18.0, 26.0))
    left, right, ax_w = 0.8, 0.3, width - 1.1
    map_h = max(1.5, ax_w * num_lanes * w / length)
    band_h, title_h, bottom_h = 1.6, 0.5, 0.6 + 0.2 * key_lines
    height = title_h + band_h + map_h + bottom_h
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((left / width, bottom_h / height, ax_w / width, map_h / height))
    band = fig.add_axes((left / width, (bottom_h + map_h) / height, ax_w / width, band_h / height))
    for lane in range(num_lanes):
        y0 = (lane - num_lanes / 2) * w
        for s in sections:
            ax.add_patch(plt.Rectangle((s.x0, y0), s.length, w, color=C.friction_rgba(s.friction)[:3],
                                       ec="white", lw=0.5))
        ax.imshow(np.clip(detail, -0.06, 0.06), extent=(0, length, y0, y0 + w), origin="lower",
                  cmap="RdBu_r", alpha=0.45, vmin=-0.06, vmax=0.06, aspect="auto", zorder=2)
        ax.axhline(y0 + w / 2, color="k", lw=0.5, ls=":")
    if C.is_directional(course):          # body heading per section: an arrow in every lane
        for lane in range(num_lanes):
            yc = (lane - num_lanes / 2) * w + w / 2
            for s in sections:
                xc = (s.x0 + s.x1) / 2
                if s.mode == "spin":
                    ax.text(xc, yc, "spin", ha="center", va="center", fontsize=8,
                            color="purple", zorder=4)
                    continue
                yaw = math.radians(C.MODE_YAW_DEG[s.mode])
                r = min(0.6, 0.4 * s.length)
                ax.annotate("", xy=(xc + r * math.cos(yaw), yc + r * math.sin(yaw)),
                            xytext=(xc - r * math.cos(yaw), yc - r * math.sin(yaw)),
                            arrowprops={"arrowstyle": "-|>", "color": "purple", "lw": 1.5},
                            zorder=4)
    spawn = float(course.get("spawn_x_m", 1.0))
    ax.axvline(spawn, color="green", lw=1.5, label="spawn")
    ax.axvline(finish, color="red", lw=1.5, label="finish line")
    band.set_xlim(0, length)
    band.set_ylim(0, 1)
    band.axis("off")
    for s, tag in zip(sections, tags):
        xc = (s.x0 + s.x1) / 2
        band.plot([xc, xc], [0, 0.04], color="k", lw=0.6)
        band.text(xc, 0.06, tag, rotation=60, rotation_mode="anchor", ha="left", va="center",
                  fontsize=7)
    ax.set_xlim(0, length)
    ax.set_ylim(-num_lanes * w / 2, num_lanes * w / 2)
    ax.set_xlabel("x along the course, m")
    ax.set_ylabel("y, m")
    fig.suptitle(f"Gauntlet top view ({num_lanes} of the lanes; blue tint = slippery, "
                 "red/blue shading = rough and blocks height"
                 + ("; purple arrows = body heading of each section's mode, travel is +x"
                    if C.is_directional(course) else "") + ")",
                 y=1 - 0.12 / height, va="top", fontsize=11)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper right", fontsize=8, frameon=False,
               bbox_to_anchor=(1 - 0.05 / width, 1 - 0.4 / height))
    if key:
        fig.text(left / width, 0.08 / height, _key_text(key), fontsize=7, va="bottom")
    paths.append(out_dir / "gauntlet_top.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(width, 4.5 + 0.15 * key_lines))
    xf = np.arange(0.0, length, 0.002)
    ytop = max(1.3, float(np.nanmax(Z)) + 0.8)      # room for the labels above the track
    for yv, style in ((0.0, "k-"), (0.5, "c-")):
        ax.plot(xf, C.surface_height(prims, xf, np.full_like(xf, yv)), style, lw=0.8,
                label=f"surface at y = {yv:+.1f} m")
    for s, tag in zip(sections, tags):
        ax.axvspan(s.x0, s.x1, color=C.friction_rgba(s.friction)[:3], alpha=0.35, lw=0)
        ax.axvline(s.x0, color="grey", lw=0.4)
        ax.text((s.x0 + s.x1) / 2, ytop - 0.05, tag, rotation=90, ha="center", va="top", fontsize=7)
    ax.axvline(finish, color="red", lw=1.0)
    ax.set_xlim(0, length)
    ax.set_ylim(min(-0.4, float(np.nanmin(Z)) - 0.1), ytop)
    ax.set_xlabel("x along the course, m")
    ax.set_ylabel("height, m")
    ax.set_title(f"Gauntlet side elevation ({length:.1f} m; finish line at {finish:.1f} m)")
    ax.legend(loc="lower left", fontsize=8)
    if key:
        fig.text(0.01, 0.01, _key_text(key), fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, 0.04 + 0.03 * key_lines if key else 0, 1, 1))
    paths.append(out_dir / "gauntlet_side.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)
    return paths


def preview_ladders(course: dict, out_dir: Path, tracks: list[str] | None = None) -> list[Path]:
    """ladders_side.png: the side elevation of each ladder (centre line), one subplot per
    ladder, levels shaded alternately and labelled with their value."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    names = tracks or C.ladder_names(course)
    fig, axes = plt.subplots(len(names), 1, figsize=(16, 2.4 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        lc = C.ladder_course(course, name)
        sections = C.gauntlet_sections(lc)
        prims = C.lane_primitives(lc)
        length = sections[-1].x1
        xf = np.arange(0.0, length, 0.005)
        z = C.surface_height(prims, xf, np.zeros_like(xf))
        lad = lc["ladder"]
        top = float(np.nanmax(z))
        for k, v in enumerate(lad["levels"], 1):
            secs = [s for s in sections if C.section_level(s.label) == k]
            a, b = secs[0].x0, secs[-1].x1
            ax.axvspan(a, b, color="#0072B2" if k % 2 else "#E69F00", alpha=0.12, lw=0)
            ax.text((a + b) / 2, top + 0.05, f"L{k}\n{format_value(v, lad['unit'])}",
                    ha="center", va="bottom", fontsize=7)
        bottom = min(-0.2, float(np.nanmin(z)) - 0.1)
        for s in sections:
            if s.friction < 1.0:
                ax.fill_between([s.x0, s.x1], bottom, bottom + 0.06,
                                color=C.friction_rgba(s.friction)[:3], lw=0)
        ax.plot(xf, z, "k-", lw=0.8)
        finish = length - float(lc.get("finish_margin_m", 1.0))
        ax.axvline(float(lc.get("spawn_x_m", 1.0)), color="green", lw=1)
        ax.axvline(finish, color="red", lw=1)
        ax.set_xlim(0, length)
        ax.set_ylim(bottom, top + 0.45)
        ax.set_ylabel("height, m")
        ax.set_title(f"ladder '{name}' ({length:.1f} m, {len(lad['levels'])} levels; strip at the "
                     "bottom = friction below 1, bluer = more slippery; green = spawn, red = "
                     "finish line)", fontsize=9, loc="left")
    axes[-1, 0].set_xlabel("x along the ladder, m")
    fig.tight_layout()
    path = out_dir / "ladders_side.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return [path]


def preview_beacon(course: dict, out_dir: Path, k: int, seed: int) -> list[Path]:
    """Top view coloured by height, tile outlines labelled with their friction (slippery
    tiles hatched), and the start and goal pairs of one episode seed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fld = C.beacon_field(course)
    starts, goals = C.beacon_pairs(course, k, seed)
    size, t = fld.size, fld.tile_size
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(fld.heights, extent=(0, size, 0, size), origin="lower", cmap="terrain")
    fig.colorbar(im, ax=ax, label="height, m", shrink=0.8)
    lo = float((course.get("field") or {}).get("friction_range", [0.4, 1.0])[0])
    for r in range(fld.tiles):
        for c in range(fld.tiles):
            mu = float(fld.friction[r, c])
            ax.add_patch(plt.Rectangle((c * t, r * t), t, t, fill=False, ec="k", lw=0.6,
                                       hatch="///" if mu < lo else None))
            ax.text(c * t + 0.3, r * t + 0.3, f"{mu:.2f}", fontsize=6, color="k")
    for i, (s, g) in enumerate(zip(starts, goals)):
        ax.annotate("", xy=g, xytext=s, arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.2})
        ax.plot(*s, "o", color="white", mec="k", ms=6)
        ax.plot(*g, "*", color="red", mec="k", ms=11)
        ax.text(s[0] + 0.4, s[1] + 0.4, str(i), fontsize=8, color="white",
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1, "lw": 0})
    ax.plot(*fld.rocks[:, :2].T, ".", color="saddlebrown", ms=1.5, label="rocks")
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_title(f"Beacon field (seed {course.get('seed')}): height, tile friction "
                 f"(hatched < {lo:g}), {k} start (o) -> goal (*) pairs of episode seed {seed}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = out_dir / "beacon_top.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]
