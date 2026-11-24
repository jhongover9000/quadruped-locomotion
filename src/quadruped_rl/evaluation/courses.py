"""Run velocity-command policies on the evaluation courses (envs/courses.py) and score them.

A shared high-level controller turns each robot's world pose and its goal into a
velocity command (vx, vy, wz); the policy under test only does locomotion, so
differences between policies come from the policy alone.

- gauntlet: lane following. vx = speed, vy = -k_y * lateral error, wz = k_yaw * heading
  error, clipped to the gains' limits and to the policy's trained command range.
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
    """Lane following on the gauntlet. Positions in world coordinates."""

    def __init__(self, course: dict, num_envs: int, ranges: dict, dt: float,
                 speed: float | None = None, time_limit_s: float | None = None):
        self.course, self.ranges, self.dt = course, ranges, dt
        frame = C.course_frame(course, num_envs)
        self.x0, self.lane_y = frame["x0"], np.array(frame["lane_y"])
        self.gains = dict(course["controller"])
        if speed is not None:
            self.gains["speed"] = float(speed)
        self.length = C.course_length(course)
        self.finish_x = self.length - float(course.get("finish_margin_m", 1.0))
        self.time_limit_s = (time_limit_s if time_limit_s is not None else
                             float(course.get("time_limit_factor", 3.0)) * self.length
                             / self.gains["speed"])
        self.attempts = _Attempts(num_envs, max(int(round(self.time_limit_s / dt)), 1))

    def start_pose(self):
        return None                     # mjlab's lane origins, facing +x

    def command(self, pos: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        g = self.gains
        cmd = lane_command(pos[:, 1], yaw, self.lane_y, speed=g["speed"], k_y=g["k_y"],
                           vy_max=g["vy_max"], k_yaw=g["k_yaw"], wz_max=g["wz_max"],
                           ranges=self.ranges)
        cmd[~self.attempts.running] = 0.0
        return cmd

    def update(self, t: int, pos: np.ndarray, yaw: np.ndarray, terminated, time_out) -> bool:
        a = self.attempts
        a.resets(t, terminated, time_out)
        a._end(pos[:, 0] - self.x0 >= self.finish_x, "finished", t + 1)
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


def gauntlet_results(rollout: dict, ctrl: GauntletController, mass: float, legs,
                     continuous=None) -> tuple[list[dict], list[dict]]:
    """(section rows: one per robot and section, robot rows: one per robot)."""
    dt = float(rollout["dt"])
    sections = C.gauntlet_sections(ctrl.course)
    status, end = ctrl.attempts.status, ctrl.attempts.end
    rows, robots = [], []
    for n in range(rollout["qpos"].shape[1]):
        T = int(min(end[n], rollout["qpos"].shape[0]))
        x = rollout["qpos"][:T, n, 0].astype(float) - ctrl.x0
        y = rollout["qpos"][:T, n, 1].astype(float) - ctrl.lane_y[n]
        fell = status[n] in ("fell", "out_of_bounds")
        ends = [s.x1 for s in sections]
        here = sections[min(int(np.searchsorted(ends, x[-1], side="right")), len(sections) - 1)] \
            if T else None
        robot = {"robot": n, "status": status[n], "finished": status[n] == "finished",
                 "total_time_s": T * dt if status[n] == "finished" else None,
                 "x_reached_m": float(x.max()) if T else 0.0, "fell": fell,
                 "fall_section": here.label if fell and here else None,
                 "fall_x_m": float(x[-1]) if fell and T else None,
                 "fall_time_s": T * dt if fell else None}
        robots.append(robot)
        for s in sections:
            b_eff = min(s.x1, ctrl.finish_x) if s is sections[-1] else s.x1
            enter = _first(x >= s.x0) if T else None
            exit_ = _first(x >= b_eff, enter) if enter is not None else None
            row = {"robot": n, "section": s.index, "label": s.label, "type": s.type,
                   "friction": s.friction, "entered": enter is not None, "exited": exit_ is not None,
                   "time_s": None, "mean_speed": None, "lateral_rms_m": None}
            stop = exit_ if exit_ is not None else T
            if enter is not None and stop > enter:
                row["time_s"] = (stop - enter) * dt
                if stop - enter >= 2:
                    row["mean_speed"] = float((x[stop - 1] - x[enter]) / ((stop - 1 - enter) * dt))
                row["lateral_rms_m"] = float(np.sqrt(np.mean(y[enter:stop] ** 2)))
                row.update(_effort(rollout, n, enter, stop, dt, mass, legs, continuous))
            else:
                row.update(_effort(rollout, n, 0, 0, dt, mass, legs, continuous))
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


def aggregate_gauntlet(rows: list[dict]) -> list[dict]:
    """Per section, over robots (and seeds): completion rate of those that entered,
    median time of those that got through, mean speed, cost of transport, falls."""
    out = []
    for idx in sorted({r["section"] for r in rows}):
        sec = [r for r in rows if r["section"] == idx]
        entered = [r for r in sec if r["entered"]]
        done = [r for r in entered if r["exited"]]
        out.append({"section": idx, "label": sec[0]["label"], "friction": sec[0]["friction"],
                    "entered": len(entered), "completed": len(done),
                    "completion": len(done) / len(entered) if entered else None,
                    "median_time_s": _median(_vals(done, "time_s")),
                    "mean_speed": _mean(_vals(entered, "mean_speed")),
                    "cot": _mean(_vals(done, "cot")),
                    "falls": sum(r["robot_fell"] and r["robot_fall_section"] == r["label"] for r in sec)})
    return out


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


def format_gauntlet(run_label: str, rows: list[dict], robots: list[dict]) -> str:
    agg = aggregate_gauntlet(rows)
    finished = sum(r["finished"] for r in robots)
    times = [r["total_time_s"] for r in robots if r["total_time_s"] is not None]
    lines = [f"Run: {run_label}",
             (f"{len(robots)} attempts; finished {finished}; median time "
              f"{_f(_median(times), '.1f', 0)} s; falls {sum(r['fell'] for r in robots)}"), "",
             (f"{'section':<20}{'mu':>5}{'entered':>9}{'done':>6}{'rate':>7}{'median s':>10}"
              f"{'speed':>7}{'CoT':>7}{'falls':>7}")]
    for a in agg:
        lines.append(f"{a['label']:<20}{a['friction']:>5.2f}{a['entered']:>9}{a['completed']:>6}"
                     f"{_f(a['completion'], '.0%', 7)}{_f(a['median_time_s'], '.1f', 10)}"
                     f"{_f(a['mean_speed'], '.2f', 7)}{_f(a['cot'], '.2f', 7)}{a['falls']:>7}")
    lines += ["", ("rate: completed / entered;  median s: time in the section of those that got "
                   "through;  speed: mean forward m/s;  CoT: mechanical cost of transport.")]
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

    fig, ax = plt.subplots(figsize=(18, 2.4 + 0.5 * num_lanes))
    for lane in range(num_lanes):
        y0 = (lane - num_lanes / 2) * w
        for s in sections:
            ax.add_patch(plt.Rectangle((s.x0, y0), s.length, w, color=C.friction_rgba(s.friction)[:3],
                                       ec="white", lw=0.5))
        ax.imshow(np.clip(detail, -0.06, 0.06), extent=(0, length, y0, y0 + w), origin="lower",
                  cmap="RdBu_r", alpha=0.45, vmin=-0.06, vmax=0.06, aspect="auto", zorder=2)
        ax.axhline(y0 + w / 2, color="k", lw=0.5, ls=":")
    spawn = float(course.get("spawn_x_m", 1.0))
    ax.axvline(spawn, color="green", lw=1.5, label="spawn")
    ax.axvline(finish, color="red", lw=1.5, label="finish line")
    for s in sections:
        ax.text((s.x0 + s.x1) / 2, num_lanes * w / 2 + 0.15, f"{s.label}\nmu {s.friction:g}",
                ha="center", va="bottom", fontsize=7, rotation=60)
    ax.set_xlim(0, length)
    ax.set_ylim(-num_lanes * w / 2, num_lanes * w / 2 + 3.2)
    ax.set_xlabel("x along the course, m")
    ax.set_ylabel("y, m")
    ax.set_title(f"Gauntlet top view ({num_lanes} of the lanes; blue tint = slippery, "
                 "red/blue shading = rough and blocks height)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    paths.append(out_dir / "gauntlet_top.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(18, 4.5))
    xf = np.arange(0.0, length, 0.002)
    for yv, style in ((0.0, "k-"), (0.5, "c-")):
        ax.plot(xf, C.surface_height(prims, xf, np.full_like(xf, yv)), style, lw=0.8,
                label=f"surface at y = {yv:+.1f} m")
    for s in sections:
        ax.axvspan(s.x0, s.x1, color=C.friction_rgba(s.friction)[:3], alpha=0.35, lw=0)
        ax.axvline(s.x0, color="grey", lw=0.4)
        ax.text((s.x0 + s.x1) / 2, 1.25, s.label, rotation=90,
                ha="center", va="top", fontsize=7)
    ax.axvline(finish, color="red", lw=1.0)
    ax.set_xlim(0, length)
    ax.set_ylim(min(-0.4, float(np.nanmin(Z)) - 0.1), 1.3)
    ax.set_xlabel("x along the course, m")
    ax.set_ylabel("height, m")
    ax.set_title(f"Gauntlet side elevation ({length:.1f} m; finish line at {finish:.1f} m)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    paths.append(out_dir / "gauntlet_side.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)
    return paths


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
