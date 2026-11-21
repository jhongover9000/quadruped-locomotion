"""Load a trained run and record what its policy does under a command schedule.

Used by scripts/play.py (live viewer) and scripts/record.py (videos, plots, rollout files).
Recording needs the GPU environment; everything recorded is plain numpy, so replaying and
rendering (envs/../evaluation/render.py) does not.

A *schedule* is a list of segments, each holding one velocity command [vx, vy, wz] for a
duration. During a segment every robot is given that command, the standing and resampling
logic of the task's command generator is switched off, and the policy runs as in play.py.

Tracking runs (SuperMimic) have no command schedule: each robot plays one reference clip
(``record_tracking_rollout``), and the reference takes the command's place in the recording.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# name -> [(seconds, vx m/s, vy m/s, wz rad/s, label)]
SCENARIOS: dict[str, list[tuple[float, float, float, float, str]]] = {
    "tour": [
        (2.0, 0.0, 0.0, 0.0, "stand"),
        (4.0, 0.5, 0.0, 0.0, "walk forward 0.5 m/s"),
        (4.0, 1.0, 0.0, 0.0, "forward 1.0 m/s"),
        (4.0, -0.5, 0.0, 0.0, "backward 0.5 m/s"),
        (3.0, 0.0, 0.4, 0.0, "sideways left 0.4 m/s"),
        (3.0, 0.0, -0.4, 0.0, "sideways right 0.4 m/s"),
        (3.0, 0.0, 0.0, 1.0, "turn left 1.0 rad/s"),
        (3.0, 0.0, 0.0, -1.0, "turn right 1.0 rad/s"),
        (4.0, 0.5, 0.0, 0.5, "forward + turn"),
        (2.0, 0.0, 0.0, 0.0, "stand"),
    ],
    "forward": [(2.0, 0.0, 0.0, 0.0, "stand"), (4.0, 0.5, 0.0, 0.0, "forward 0.5 m/s"),
                (4.0, 1.0, 0.0, 0.0, "forward 1.0 m/s"), (4.0, 1.5, 0.0, 0.0, "forward 1.5 m/s"),
                (2.0, 0.0, 0.0, 0.0, "stand")],
    "stand": [(10.0, 0.0, 0.0, 0.0, "stand")],
}


@dataclass
class Segment:
    duration_s: float
    command: tuple[float, float, float]
    label: str


def make_schedule(scenario: str = "tour", command: list[float] | None = None,
                  duration_s: float = 20.0) -> list[Segment] | None:
    """A fixed ``command`` wins over ``scenario``. Returns None for "random": the task's own
    command sampling is left in charge."""
    if command is not None:
        vx, vy, wz = (float(c) for c in command)
        return [Segment(duration_s, (vx, vy, wz), f"command vx {vx:+.2f} vy {vy:+.2f} wz {wz:+.2f}")]
    if scenario == "random":
        return None
    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{scenario}'. Available: {sorted(SCENARIOS) + ['random']}")
    return [Segment(d, (vx, vy, wz), label) for d, vx, vy, wz, label in SCENARIOS[scenario]]


def fixed_command_task(task: dict, command: list[float]) -> dict:
    """Task config whose command generator always samples ``command`` and never stands."""
    task = copy.deepcopy(task)
    vx, vy, wz = (float(c) for c in command)
    task["commands"].update(lin_vel_x=[vx, vx], lin_vel_y=[vy, vy], ang_vel_z=[wz, wz],
                            standing_fraction=0.0)
    return task


# ---- loading a run ------------------------------------------------------------------

def latest_checkpoint(run_dir: Path) -> Path:
    """The newest numbered checkpoint, whichever trainer wrote the run."""
    from quadruped_rl.algorithms.checkpoints import list_checkpoints

    found = list_checkpoints(run_dir)
    if not found:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    return found[-1]


def resolve_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    if checkpoint is None:
        return latest_checkpoint(run_dir)
    ckpt = Path(checkpoint)
    if not ckpt.is_absolute() and not ckpt.exists():
        for candidate in (run_dir / "checkpoints" / ckpt, run_dir / ckpt):
            if candidate.exists():
                return candidate
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint} not found in {run_dir}")
    return ckpt


@dataclass
class LoadedRun:
    env: object            # raw mjlab environment
    viewer_env: object     # what the policy steps (raw env, or rsl_rl's wrapper)
    policy: object         # obs (from viewer_env.get_observations()) -> actions
    robot: object
    task: dict
    run_dir: Path | None
    checkpoint: Path | None
    kind: str = "velocity"                 # "velocity" or "tracking" (algorithms/base.py)
    agent: object = None
    clips: list = field(default_factory=list)   # tracking: the clip files in the environment


def load_run(run_dir: Path | None, checkpoint: str | None = None, *, robot_name: str = "go1",
             task_name: str | None = None, num_envs: int = 16, device: str | None = None,
             task_override=None, seed: int | None = None, clips: list[str] | None = None,
             groups: str | None = None, terminations: bool = True) -> LoadedRun:
    """Build a play-mode environment and the run's policy (zero actions without a run).

    The task comes from the run's saved config unless ``task_name`` names another one of
    the same kind (for example, a flat-ground policy on rough_velocity).
    ``task_override(task) -> task`` edits the task config before the environment is built.
    For tracking runs, ``clips`` (names) or ``groups`` (manifest groups) choose the motions;
    by default the task's eval_groups, one clip per environment.
    """
    import torch

    from quadruped_rl.algorithms import get_algorithm
    from quadruped_rl.algorithms.base import task_kind
    from quadruped_rl.envs.builder import make_env
    from quadruped_rl.robots import load_robot, robot_from_dict
    from quadruped_rl.utils.config import load_group

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    algo_cfg = None
    if run_dir is not None:
        with open(Path(run_dir) / "config.yaml", encoding="utf-8") as f:
            saved = yaml.safe_load(f)
        robot, task, algo_cfg = robot_from_dict(saved["robot"]), saved["task"], saved["algo"]
    else:
        robot, task = load_robot(robot_name), load_group("tasks", task_name or "flat_velocity")
    if task_name is not None and run_dir is not None:
        other = load_group("tasks", task_name)
        if task_kind(other) != task_kind(task):
            raise ValueError(f"This run was trained on a {task_kind(task)} task "
                             f"('{task.get('name')}'); it cannot be played on the "
                             f"{task_kind(other)} task '{task_name}'.")
        task = other
    if task_override is not None:
        task = task_override(task)

    if algo_cfg is None:
        if task_kind(task) != "velocity":
            raise ValueError("Zero-action play needs a velocity task; for tracking, give a run.")
        env = make_env(robot, task, num_envs=num_envs, device=device, play=True, seed=seed)
        num_actions = env.action_manager.total_action_dim

        def policy(obs):
            return torch.zeros(env.num_envs, num_actions, device=device)
        return LoadedRun(env, env, policy, robot, task, None, None)

    ckpt = resolve_checkpoint(Path(run_dir), checkpoint)
    agent = get_algorithm(algo_cfg["name"])(robot=robot, task_cfg=task, algo_cfg=algo_cfg,
                                            run_dir=Path(run_dir), device=device)
    kind, clip_files = task_kind(task), []
    if kind == "tracking":
        clip_files = agent.clip_files(groups or task["motions"].get("eval_groups"), names=clips)
        if len(clip_files) > num_envs:           # spread over the library, not its first names
            step = len(clip_files) / num_envs
            clip_files = [clip_files[int(i * step)] for i in range(num_envs)]
        env = agent.make_env(min(num_envs, len(clip_files)), play=True, seed=seed,
                             clips=clip_files, terminations=terminations)
    else:
        env = agent.make_env(num_envs, play=True, seed=seed)
    viewer_env, policy = agent.make_policy(env, ckpt)
    return LoadedRun(env, viewer_env, policy, robot, task, Path(run_dir), ckpt, kind, agent,
                     clip_files)


# ---- recording ----------------------------------------------------------------------

def record_rollout(run: LoadedRun, schedule: list[Segment] | None, duration_s: float,
                   progress: bool = True) -> dict[str, np.ndarray]:
    """Step every robot for the schedule's length (or ``duration_s`` without one).

    Returns arrays with time along axis 0 and robot along axis 1. Values at step t are read
    after the t-th env.step(), so after a fall ``qpos`` already shows the reset robot and
    ``fell[t]`` marks it.
    """
    import mujoco
    import torch

    if run.kind == "tracking":
        return record_tracking_rollout(run, duration_s, progress)

    from quadruped_rl.envs.commands import COMMAND_NAME

    env, robot_cfg = run.env, run.robot
    dt = float(env.step_dt)
    total_s = sum(s.duration_s for s in schedule) if schedule else duration_s
    steps = round(total_s / dt)
    term = env.command_manager.get_term(COMMAND_NAME)
    robot = env.scene["robot"]
    site_ids = [robot.site_names.index(s) for s in robot_cfg.foot_sites]
    has_contacts = "feet_contact" in env.scene.sensors
    # foot_pos_w: lowest point of each foot collision geom (sphere centre minus radius in z;
    # other geom types: their centre), for swing clearance in the summary.
    geom_ids = [robot.geom_names.index(g) for g in robot_cfg.foot_geoms]
    mj_model = env.sim.mj_model
    foot_radius = torch.zeros(len(geom_ids), 3, device=env.device)
    for i, g in enumerate(geom_ids):
        gid = int(robot.indexing.geom_ids[g])
        if mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
            foot_radius[i, 2] = float(mj_model.geom_size[gid, 0])

    seg_end = np.cumsum([s.duration_s for s in schedule]) if schedule else None
    log: dict[str, list] = {k: [] for k in (
        "qpos", "qvel", "command", "base_lin_vel_b", "base_ang_vel_b", "joint_pos", "joint_vel",
        "actions", "torques", "foot_vel_w", "foot_pos_w", "foot_contact", "foot_force_w",
        "reward", "fell", "timeout", "segment")}

    obs_env = run.viewer_env
    run.env.reset()
    with torch.inference_mode():
        for t in range(steps):
            seg = int(np.searchsorted(seg_end, t * dt, side="right")) if schedule else -1
            seg = min(seg, len(schedule) - 1) if schedule else -1
            if schedule:
                cmd = torch.tensor(schedule[seg].command, device=env.device)
                term.vel_command_b[:] = cmd
                term.is_standing_env[:] = False
                if hasattr(term, "is_world_env"):
                    term.is_world_env[:] = False
                term.time_left[:] = 1e9          # no resampling mid-segment
            command = term.vel_command_b.clone()
            obs = obs_env.get_observations()
            actions = run.policy(obs)
            obs_env.step(actions)

            # Many of these are views into live simulation memory: copy every one.
            snap = {
                "qpos": env.sim.data.qpos, "qvel": env.sim.data.qvel, "command": command,
                "base_lin_vel_b": robot.data.root_link_lin_vel_b,
                "base_ang_vel_b": robot.data.root_link_ang_vel_b,
                "joint_pos": robot.data.joint_pos, "joint_vel": robot.data.joint_vel,
                "actions": actions, "torques": robot.data.actuator_force,
                "foot_vel_w": robot.data.site_lin_vel_w[:, site_ids],
                "foot_pos_w": robot.data.geom_pos_w[:, geom_ids] - foot_radius,
                "foot_contact": (env.scene["feet_contact"].data.found > 0 if has_contacts else
                                 torch.zeros(env.num_envs, len(site_ids), dtype=torch.bool,
                                             device=env.device)),
                # Net contact force on each foot, world frame, in the feet_contact sensor's
                # foot order (the same order as foot_contact); zeros without the sensor.
                "foot_force_w": (env.scene["feet_contact"].data.force if has_contacts else
                                 torch.zeros(env.num_envs, len(site_ids), 3, device=env.device)),
                "reward": env.reward_buf, "fell": env.reset_terminated,
                "timeout": env.reset_time_outs,
                "segment": torch.full((env.num_envs,), seg, device=env.device),
            }
            for k, v in snap.items():
                log[k].append(v.clone())
            # Copy to the CPU in chunks to keep GPU memory flat on long rollouts.
            if (t + 1) % 250 == 0 or t == steps - 1:
                for k, v in log.items():
                    log[k] = [_to_numpy(x) for x in v]
                if progress:
                    print(f"  recorded {t + 1}/{steps} steps ({(t + 1) * dt:.1f} s)", flush=True)

    out = {k: np.stack([_to_numpy(x) for x in v]) for k, v in log.items()}
    out["dt"] = np.float32(dt)
    return out


FOOT_CONTACT_HEIGHT = 0.035   # m: a foot site this close to flat ground counts as in contact


def record_tracking_rollout(run: LoadedRun, duration_s: float | None = None,
                            progress: bool = True) -> dict[str, np.ndarray]:
    """Play one reference clip per robot from its first frame and record robot and reference.

    Same arrays as ``record_rollout`` so the renderer, plots, metrics and replay work
    unchanged, with the reference in the role of the command: ``command`` is the reference
    base velocity [vx, vy, wz] in the reference's own body frame. Added: ``ref_qpos`` (the
    reference pose, drawn as a ghost), ``clip_frame`` and ``clip`` (which clip a robot plays).
    A clip that ends, or a robot that falls, restarts the same clip.
    ``duration_s`` None records until the longest clip has played once (at most 30 s).
    The tracking environment has no contact sensor, so ``foot_contact`` is foot height
    below FOOT_CONTACT_HEIGHT (flat ground).
    """
    import torch
    from mjlab.utils.lab_api.math import quat_apply_inverse

    env, robot_cfg = run.env, run.robot
    dt = float(env.step_dt)
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    robot = env.scene["robot"]
    n = env.num_envs
    if list(robot.joint_names) != list(robot_cfg.joint_names):
        raise ValueError(f"joint order differs: env {list(robot.joint_names)} vs "
                         f"robot config {list(robot_cfg.joint_names)}")
    clip_ids = torch.arange(n, device=env.device) % lib.num_clips
    starts = lib.clip_start[clip_ids]
    lengths = np.array([int(lib.clips[int(c)].length) for c in clip_ids])
    if duration_s is None:
        duration_s = min(30.0, float(lengths.max()) * dt)
    steps = round(duration_s / dt)
    site_ids = [robot.site_names.index(s) for s in robot_cfg.foot_sites if s in robot.site_names]
    root_i = list(cmd.cfg.body_names).index(cmd.cfg.root_body_name)

    log: dict[str, list] = {k: [] for k in (
        "qpos", "qvel", "command", "base_lin_vel_b", "base_ang_vel_b", "joint_pos", "joint_vel",
        "actions", "torques", "foot_vel_w", "foot_contact", "reward", "fell", "timeout", "segment",
        "ref_qpos", "clip_frame")}

    def to_cpu(done: int) -> None:
        for k, v in log.items():
            log[k] = [_to_numpy(x) for x in v]
        if progress:
            print(f"  recorded {done}/{steps} steps ({done * dt:.1f} s)", flush=True)

    obs_env = run.viewer_env
    cmd.forced_frame[:] = starts            # every reset (start, fall, clip end) replays its clip
    env.reset()
    with torch.inference_mode():
        for t in range(steps):
            cmd.forced_frame[:] = starts
            obs = obs_env.get_observations()
            actions = run.policy(obs)
            obs_env.step(actions)

            ref_quat = cmd.root_quat_w
            ref_v = quat_apply_inverse(ref_quat, cmd.body_lin_vel_w[:, root_i])
            ref_w = quat_apply_inverse(ref_quat, cmd.body_ang_vel_w[:, root_i])
            if site_ids:
                foot_vel = robot.data.site_lin_vel_w[:, site_ids]
                contact = robot.data.site_pos_w[:, site_ids, 2] < FOOT_CONTACT_HEIGHT
            else:
                foot_vel = torch.zeros(n, 4, 3, device=env.device)
                contact = torch.zeros(n, 4, dtype=torch.bool, device=env.device)
            snap = {
                "qpos": env.sim.data.qpos, "qvel": env.sim.data.qvel,
                "command": torch.stack([ref_v[:, 0], ref_v[:, 1], ref_w[:, 2]], dim=1),
                "base_lin_vel_b": robot.data.root_link_lin_vel_b,
                "base_ang_vel_b": robot.data.root_link_ang_vel_b,
                "joint_pos": robot.data.joint_pos, "joint_vel": robot.data.joint_vel,
                "actions": actions, "torques": robot.data.actuator_force,
                "foot_vel_w": foot_vel, "foot_contact": contact,
                "reward": env.reward_buf, "fell": env.reset_terminated,
                "timeout": env.reset_time_outs,
                "segment": torch.full((n,), -1, device=env.device),
                "ref_qpos": torch.cat([cmd.root_pos_w, ref_quat, cmd.joint_pos], dim=1),
                "clip_frame": cmd.frame - starts,
            }
            for k, v in snap.items():
                log[k].append(v.clone())
            if (t + 1) % 250 == 0 or t == steps - 1:
                to_cpu(t + 1)

    out = {k: np.stack([_to_numpy(x) for x in v]) for k, v in log.items()}
    out["dt"] = np.float32(dt)
    out["clip_length"] = lengths
    out["clip_names"] = np.array([lib.clips[int(c)].name for c in clip_ids])
    return out


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    import torch

    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    t = t.detach().to("cpu")
    return t.numpy().astype(np.float32) if t.dtype.is_floating_point else t.numpy().copy()
