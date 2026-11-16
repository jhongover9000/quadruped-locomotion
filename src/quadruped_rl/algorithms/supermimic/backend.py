"""Bridge to the SuperMimic reference implementation (the ``supermimic`` package).

The world model, imagined rollouts, BPTT policy update, tracking environment and motion
library all live in the reference repository; this module finds it, translates this
repository's YAML configs into its ``TrainConfig``, and checks that the robot it simulates
is the robot described by our robot YAML.

Where the reference repository is looked for, in order:

    1. ``backend.root`` in configs/algorithms/supermimic.yaml
    2. the ``SUPERMIMIC_ROOT`` environment variable
    3. ``../supermimic-quadruped`` next to this repository
    4. an installed ``supermimic`` package (``pip install --no-deps -e <root>``)
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import sys
from pathlib import Path

from quadruped_rl.utils.paths import REPO_ROOT

# TrainConfig sections that the algorithm YAML may set field by field.
SECTIONS = ("env", "world_model", "wm_trainer", "policy", "critic", "policy_trainer", "schedule")


class BackendNotFound(ImportError):
    pass


def find_root(algo_cfg: dict | None = None) -> Path | None:
    """Directory of the reference repository, or None when only an installed package exists."""
    configured = ((algo_cfg or {}).get("backend") or {}).get("root")
    candidates = [configured, os.environ.get("SUPERMIMIC_ROOT"), REPO_ROOT.parent / "supermimic-quadruped"]
    for c in candidates:
        if c and (Path(c).expanduser() / "src" / "supermimic").is_dir():
            return Path(c).expanduser().resolve()
    return None


def ensure_importable(algo_cfg: dict | None = None) -> Path | None:
    """Make ``import supermimic`` work; returns the repository root if one was found."""
    root = find_root(algo_cfg)
    if root is not None and str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    try:
        importlib.import_module("supermimic")
    except ImportError as e:
        raise BackendNotFound(
            "The SuperMimic reference implementation was not found. Put the repository at "
            f"{REPO_ROOT.parent / 'supermimic-quadruped'}, or point SUPERMIMIC_ROOT (or "
            "backend.root in configs/algorithms/supermimic.yaml) at it.") from e
    return root


def resolve_path(path: str | Path | None, root: Path | None, robot_spec: str) -> Path | None:
    """Motion paths: ``{robot}`` is the spec name; relative paths are tried against this
    repository first, then the reference repository."""
    if path is None:
        return None
    p = Path(str(path).format(robot=robot_spec)).expanduser()
    if p.is_absolute():
        return p
    for base in (REPO_ROOT, root):
        if base is not None and (base / p).exists():
            return base / p
    return (root or REPO_ROOT) / p


def spec_name(robot_name: str, algo_cfg: dict) -> str:
    specs = (algo_cfg.get("backend") or {}).get("robot_specs") or {}
    if robot_name not in specs:
        raise KeyError(
            f"Robot '{robot_name}' has no SuperMimic RobotSpec. Write one in the reference "
            "repository (docs/PORTING.md there), then add it to backend.robot_specs in "
            f"configs/algorithms/supermimic.yaml. Known: {sorted(specs)}")
    return specs[robot_name]


def _set_fields(obj, values: dict, where: str) -> None:
    fields = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in (values or {}).items():
        if key not in fields:
            raise KeyError(f"supermimic.yaml: '{where}.{key}' is not a field of "
                           f"{type(obj).__name__}. Fields: {sorted(fields)}")
        if value is None and getattr(obj, key) is not None and "None" not in str(fields[key].type):
            continue                                  # null in YAML = keep the default
        current = getattr(obj, key)
        if isinstance(value, list) and (isinstance(current, tuple) or "tuple" in str(fields[key].type)):
            value = tuple(value)
        setattr(obj, key, value)


def build_train_config(robot_name: str, task_cfg: dict, algo_cfg: dict, run_dir: Path,
                       device: str, seed: int, resume: Path | None = None):
    """Translate (robot, task YAML, algorithm YAML) into the reference ``TrainConfig``.

    The run directory becomes the reference log directory, so its TensorBoard events
    (tb/), config.json and ckpt_*.pt sit next to our config.yaml.
    """
    root = ensure_importable(algo_cfg)
    from supermimic.config import TrainConfig

    cfg = TrainConfig()
    cfg.run_name, cfg.log_root = Path(run_dir).name, Path(run_dir).parent
    cfg.robot = spec_name(robot_name, algo_cfg)
    cfg.seed, cfg.device = int(seed), device
    cfg.resume = Path(resume) if resume is not None else None

    motions = task_cfg.get("motions") or {}
    if "dir" not in motions:
        raise KeyError("The task needs a motions.dir (see configs/tasks/motion_tracking.yaml)")
    cfg.data.npz_dir = resolve_path(motions["dir"], root, cfg.robot)
    cfg.data.manifest = resolve_path(motions.get("manifest"), root, cfg.robot)
    cfg.data.group = motions.get("train_groups")
    cfg.data.exclude = motions.get("exclude")
    cfg.data.clips = motions.get("clips")
    cfg.data.limit = motions.get("limit")

    for section in SECTIONS:
        _set_fields(getattr(cfg, section), algo_cfg.get(section), section)

    # Task-level settings (what is tracked, when an episode ends, how clips are sampled).
    cfg.env.future_frames = int(motions.get("future_frames", cfg.env.future_frames))
    cfg.env.episode_length_s = float((task_cfg.get("sim") or {}).get("episode_length_s",
                                                                   cfg.env.episode_length_s))
    term = task_cfg.get("terminations") or {}
    cfg.env.root_height_threshold = term.get("root_height_error_m")
    cfg.env.root_tilt_threshold = term.get("root_tilt_error")
    sampling = motions.get("sampling") or {}
    cfg.env.sampling_mode = sampling.get("mode", cfg.env.sampling_mode)
    cfg.env.clip_weight_alpha = float(sampling.get("clip_weight_alpha", cfg.env.clip_weight_alpha))
    cfg.env.static_bin_factor = float(sampling.get("static_bin_factor", cfg.env.static_bin_factor))

    # Keys every algorithm in this repository shares.
    cfg.env.num_envs = int(algo_cfg["num_envs"])
    cfg.schedule.total_iterations = int(algo_cfg["max_iterations"])
    cfg.schedule.checkpoint_interval = int(algo_cfg.get("save_interval", cfg.schedule.checkpoint_interval))
    if algo_cfg.get("max_hours") is not None:
        cfg.schedule.max_hours = float(algo_cfg["max_hours"])
    # Batch sizes follow the environment count unless set (reference: 10240 / 20480 at 20480 envs).
    if (algo_cfg.get("wm_trainer") or {}).get("batch_size") is None:
        cfg.wm_trainer.batch_size = max(256, cfg.env.num_envs // 2)
    if (algo_cfg.get("policy_trainer") or {}).get("batch_size") is None:
        cfg.policy_trainer.batch_size = max(256, cfg.env.num_envs)
    return cfg


def robot_spec_differences(robot, spec, action_cfg: dict | None = None) -> list[str]:
    """Where our robot YAML and the SuperMimic RobotSpec describe different robots.

    A velocity policy (trained from the YAML) and a tracking policy (trained from the spec)
    are only comparable, and only deployable through the same hardware interface, if these
    agree. Returns human-readable differences; empty means they match.
    """
    import math

    out: list[str] = []
    if list(robot.joint_names) != list(spec.joint_names):
        return [f"joint order: YAML {list(robot.joint_names)} vs spec {list(spec.joint_names)}"]

    entity = spec.entity_cfg_fn(False)
    home = _spec_home(entity, spec.joint_names)
    for name, ours, theirs in zip(robot.joint_names, robot.default_joint_pos, home):
        if theirs is not None and abs(ours - theirs) > 1e-3:
            out.append(f"default pose {name}: YAML {ours:+.3f} rad vs spec {theirs:+.3f} rad")

    act = robot.actuator
    for a in entity.articulation.actuators:
        for j, name in enumerate(robot.joint_names):
            if not _matches(a.target_names_expr, name):
                continue
            for label, ours, theirs, tol in (
                    ("kp", act.kp[j], a.stiffness, 0.02), ("kd", act.kd[j], a.damping, 0.02),
                    ("torque limit", act.torque_limit[j], a.effort_limit, 0.01),
                    ("armature", act.armature[j], a.armature, 0.02)):
                if theirs is not None and not math.isclose(ours, theirs, rel_tol=tol):
                    out.append(f"{label} {name}: YAML {ours:.4g} vs spec {theirs:.4g}")

    if action_cfg is not None:
        scale = float(action_cfg.get("scale", 0.25))
        relative = action_cfg.get("scale_mode", "absolute") == "effort_relative"
        for j, name in enumerate(robot.joint_names):
            ours = scale * act.torque_limit[j] / act.kp[j] if relative else scale
            theirs = spec.action_scale.get(name) if isinstance(spec.action_scale, dict) else spec.action_scale
            if theirs is not None and not math.isclose(ours, theirs, rel_tol=0.02):
                out.append(f"action scale {name}: velocity task {ours:.4g} rad vs spec {theirs:.4g} rad")
    return out


def _matches(patterns, name: str) -> bool:
    import re

    return any(re.fullmatch(p, name) for p in patterns)


def _spec_home(entity, joint_names) -> list[float | None]:
    import re

    table = entity.init_state.joint_pos or {}
    home = []
    for name in joint_names:
        hits = [v for pattern, v in table.items() if re.fullmatch(pattern, name)]
        home.append(float(hits[-1]) if hits else None)
    return home


def collapse(differences: list[str], limit: int = 6) -> str:
    shown = differences[:limit]
    more = len(differences) - len(shown)
    return "\n".join(f"  - {d}" for d in shown) + (f"\n  - ... and {more} more" if more > 0 else "")
