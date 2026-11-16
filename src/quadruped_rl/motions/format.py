"""Reference-motion clip format (.npz), shared with the SuperMimic reference implementation.

One file per clip, sampled at a fixed rate (the libraries we build are 50 Hz, the control
rate). T frames, J joints, B bodies of the robot model:

    fps              scalar
    joint_names      [J]        the robot's joint order (must equal the robot config's)
    body_names       [B]        every body of the robot model, in model order
    joint_pos        [T, J]     rad
    joint_vel        [T, J]     rad/s
    body_pos_w       [T, B, 3]  m, world frame
    body_quat_w      [T, B, 4]  w, x, y, z
    body_lin_vel_w   [T, B, 3]  m/s
    body_ang_vel_w   [T, B, 3]  rad/s

The root (trunk) pose is the root body's row of ``body_pos_w`` / ``body_quat_w``.
A library is a directory of clips plus ``manifest.csv`` (columns: name, keep, group, ...),
where ``group`` splits clips into train / hard / holdout (see motions/motion_lib.py).

This module only needs numpy, so clips can be inspected without the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARRAYS = {"joint_pos": 2, "joint_vel": 2, "body_pos_w": 3, "body_quat_w": 3,
          "body_lin_vel_w": 3, "body_ang_vel_w": 3}


@dataclass
class Clip:
    name: str
    fps: float
    joint_names: list[str]
    body_names: list[str]
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray

    @property
    def frames(self) -> int:
        return self.joint_pos.shape[0]

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps

    def root(self, root_body: str) -> tuple[np.ndarray, np.ndarray]:
        """(position [T, 3], quaternion wxyz [T, 4]) of the root body."""
        i = self.body_names.index(root_body)
        return self.body_pos_w[:, i], self.body_quat_w[:, i]

    def mean_speed(self, root_body: str) -> float:
        """Mean horizontal speed of the root, m/s."""
        i = self.body_names.index(root_body)
        return float(np.linalg.norm(self.body_lin_vel_w[:, i, :2], axis=-1).mean())


def load_clip(path: str | Path) -> Clip:
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        missing = [k for k in (*ARRAYS, "fps", "joint_names", "body_names") if k not in f]
        if missing:
            raise ValueError(f"{path.name}: missing {missing}; not a clip in this format")
        clip = Clip(name=path.stem, fps=float(f["fps"]),
                    joint_names=[str(n) for n in f["joint_names"]],
                    body_names=[str(n) for n in f["body_names"]],
                    **{k: np.asarray(f[k]) for k in ARRAYS})
    validate_clip(clip)
    return clip


def validate_clip(clip: Clip, joint_names: list[str] | None = None) -> None:
    """Shapes agree, quaternions are unit, values are finite; optionally the joint order
    matches a robot config's ``joint_names``."""
    T, J, B = clip.frames, len(clip.joint_names), len(clip.body_names)
    expected = {"joint_pos": (T, J), "joint_vel": (T, J), "body_pos_w": (T, B, 3),
                "body_quat_w": (T, B, 4), "body_lin_vel_w": (T, B, 3), "body_ang_vel_w": (T, B, 3)}
    for key, shape in expected.items():
        value = getattr(clip, key)
        if value.shape != shape:
            raise ValueError(f"{clip.name}: {key} has shape {value.shape}, expected {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{clip.name}: {key} contains NaN or inf")
    if T < 2 or clip.fps <= 0:
        raise ValueError(f"{clip.name}: needs at least 2 frames and a positive fps")
    norms = np.linalg.norm(clip.body_quat_w, axis=-1)
    if np.abs(norms - 1.0).max() > 1e-3:
        raise ValueError(f"{clip.name}: body_quat_w is not unit length (max error "
                         f"{np.abs(norms - 1.0).max():.3g})")
    if joint_names is not None and list(joint_names) != clip.joint_names:
        raise ValueError(f"{clip.name}: joint order {clip.joint_names} differs from the robot's "
                         f"{list(joint_names)}")
