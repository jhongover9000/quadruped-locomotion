"""Reference-motion clip format: loading, validation, and the library view."""

import numpy as np
import pytest

from quadruped_rl.motions.format import load_clip, validate_clip
from quadruped_rl.motions.motion_lib import MotionLibrary

JOINTS, BODIES = ["a_joint", "b_joint"], ["trunk", "a", "b"]


def write_clip(path, frames=20, speed=0.5, **override):
    t = np.arange(frames) / 50.0
    quat = np.zeros((frames, len(BODIES), 4))
    quat[..., 0] = 1.0
    pos = np.zeros((frames, len(BODIES), 3))
    pos[:, :, 0] = (speed * t)[:, None]
    vel = np.zeros_like(pos)
    vel[..., 0] = speed
    data = {"fps": 50.0, "joint_names": np.array(JOINTS), "body_names": np.array(BODIES),
            "joint_pos": np.zeros((frames, 2)), "joint_vel": np.zeros((frames, 2)),
            "body_pos_w": pos, "body_quat_w": quat, "body_lin_vel_w": vel,
            "body_ang_vel_w": np.zeros_like(pos)}
    data.update(override)
    np.savez(path, **data)


def test_clip_round_trip(tmp_path):
    write_clip(tmp_path / "walk.npz", frames=100, speed=0.8)
    clip = load_clip(tmp_path / "walk.npz")
    assert (clip.name, clip.frames, clip.duration_s) == ("walk", 100, 2.0)
    assert clip.mean_speed("trunk") == pytest.approx(0.8)
    pos, quat = clip.root("trunk")
    assert pos.shape == (100, 3) and quat.shape == (100, 4)
    validate_clip(clip, JOINTS)
    with pytest.raises(ValueError, match="joint order"):
        validate_clip(clip, list(reversed(JOINTS)))


def test_bad_clips_are_rejected(tmp_path):
    write_clip(tmp_path / "short.npz", joint_vel=np.zeros((19, 2)))
    with pytest.raises(ValueError, match="joint_vel has shape"):
        load_clip(tmp_path / "short.npz")
    write_clip(tmp_path / "nan.npz", joint_pos=np.full((20, 2), np.nan))
    with pytest.raises(ValueError, match="NaN"):
        load_clip(tmp_path / "nan.npz")
    write_clip(tmp_path / "quat.npz", body_quat_w=np.ones((20, 3, 4)))
    with pytest.raises(ValueError, match="not unit length"):
        load_clip(tmp_path / "quat.npz")
    np.savez(tmp_path / "other.npz", root_pos=np.zeros((5, 3)))
    with pytest.raises(ValueError, match="missing"):
        load_clip(tmp_path / "other.npz")


def test_library_groups_and_names(tmp_path):
    rows = [("walk", "1", "train"), ("walk_mirror", "1", "train"), ("trot", "1", "holdout"),
            ("flip", "0", "train"), ("gone", "1", "hard")]
    for name, _, _ in rows[:4]:
        write_clip(tmp_path / f"{name}.npz")
    (tmp_path / "manifest.csv").write_text("name,keep,group\n" + "\n".join(",".join(r) for r in rows))
    lib = MotionLibrary(tmp_path)
    assert lib.groups() == {"train": 2, "holdout": 1, "hard": 1}       # keep=0 is dropped
    assert lib.names("train") == ["walk"]                               # no mirrors, no missing files
    assert lib.names("train", exclude=None) == ["walk", "walk_mirror"]
    assert lib.names("train,holdout") == ["walk", "trot"]
    assert lib.summary("trunk", "holdout")[0]["frames"] == 20
