"""Recording files and summaries (scripts/record.py, scripts/replay.py) on synthetic data."""

import numpy as np
import pytest

from quadruped_rl.evaluation import render
from quadruped_rl.evaluation.rollout import SCENARIOS, fixed_command_task, make_schedule

T, N, DT = 200, 3, 0.02


def fake_rollout(fall_robot: int | None = None, fall_step: int = 120) -> tuple[dict, dict]:
    """Every robot tracks vx = 1 exactly; one may fall."""
    cmd = np.zeros((T, N, 3), np.float32)
    cmd[:, :, 0] = 1.0
    qpos = np.zeros((T, N, 19), np.float32)
    qpos[:, :, 3] = 1.0                                   # identity quaternion
    qpos[:, :, 0] = np.arange(T)[:, None] * DT            # moving forward at 1 m/s
    fell = np.zeros((T, N), bool)
    if fall_robot is not None:
        fell[fall_step, fall_robot] = True
    rollout = {
        "qpos": qpos, "qvel": np.zeros((T, N, 18), np.float32), "command": cmd,
        "base_lin_vel_b": cmd.copy(), "base_ang_vel_b": np.zeros((T, N, 3), np.float32),
        "joint_pos": np.zeros((T, N, 12), np.float32), "joint_vel": np.zeros((T, N, 12), np.float32),
        "actions": np.zeros((T, N, 12), np.float32), "torques": np.zeros((T, N, 12), np.float32),
        "foot_vel_w": np.zeros((T, N, 4, 3), np.float32), "foot_contact": np.ones((T, N, 4), bool),
        "reward": np.ones((T, N), np.float32), "fell": fell, "timeout": np.zeros((T, N), bool),
        "segment": np.zeros((T, N), np.int64), "dt": np.float32(DT),
    }
    meta = {"dt": DT, "mass": 12.0, "root_body": "trunk", "joint_names": [f"j{i}" for i in range(12)],
            "foot_names": ["FR", "FL", "RR", "RL"],
            "segments": [{"label": "forward", "command": [1.0, 0.0, 0.0], "start_s": 0.0,
                          "end_s": T * DT}]}
    return rollout, meta


def test_schedules():
    tour = make_schedule("tour")
    assert len(tour) == len(SCENARIOS["tour"]) and tour[0].command == (0.0, 0.0, 0.0)
    fixed = make_schedule("tour", command=[1.0, 0.0, 0.5], duration_s=7.0)
    assert len(fixed) == 1 and fixed[0].command == (1.0, 0.0, 0.5) and fixed[0].duration_s == 7.0
    assert make_schedule("random") is None
    with pytest.raises(KeyError):
        make_schedule("moonwalk")


def test_fixed_command_task_pins_the_sampler():
    task = {"commands": {"lin_vel_x": [-1, 1], "lin_vel_y": [-0.5, 0.5], "ang_vel_z": [-1, 1],
                         "standing_fraction": 0.1}}
    out = fixed_command_task(task, [0.8, 0.0, -0.2])
    assert out["commands"]["lin_vel_x"] == [0.8, 0.8] and out["commands"]["ang_vel_z"] == [-0.2, -0.2]
    assert out["commands"]["standing_fraction"] == 0.0
    assert task["commands"]["lin_vel_x"] == [-1, 1]       # input left alone


def test_valid_mask_skips_settling_and_falls():
    rollout, meta = fake_rollout(fall_robot=1, fall_step=120)
    ok = render.valid_mask(rollout, meta, settle_s=1.0)
    k = round(1.0 / DT)
    assert not ok[:k].any() and ok[k:, 0].all()            # first second of the segment
    assert not ok[120:120 + k + 1, 1].any() and ok[120 + k + 1:, 1].all()   # after the fall


def test_summary_of_perfect_tracking():
    rollout, meta = fake_rollout(fall_robot=2)
    s = render.summarize(rollout, meta)
    seg = s["segments"][0]
    assert s["total_falls"] == 1 and s["falls_per_robot"] == [0, 0, 1]
    assert seg["rmse_xy"] == pytest.approx(0.0) and seg["mean_vx"] == pytest.approx(1.0)
    assert s["robots_without_falls"] == 2
    assert "forward" in render.format_summary(s, meta)


def test_recording_round_trip(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='trunk'><freejoint/><geom size='0.1'/></body></worldbody></mujoco>")
    rollout, meta = fake_rollout()
    rollout = {**rollout, "qpos": rollout["qpos"][..., :7], "qvel": rollout["qvel"][..., :6]}
    render.save_recording(tmp_path, rollout, model, meta)
    model2, rollout2, meta2 = render.load_recording(tmp_path / "rollout.npz")
    assert model2.nq == 7 and meta2["segments"] == meta["segments"]
    np.testing.assert_array_equal(rollout2["qpos"], rollout["qpos"])
