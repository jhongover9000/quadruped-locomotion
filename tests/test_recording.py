"""Recording files and summaries (scripts/record.py, scripts/replay.py) on synthetic data."""

import copy

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


def test_torque_table_and_continuous_rating():
    """RMS and peak |torque| per joint kind, max over legs, on the steps rmse uses."""
    rollout, meta = fake_rollout(fall_robot=1, fall_step=120)
    tau = np.zeros((T, N, 12), np.float32)
    tau[:, :, 2] = 4.0                                    # FR calf: 4 N m
    tau[:, :, 11] = np.where(np.arange(T) % 2, 6.0, -6.0)[:, None]   # RL calf: +-6 N m, RMS 6
    tau[:, :, 1] = 3.0
    tau[: round(1.0 / DT), :, 1] = 50.0                   # first second: settling, excluded
    tau[121:125, 1, 1] = 40.0                             # robot 1 just after its fall: excluded
    rollout["torques"] = tau
    s = render.summarize(rollout, meta)
    t = s["segments"][0]["torque_Nm"]
    assert t["calf"]["rms"] == pytest.approx(6.0) and t["calf"]["peak"] == pytest.approx(6.0)
    assert t["calf"]["per_leg"]["FR"]["rms"] == pytest.approx(4.0)
    assert t["thigh"]["rms"] == pytest.approx(3.0) and t["thigh"]["peak"] == pytest.approx(3.0)
    assert t["hip"]["rms"] == 0.0 and "over_continuous" not in t["hip"]
    assert "*" not in render.format_summary(s, meta).split("Joint torque")[1]

    meta["continuous_torque"] = [5.0] * 12
    s = render.summarize(rollout, meta)
    t = s["segments"][0]["torque_Nm"]
    assert t["calf"]["over_continuous"] and not t["thigh"]["over_continuous"]
    assert t["calf"]["per_leg"]["RL"]["over_continuous"] and not t["calf"]["per_leg"]["FR"]["over_continuous"]
    text = render.format_summary(s, meta)
    assert "6.00*" in text and "3.00 " in text and "continuous torque rating (hip 5, thigh 5, calf 5" in text


def test_continuous_torque_in_robot_config():
    from quadruped_rl.robots.config import load_robot, robot_from_dict
    from quadruped_rl.utils.config import load_group

    raw = copy.deepcopy(load_group("robots", "go1"))
    assert robot_from_dict(raw).actuator.continuous_torque is None
    raw["actuator"]["continuous_torque"] = [5, 6, 7]
    assert robot_from_dict(raw).actuator.continuous_torque == [5.0, 6.0, 7.0] * 4
    raw["actuator"]["continuous_torque"] = [5, 6]
    with pytest.raises(ValueError, match="continuous_torque"):
        robot_from_dict(raw)
    assert load_robot("astra").actuator.continuous_torque == [5.0] * 12


def test_gait_metrics():
    """Synthetic trot at 2 Hz: each foot 13 of 25 steps in contact, swing peak 60 mm; known joint
    sine waves, trunk height and power."""
    rollout, meta = fake_rollout(fall_robot=1, fall_step=120)
    t = np.arange(T)
    period = round(0.5 / DT)                              # 25 steps = 2 Hz
    phase = ((t + 7) % period) / period                   # [0, 1); touchdowns off the mask edges
    swing = phase >= 0.5
    contact = np.repeat(~swing[:, None, None], N, 1).repeat(4, 2)
    height = np.where(swing, 0.06 * np.sin(np.pi * (phase - 0.5) / 0.5), 0.0)
    foot = np.zeros((T, N, 4, 3), np.float32)
    foot[..., 2] = height[:, None, None]
    q = np.zeros((T, N, 12), np.float32)
    q[:, :, 1::3] = 0.8 + 0.2 * np.sin(2 * np.pi * t / period)[:, None, None]
    q[:, :, 2::3] = -1.5
    rollout.update(foot_contact=contact, foot_pos_w=foot, joint_pos=q)
    rollout["qpos"][:, :, 2] = 0.40
    rollout["torques"][:] = 2.0
    rollout["joint_vel"][:] = -1.5                        # |tau qd| = 3 W per joint
    s = render.summarize(rollout, meta)
    g = s["segments"][0]["gait"]
    assert g["stride_hz"] == pytest.approx(2.0, rel=0.05)
    assert g["duty_factor"] == pytest.approx(13 / 25, abs=0.02)
    assert g["clearance_mm_median"] == pytest.approx(60.0, abs=1.5)
    assert g["clearance_mm_p90"] == pytest.approx(60.0, abs=1.5)
    assert g["swings"] > 0
    assert g["thigh_excursion_rad"] == pytest.approx(0.4 * 0.95, abs=0.03)   # p95-p5 of 0.2 sin
    assert g["calf_excursion_rad"] == pytest.approx(0.0) and g["calf_mean_rad"] == pytest.approx(-1.5)
    assert g["trunk_height_m"] == pytest.approx(0.40) and g["power_W"] == pytest.approx(36.0)
    text = render.format_summary(s, meta)
    assert "Gait (same steps as rmse)" in text and "  60" in text.split("Gait")[1]

    del rollout["foot_pos_w"]                             # older recordings: no clearance
    g = render.summarize(rollout, meta)["segments"][0]["gait"]
    assert g["clearance_mm_median"] is None and g["stride_hz"] == pytest.approx(2.0, rel=0.05)
    render.format_summary(render.summarize(rollout, meta), meta)


def test_summary_lists_spawn_tiles():
    rollout, meta = fake_rollout(fall_robot=2)
    assert "Spawn tiles" not in render.format_summary(render.summarize(rollout, meta), meta)
    meta["terrain_tiles"] = [
        {"robot": 0, "row": 0, "col": 1, "type": "flat", "difficulty": 0.1, "slope_deg": None},
        {"robot": 1, "row": 2, "col": 3, "type": "hf_pyramid_slope", "difficulty": 0.8, "slope_deg": 25.64},
        {"robot": 2, "row": 4, "col": 0, "type": "hf_pyramid_slope_inv", "difficulty": 1.0,
         "slope_deg": 30.96}]
    text = render.format_summary(render.summarize(rollout, meta), meta).split("Spawn tiles")[1]
    lines = text.splitlines()
    assert "hf_pyramid_slope" in lines[3] and "25.6 deg" in lines[3] and lines[3].endswith("0")
    assert "31.0 deg" in lines[4] and lines[4].endswith("1")              # robot 2 fell once
    assert "flat" in lines[2] and lines[2].split()[-2] == "-"


def test_summary_lists_spawn_tiles_of_every_robot():
    """All 16 robots of a 16-robot recording appear in the Spawn tiles block."""
    rollout, meta = fake_rollout()
    n = 16
    rollout = {k: (np.repeat(v[:, :1], n, axis=1) if isinstance(v, np.ndarray) and v.ndim >= 2 else v)
               for k, v in rollout.items()}
    meta["terrain_tiles"] = [{"robot": i, "row": i % 5, "col": i // 5, "type": "flat",
                              "difficulty": 0.5, "slope_deg": None} for i in range(n)]
    text = render.format_summary(render.summarize(rollout, meta), meta).split("Spawn tiles")[1]
    listed = [int(line.split()[0]) for line in text.splitlines()[2:] if line.strip()[:1].isdigit()]
    assert listed == list(range(n))
