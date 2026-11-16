"""Every config loads, inherits correctly, and every robot passes validation."""

import pytest

from quadruped_rl.robots import load_robot
from quadruped_rl.utils.config import apply_overrides, available, load_group


@pytest.mark.parametrize("name", available("robots"))
def test_robot_configs_validate(name):
    robot = load_robot(name)
    assert robot.num_joints == 12


@pytest.mark.parametrize("group", ["tasks", "algorithms", "eval", "sweeps"])
def test_group_loads(group):
    for name in available(group):
        assert isinstance(load_group(group, name), dict)


def test_rough_inherits_flat():
    flat = load_group("tasks", "flat_velocity")
    rough = load_group("tasks", "rough_velocity")
    assert rough["sim"] == flat["sim"]
    assert rough["terrain"]["type"] == "curriculum"
    assert rough["rewards"]["track_lin_vel_xy"] == flat["rewards"]["track_lin_vel_xy"]


def test_replace_drops_parent_entries():
    from quadruped_rl.utils.config import deep_merge

    parent = {"rewards": {"track_lin_vel_xy": 1.0, "action_rate": -0.01}}
    merged = deep_merge(parent, {"rewards": {"_replace": True, "imitate": 0.5}})
    assert merged["rewards"] == {"imitate": 0.5}
    assert deep_merge(parent, {"rewards": {"imitate": 0.5}})["rewards"]["track_lin_vel_xy"] == 1.0


def test_motion_tracking_describes_the_clip_library():
    tracking = load_group("tasks", "motion_tracking")
    assert tracking["kind"] == "tracking"
    for key in ("dir", "manifest", "train_groups", "eval_groups", "future_frames"):
        assert key in tracking["motions"]


def test_algorithms_name_a_real_task():
    tasks = set(available("tasks"))
    for name in available("algorithms"):
        assert load_group("algorithms", name)["default_task"] in tasks


def test_overrides():
    cfg = apply_overrides({"rewards": {"a": 1}}, ["rewards.a=-0.5", "commands.base_height=[0.2, 0.3]"])
    assert cfg["rewards"]["a"] == -0.5
    assert cfg["commands"]["base_height"] == [0.2, 0.3]


def test_template_cannot_be_loaded():
    with pytest.raises(ValueError):
        load_group("robots", "_template")
