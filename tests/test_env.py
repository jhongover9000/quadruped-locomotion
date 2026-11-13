"""The environment builds from YAML and steps. Needs mjlab: run with `pytest -m sim`."""

import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

from quadruped_rl.envs.builder import make_env  # noqa: E402
from quadruped_rl.robots import load_robot  # noqa: E402
from quadruped_rl.utils.config import load_group  # noqa: E402

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def env():
    e = make_env(load_robot("go1"), load_group("tasks", "flat_velocity"), num_envs=4, device="cpu")
    yield e
    e.close()


def test_observation_and_action_sizes(env):
    obs, _ = env.reset()
    # actor: ang vel 3 + gravity 3 + command 3 + q 12 + qd 12 + last action 12
    assert obs["actor"].shape == (4, 45)
    # critic adds lin vel 3 and foot contacts 4
    assert obs["critic"].shape == (4, 52)
    assert env.action_manager.total_action_dim == 12


def test_robot_stands_under_zero_action(env):
    env.reset()
    for _ in range(50):
        _, reward, terminated, _, _ = env.step(torch.zeros(4, 12))
    assert not terminated.any()
    height = env.scene["robot"].data.root_link_pos_w[:, 2]
    assert (height > 0.2).all()
    assert (reward >= 0).all()   # clip_reward_at_zero


def test_joint_order_is_checked():
    robot = load_robot("go1")
    robot.joint_names = robot.joint_names[3:] + robot.joint_names[:3]
    with pytest.raises(ValueError, match="joint_names"):
        make_env(robot, load_group("tasks", "flat_velocity"), num_envs=1, device="cpu")
