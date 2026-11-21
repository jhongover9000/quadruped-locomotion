"""dc_motor actuator model and joint friction / armature randomization (robot YAML)."""

import copy

import numpy as np
import pytest

from quadruped_rl.robots.config import load_robot, robot_from_dict
from quadruped_rl.utils.config import load_group


def test_dc_motor_config():
    astra = load_robot("astra")
    assert astra.actuator.model == "dc_motor"
    assert astra.actuator.params["saturation_effort"] == [250.0] * 12
    assert astra.actuator.params["no_load_speed"] == [22.6] * 12
    stance2 = load_robot("astra_stance2")
    assert stance2.actuator.model == "dc_motor" and stance2.actuator.params == astra.actuator.params
    for name in ("astra", "astra_stance2"):
        dr = load_robot(name).domain_randomization
        assert dr["frictionloss_scale"] == [1.0, 1.25] and dr["armature_scale"] == [0.5, 1.5]
    for name in ("go1", "testdog"):
        robot = load_robot(name)
        assert robot.actuator.model == "ideal_pd"
        assert "frictionloss_scale" not in robot.domain_randomization
        assert "armature_scale" not in robot.domain_randomization

    raw = copy.deepcopy(load_group("robots", "astra"))
    del raw["actuator"]["params"]["no_load_speed"]
    with pytest.raises(ValueError, match="no_load_speed"):
        robot_from_dict(raw)
    raw = copy.deepcopy(load_group("robots", "astra"))
    raw["domain_randomization"]["armature_scale"] = [1.5, 0.5]
    with pytest.raises(ValueError, match="armature_scale"):
        robot_from_dict(raw)


def test_dc_motor_torque_speed_limit():
    """Available torque clip(tau, -limit_neg(w), limit_pos(w)) with
    limit_pos = clamp(sat * (1 - w / w0), 0, torque_limit), limit_neg its mirror."""
    torch = pytest.importorskip("torch")
    dc = pytest.importorskip("mjlab.actuator.dc_actuator")
    act = load_robot("astra").actuator
    sat, w0, tl = act.params["saturation_effort"][2], act.params["no_load_speed"][2], act.torque_limit[2]
    w = torch.tensor([0.0, 15.0, 21.0, 22.6, -15.0, -21.0])
    n = len(w)

    def clip(tau):
        return dc.dc_motor_clip(torch.full((n,), tau), torch.full((n,), sat), torch.full((n,), w0),
                                torch.full((n,), tl), w).numpy()

    mid = sat * (1 - 21.0 / 22.6)                         # 17.7 N m at 21 rad/s
    np.testing.assert_allclose(clip(100.0), [30, 30, mid, 0, 30, 30], atol=1e-4)
    np.testing.assert_allclose(clip(-100.0), [-30, -30, -30, -30, -30, -mid], atol=1e-4)
    np.testing.assert_allclose(clip(10.0), [10, 10, 10, 0, 10, 10], atol=1e-4)   # inside: untouched


@pytest.mark.sim
def test_astra_env_actuators_and_joint_randomization():
    pytest.importorskip("mjlab")
    from mjlab.actuator import BuiltinPositionActuator, DcMotorActuator

    from quadruped_rl.envs.builder import make_env

    n = 16
    robot = load_robot("astra")
    env = make_env(robot, load_group("tasks", "flat_velocity"), num_envs=n, device="cpu", seed=0)
    try:
        ent = env.scene["robot"]
        assert all(isinstance(a, DcMotorActuator) for a in ent.actuators) and len(ent.actuators) == 12
        sat = np.concatenate([a.saturation_effort.cpu().numpy() for a in ent.actuators], axis=1)
        w0 = np.concatenate([a.velocity_limit_motor.cpu().numpy() for a in ent.actuators], axis=1)
        assert sat.shape == (n, 12) and np.allclose(sat, 250.0) and np.allclose(w0, 22.6)

        dof = ent.indexing.joint_v_adr.cpu().numpy()
        fric = env.sim.model.dof_frictionloss.cpu().numpy()[:, dof]      # [n, 12]
        arm = env.sim.model.dof_armature.cpu().numpy()[:, dof]
        f0, a0 = np.array(robot.actuator.frictionloss), np.array(robot.actuator.armature)
        assert fric.shape == arm.shape == (n, 12)
        assert (fric >= f0 * 1.0 - 1e-6).all() and (fric <= f0 * 1.25 + 1e-6).all()
        assert (arm >= a0 * 0.5 - 1e-7).all() and (arm <= a0 * 1.5 + 1e-7).all()
        # One draw per env and per joint: values differ across envs and across joints.
        assert np.ptp(fric[:, 2]) > 0.05 and np.ptp(arm[:, 2]) > 0.002
        assert np.ptp(fric[0, 2::3]) > 0 and np.ptp(arm[0]) > 0
        env.reset()
        for _ in range(5):
            env.step(torch_zeros(env))
    finally:
        env.close()

    go1 = make_env(load_robot("go1"), load_group("tasks", "flat_velocity"), num_envs=4, device="cpu")
    try:
        assert all(isinstance(a, BuiltinPositionActuator) for a in go1.scene["robot"].actuators)
        dof = go1.scene["robot"].indexing.joint_v_adr.cpu().numpy()
        fric = go1.sim.model.dof_frictionloss.cpu().numpy()
        fric = fric[:, dof] if fric.ndim == 2 else fric[None, dof]
        assert np.ptp(fric, axis=0).max() == 0.0             # not randomized
    finally:
        go1.close()


def torch_zeros(env):
    import torch

    return torch.zeros(env.num_envs, env.action_manager.total_action_dim)
