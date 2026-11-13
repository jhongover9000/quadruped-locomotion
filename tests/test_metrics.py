import numpy as np

from quadruped_rl.evaluation import metrics


def test_jerk_of_quadratic_velocity():
    dt = 0.01
    t = np.arange(0, 1, dt)
    qd = (t**2)[:, None]               # qd = t^2  ->  q''' = 2  ->  jerk^2 = 4
    assert np.isclose(metrics.joint_jerk(qd, dt), 4.0)


def test_linear_actions_are_perfectly_smooth():
    a = np.linspace(0, 1, 50)[:, None].repeat(12, axis=1)
    assert np.isclose(metrics.action_smoothness(a), 0.0)


def test_high_freq_fraction():
    dt = 0.02
    t = np.arange(0, 10, dt)
    assert metrics.high_freq_fraction(np.sin(2 * np.pi * 1 * t), dt) < 0.01
    assert metrics.high_freq_fraction(np.sin(2 * np.pi * 15 * t), dt) > 0.99


def test_cost_of_transport():
    T = 100
    cot = metrics.cost_of_transport(np.ones((T, 1)), np.ones((T, 1)),
                                    np.tile([1.0, 0.0], (T, 1)), mass=1.0)
    assert np.isclose(cot, 1 / metrics.G)


def test_foot_slip_ignores_swing_feet():
    vel = np.zeros((10, 4, 2))
    vel[:, 0] = [1.0, 0.0]              # foot 0 moves...
    contact = np.ones((10, 4), dtype=bool)
    contact[:, 0] = False               # ...but only while in the air
    assert metrics.foot_slip(vel, contact) == 0.0
