import numpy as np

from quadruped_rl.sysid import mass_properties as mp


def test_symmetric_scales_give_centered_com():
    feet = np.array([[0.2, -0.1], [0.2, 0.1], [-0.2, -0.1], [-0.2, 0.1]])
    com, total = mp.com_from_scales([30.0] * 4, feet)
    assert np.allclose(com, 0.0) and total == 120.0


def test_tilt_test_recovers_height():
    W, L, a, h, theta = 100.0, 0.5, 0.2, 0.08, np.deg2rad(25)
    f_level = W * a / L
    f_tilted = W * (a - h * np.tan(theta)) / L
    assert np.isclose(mp.com_height_from_tilt(W, L, f_level, f_tilted, theta), h)


def test_point_mass_pendulum_has_zero_com_inertia():
    m, d, g = 1.0, 0.5, mp.G
    period = 2 * np.pi * np.sqrt(d / g)  # simple pendulum
    assert np.isclose(mp.inertia_compound_pendulum(m, d, period), 0.0, atol=1e-12)


def test_reflected_inertia():
    assert np.isclose(mp.reflected_inertia(1e-5, 10), 1e-3)
