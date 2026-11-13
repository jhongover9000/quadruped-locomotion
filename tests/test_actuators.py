import numpy as np

from quadruped_rl.actuators import make_actuator
from quadruped_rl.robots import load_robot


def test_ideal_pd_clips_to_torque_limit():
    robot = load_robot("go1")
    act = make_actuator(robot.actuator)
    q = np.zeros(12)
    tau = act.compute(q + 10.0, q, np.zeros(12))   # huge error -> saturate
    assert np.allclose(tau, robot.actuator.torque_limit)
