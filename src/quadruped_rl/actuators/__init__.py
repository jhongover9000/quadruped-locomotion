"""Actuator models: map policy joint targets to the torque the motor actually delivers."""

from quadruped_rl.actuators.base import Actuator
from quadruped_rl.actuators.ideal_pd import IdealPD


def make_actuator(spec) -> Actuator:
    """Build the actuator named by ``RobotConfig.actuator.model``."""
    if spec.model == "ideal_pd":
        return IdealPD(spec)
    if spec.model == "analytic":
        from quadruped_rl.actuators.analytic import AnalyticActuator

        return AnalyticActuator(spec)
    if spec.model == "actuator_net":
        from quadruped_rl.actuators.actuator_net import ActuatorNet

        return ActuatorNet(spec)
    raise ValueError(f"Unknown actuator model '{spec.model}'")
