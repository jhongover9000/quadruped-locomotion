"""Robot descriptions loaded from configs/robots/<name>.yaml."""

from quadruped_rl.robots.config import LEGS, ActuatorSpec, RobotConfig, load_robot

__all__ = ["LEGS", "ActuatorSpec", "RobotConfig", "load_robot"]
