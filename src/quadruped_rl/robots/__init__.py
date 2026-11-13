"""Robot descriptions loaded from configs/robots/<name>.yaml."""

from quadruped_rl.robots.config import LEGS, ActuatorSpec, RobotConfig, load_robot, robot_from_dict

__all__ = ["LEGS", "ActuatorSpec", "RobotConfig", "load_robot", "robot_from_dict"]
