"""Robot description. Everything robot-specific lives in configs/robots/<name>.yaml.

Swapping the stand-in for the team's robot should need only a new YAML file (plus its
model files), never changes to environments or algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from quadruped_rl.utils.config import load_group
from quadruped_rl.utils.paths import REPO_ROOT

LEGS = ("FR", "FL", "RR", "RL")


@dataclass
class ActuatorSpec:
    model: str                    # "ideal_pd" | "analytic" | "actuator_net"
    kp: float                     # N*m/rad
    kd: float                     # N*m*s/rad
    torque_limit: list[float]     # per joint, N*m
    velocity_limit: list[float]   # per joint, rad/s
    armature: list[float]         # per joint, kg*m^2
    params: dict = field(default_factory=dict)  # model-specific: delay, friction, net path


@dataclass
class RobotConfig:
    name: str
    model_path: Path
    base_body: str
    foot_geoms: list[str]
    joint_names: list[str]
    default_joint_pos: list[float]
    base_height_nominal: float
    mass_nominal: float
    actuator: ActuatorSpec
    domain_randomization: dict = field(default_factory=dict)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def validate(self) -> None:
        n = self.num_joints
        if n == 0:
            raise ValueError(f"{self.name}: joint_names is empty")
        per_joint = {
            "default_joint_pos": self.default_joint_pos,
            "actuator.torque_limit": self.actuator.torque_limit,
            "actuator.velocity_limit": self.actuator.velocity_limit,
            "actuator.armature": self.actuator.armature,
        }
        for key, values in per_joint.items():
            if len(values) != n:
                raise ValueError(f"{self.name}: {key} has {len(values)} entries, expected {n}")
        if len(self.foot_geoms) != len(LEGS):
            raise ValueError(f"{self.name}: need {len(LEGS)} foot geoms, got {len(self.foot_geoms)}")


def _per_joint(value, n: int) -> list[float]:
    """Expand a scalar, or a per-leg [hip, thigh, calf] triple, to one value per joint."""
    if isinstance(value, (int, float)):
        return [float(value)] * n
    values = [float(v) for v in value]
    if len(values) == 3 and n == 12:
        return values * 4
    return values


def load_robot(name: str) -> RobotConfig:
    raw = load_group("robots", name)
    n = len(raw["joint_names"])
    act = raw["actuator"]
    cfg = RobotConfig(
        name=raw["name"],
        model_path=REPO_ROOT / raw["model_path"],
        base_body=raw["base_body"],
        foot_geoms=list(raw["foot_geoms"]),
        joint_names=list(raw["joint_names"]),
        default_joint_pos=_per_joint(raw["default_joint_pos"], n),
        base_height_nominal=float(raw["base_height_nominal"]),
        mass_nominal=float(raw["mass_nominal"]),
        actuator=ActuatorSpec(
            model=act["model"],
            kp=float(act["kp"]),
            kd=float(act["kd"]),
            torque_limit=_per_joint(act["torque_limit"], n),
            velocity_limit=_per_joint(act["velocity_limit"], n),
            armature=_per_joint(act.get("armature", 0.0), n),
            params=act.get("params") or {},
        ),
        domain_randomization=raw.get("domain_randomization") or {},
    )
    cfg.validate()
    return cfg
