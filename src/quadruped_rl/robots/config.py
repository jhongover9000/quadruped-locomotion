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
DC_MOTOR_PARAMS = ("saturation_effort", "no_load_speed")   # actuator.params for model dc_motor


@dataclass
class ActuatorSpec:
    model: str                    # "ideal_pd" | "dc_motor" | "analytic" | "actuator_net"
    kp: list[float]               # per joint, N*m/rad
    kd: list[float]               # per joint, N*m*s/rad
    torque_limit: list[float]     # per joint, N*m
    velocity_limit: list[float]   # per joint, rad/s
    armature: list[float]         # per joint, kg*m^2
    frictionloss: list[float] | None = None     # per joint, N*m (None: keep model value)
    viscous_damping: list[float] | None = None  # per joint, N*m*s/rad
    # Torque each joint can hold continuously (thermal rating), per joint, N*m. Only
    # reported against (scripts/record.py); None: unknown.
    continuous_torque: list[float] | None = None
    # Model-specific: delay, friction, net path. dc_motor: saturation_effort (N*m, stall
    # torque at the joint) and no_load_speed (rad/s at the joint), each expanded per joint.
    params: dict = field(default_factory=dict)


@dataclass
class RobotConfig:
    name: str
    model_path: str               # repo-relative path, or "mjlab:<path inside mjlab>"
    base_body: str
    foot_geoms: list[str]
    foot_sites: list[str]
    collision_geoms: str
    base_contact_geoms: list[str]
    undesired_contact_geoms: list[str]
    joint_names: list[str]
    default_joint_pos: list[float]
    base_height_nominal: float
    mass_nominal: float
    actuator: ActuatorSpec
    domain_randomization: dict = field(default_factory=dict)
    # Imported robots: how joint angles map to the URDF's, q_urdf = offset + sign * q_sim
    # (scripts/import_robot.py); empty when the model already is the hardware convention.
    hardware: dict = field(default_factory=dict)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def model_file(self) -> Path:
        """Absolute path to the MJCF file (resolves the "mjlab:" prefix)."""
        if self.model_path.startswith("mjlab:"):
            from mjlab import MJLAB_SRC_PATH  # only needed for mjlab-bundled models

            return Path(MJLAB_SRC_PATH) / self.model_path.removeprefix("mjlab:")
        return REPO_ROOT / self.model_path

    def validate(self) -> None:
        n = self.num_joints
        if n == 0:
            raise ValueError(f"{self.name}: joint_names is empty")
        per_joint = {
            "default_joint_pos": self.default_joint_pos,
            "actuator.kp": self.actuator.kp,
            "actuator.kd": self.actuator.kd,
            "actuator.torque_limit": self.actuator.torque_limit,
            "actuator.velocity_limit": self.actuator.velocity_limit,
            "actuator.armature": self.actuator.armature,
        }
        if self.actuator.frictionloss is not None:
            per_joint["actuator.frictionloss"] = self.actuator.frictionloss
        if self.actuator.viscous_damping is not None:
            per_joint["actuator.viscous_damping"] = self.actuator.viscous_damping
        if self.actuator.continuous_torque is not None:
            per_joint["actuator.continuous_torque"] = self.actuator.continuous_torque
        if self.actuator.model == "dc_motor":
            for key in DC_MOTOR_PARAMS:
                if key not in self.actuator.params:
                    raise ValueError(f"{self.name}: actuator model dc_motor needs actuator.params.{key}")
                per_joint[f"actuator.params.{key}"] = self.actuator.params[key]
        for key, values in per_joint.items():
            if len(values) != n:
                raise ValueError(f"{self.name}: {key} has {len(values)} entries, expected {n}")
        for key in ("foot_geoms", "foot_sites"):
            if len(getattr(self, key)) != len(LEGS):
                raise ValueError(f"{self.name}: need {len(LEGS)} {key}, got {len(getattr(self, key))}")
        for key in ("urdf_joint_names", "sign", "offset"):
            if self.hardware and len(self.hardware.get(key, [])) != n:
                raise ValueError(f"{self.name}: hardware.{key} needs {n} entries")
        if min(self.actuator.kp) <= 0:
            raise ValueError(f"{self.name}: every actuator.kp must be positive")
        if self.actuator.model == "dc_motor":
            for key in DC_MOTOR_PARAMS:
                if min(self.actuator.params[key]) <= 0:
                    raise ValueError(f"{self.name}: every actuator.params.{key} must be positive")
        for key in ("frictionloss_scale", "armature_scale"):
            if key in self.domain_randomization:
                lo, hi = self.domain_randomization[key]
                if not 0 <= lo <= hi:
                    raise ValueError(f"{self.name}: domain_randomization.{key} needs 0 <= lo <= hi")


def _per_joint(value, n: int) -> list[float]:
    """Expand a scalar, or a per-leg [hip, thigh, calf] triple, to one value per joint."""
    if isinstance(value, (int, float)):
        return [float(value)] * n
    values = [float(v) for v in value]
    if len(values) == 3 and n == 12:
        return values * 4
    return values


def load_robot(name: str) -> RobotConfig:
    """Load and validate configs/robots/<name>.yaml."""
    return robot_from_dict(load_group("robots", name))


def robot_from_dict(raw: dict) -> RobotConfig:
    """Build a RobotConfig from a YAML mapping (also used for configs saved with runs)."""
    n = len(raw["joint_names"])
    act = raw["actuator"]

    def optional(key):
        return None if act.get(key) is None else _per_joint(act[key], n)

    params = dict(act.get("params") or {})
    if act["model"] == "dc_motor":
        params.update({k: _per_joint(params[k], n) for k in DC_MOTOR_PARAMS if k in params})

    cfg = RobotConfig(
        name=raw["name"],
        model_path=str(raw["model_path"]),
        base_body=raw["base_body"],
        foot_geoms=list(raw["foot_geoms"]),
        foot_sites=list(raw["foot_sites"]),
        collision_geoms=raw.get("collision_geoms", ".*"),
        base_contact_geoms=list(raw.get("base_contact_geoms") or []),
        undesired_contact_geoms=list(raw.get("undesired_contact_geoms") or []),
        joint_names=list(raw["joint_names"]),
        default_joint_pos=_per_joint(raw["default_joint_pos"], n),
        base_height_nominal=float(raw["base_height_nominal"]),
        mass_nominal=float(raw["mass_nominal"]),
        actuator=ActuatorSpec(
            model=act["model"],
            kp=_per_joint(act["kp"], n),
            kd=_per_joint(act["kd"], n),
            torque_limit=_per_joint(act["torque_limit"], n),
            velocity_limit=_per_joint(act["velocity_limit"], n),
            armature=_per_joint(act.get("armature", 0.0), n),
            frictionloss=optional("frictionloss"),
            viscous_damping=optional("viscous_damping"),
            continuous_torque=optional("continuous_torque"),
            params=params,
        ),
        domain_randomization=raw.get("domain_randomization") or {},
        hardware=raw.get("hardware") or {},
    )
    cfg.validate()
    return cfg
