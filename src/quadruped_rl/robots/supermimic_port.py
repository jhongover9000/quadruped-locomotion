"""Write a robot from configs/robots/<name>.yaml into the SuperMimic reference repository.

SuperMimic describes a robot in its own two files (its docs/PORTING.md): a small YAML for
the motion retargeting and a Python ``RobotSpec`` for training. ``sync`` generates both
from our robot YAML and model, so the YAML stays the one place robot numbers are edited:

    <supermimic>/src/supermimic/robots/assets/<name>/   copy of the MJCF and meshes
    <supermimic>/configs/robots/<name>.yaml             retargeting: legs, stance
    <supermimic>/src/supermimic/robots/<name>.py        training spec (generated, do not edit)
    <supermimic>/src/supermimic/robots/__init__.py      one registry line
    configs/algorithms/supermimic.yaml                  backend.robot_specs entry

Run it again after any change to the robot YAML (scripts/import_robot.py --sync <name>).
Quantities SuperMimic tunes per robot size (reward widths, reset noise, push speeds,
world-model scales) are scaled from the Go1 spec by standing height; the world-model scales
should then be measured (scripts/calibrate_scales.py there) and stored under ``supermimic:``
in the robot YAML, which overrides the scaled values.
"""

from __future__ import annotations

import datetime
import math
import re
import shutil
import sys
from pathlib import Path

import mujoco

from quadruped_rl.utils.paths import REPO_ROOT

LEGS = ("FR", "FL", "RR", "RL")
GO1_STAND = 0.288          # m: the Go1 spec's standing height, the reference for scaling
SCALE_KEYS = ("err_scale", "out_scale", "in_scale")


class PortError(ValueError):
    pass


def _r(x: float, digits: int = 3) -> float:
    """Round to significant digits (generated numbers are estimates, not measurements)."""
    return float(f"{float(x):.{digits}g}")


def check_layout(robot) -> mujoco.MjModel:
    """The model must follow the layout scripts/import_robot.py writes; say what differs."""
    model = mujoco.MjModel.from_xml_path(str(robot.model_file))
    want = [f"{leg}_{part}_joint" for leg in LEGS for part in ("hip", "thigh", "calf")]
    if list(robot.joint_names) != want:
        raise PortError(f"{robot.name}: joint_names must be {want} (the layout scripts/import_robot.py "
                        "writes); SuperMimic files for other layouts are written by hand "
                        "(see supermimic-quadruped/docs/PORTING.md)")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
    if names != want:
        raise PortError(f"{robot.name}: the model's joint order {names} differs from joint_names")
    for leg in LEGS:
        for kind, n in ((mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot_collision"), (mujoco.mjtObj.mjOBJ_SITE, leg)):
            if mujoco.mj_name2id(model, kind, n) < 0:
                raise PortError(f"{robot.name}: the model has no '{n}'")
    if robot.base_body != "trunk" or mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "trunk_collision") < 0:
        raise PortError(f"{robot.name}: the base body must be 'trunk' with a box geom 'trunk_collision'")
    return model


def _stand_height(model: mujoco.MjModel, home: list[float]) -> float:
    data = mujoco.MjData(model)
    data.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    data.qpos[7:] = home
    mujoco.mj_kinematics(model, data)
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot_collision") for leg in LEGS]
    return float(-min(data.geom_xpos[g][2] - model.geom_size[g][0] for g in feet))


def retarget_yaml(robot, xml_rel: str) -> str:
    q = dict(zip(robot.joint_names, robot.default_joint_pos))
    front = (q["FR_thigh_joint"], q["FR_calf_joint"])
    legs = "\n".join(f"  {leg}: [{leg}_hip_joint, {leg}_thigh_joint, {leg}_calf_joint, {leg}]"
                     for leg in ("FL", "FR", "RL", "RR"))
    return (f"# {robot.name}. GENERATED from quadruped-locomotion configs/robots/{robot.name}.yaml by\n"
            f"# scripts/import_robot.py; edit that file and re-run with --sync {robot.name}.\n"
            f"xml: {xml_rel}\nroot_body: trunk\n"
            f"# leg: [abad joint, thigh joint, calf joint, foot site]\nlegs:\n{legs}\n"
            "geoms:\n  foot: FL_foot_collision   # sphere: its radius is the foot radius\n"
            "  trunk: trunk_collision    # box: trunk-ground check\n"
            f"knee_sign: {-1 if front[1] < 0 else 1}\n"
            f"stance: [{front[0]:.4f}, {front[1]:.4f}]   # (thigh, calf) of the standing pose\n")


def spec_source(robot, model: mujoco.MjModel, action_scale: list[float], overrides: dict) -> str:
    """Python source of the RobotSpec module."""
    name, act = robot.name, robot.actuator
    home = dict(zip(robot.joint_names, (float(v) for v in robot.default_joint_pos)))
    stand = _stand_height(model, list(home.values()))
    if abs(stand - robot.base_height_nominal) > 0.01:
        raise PortError(f"{name}: base_height_nominal {robot.base_height_nominal} m does not match the "
                        f"standing pose's height {stand:.3f} m")
    s = stand / GO1_STAND            # lengths
    v = math.sqrt(s)                 # velocities, under dynamic similarity
    dr = robot.domain_randomization or {}
    com = dr.get("base_com_offset", [-0.025 * s, 0.025 * s])
    bias = float(dr.get("encoder_bias", 0.01))
    fr = dr.get("friction", [0.3, 1.2])
    scaled = {"err_scale": (_r(0.01 * s), 0.05, _r(0.09 * v), 1.0, 0.07, 2.5),
              "out_scale": (_r(0.09 * v), 1.0, 2.5, _r(0.0007 * s), 0.006, 0.015),
              "in_scale": (_r(0.26 * s), 1.0, _r(0.3 * v), 1.0, 1.2, 2.0)}
    measured = [k for k in SCALE_KEYS if overrides.get(k) is not None]
    for k in measured:
        if len(overrides[k]) != 6:
            raise PortError(f"{name}: supermimic.{k} needs 6 values, got {len(overrides[k])}")
        scaled[k] = tuple(float(x) for x in overrides[k])
    unknown = set(overrides) - set(SCALE_KEYS)
    if unknown:
        raise PortError(f"{name}: unknown keys under supermimic: {sorted(unknown)} (known: {list(SCALE_KEYS)})")
    scales_note = ("measured (supermimic: in the robot YAML)" if len(measured) == 3 else
                   "SCALED FROM THE GO1 BY STANDING HEIGHT, not measured: run scripts/calibrate_scales.py\n"
                   f"    # --robot {name} once the motion library is built and put the result under supermimic: in the robot YAML")

    def table(values) -> str:
        rows = [", ".join(f'"{j}": {float(x):.6g}' for j, x in list(zip(robot.joint_names, values))[i:i + 3])
                for i in range(0, 12, 3)]
        return "{\n    " + ",\n    ".join(rows) + ",\n}"

    def optional(values) -> str:
        return "None" if values is None else table(values)

    pose = {"x": _r(0.018 * s), "y": _r(0.018 * s), "z": _r(0.004 * s)}
    vel = {"x": _r(0.3 * v), "y": _r(0.3 * v), "z": _r(0.13 * v)}
    return f'''"""{name} (12-DoF quadruped) spec.

GENERATED {datetime.date.today().isoformat()} by quadruped-locomotion scripts/import_robot.py from
configs/robots/{name}.yaml. Do not edit: change that YAML and run
``python scripts/import_robot.py --sync {name}`` there, so PPO and SuperMimic keep training
the same robot. The model under ``assets/{name}/`` is a copy of that project's MJCF.

Where the numbers come from:

* **[yaml]**    configs/robots/{name}.yaml: standing pose, PD gains, torque limits, armature,
  joint friction, randomisation ranges;
* **[scaled]**  the Go1 spec (``go1.py``) scaled by standing height, {stand:.3f} m vs {GO1_STAND} m:
  lengths x {s:.2f}, velocities x {v:.2f} (dynamic similarity, the rule ``go1.py`` uses);
* **[paper]**   SuperMimic Tables A.1 / A.3, unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

from supermimic.robots.spec import DomainRandomization, ObsNoise, ResetNoise, RobotSpec, SimSettings, StateScales

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "{name}"
XML = ASSET_DIR / "{name}.xml"

LEGS: tuple[str, ...] = ("FR", "FL", "RR", "RL")  # the MJCF's order
JOINT_NAMES: tuple[str, ...] = tuple(f"{{leg}}_{{j}}_joint" for leg in LEGS for j in ("hip", "thigh", "calf"))

HOME: dict[str, float] = {table(home.values())}  # [yaml] default_joint_pos
STAND_Z = {stand:.4f}  # trunk height in HOME with the feet on the floor (FK)

ROOT_BODY = "trunk"
# Feet are spheres on the calf bodies: trunk, thighs and calves fix every foot.
TRACKED_BODIES: tuple[str, ...] = (ROOT_BODY,) + tuple(f"{{leg}}_{{p}}" for p in ("thigh", "calf") for leg in LEGS)
FOOT_GEOM_REGEX = r"^(FR|FL|RR|RL)_foot_collision$"
_COLLISION_REGEX = r"{robot.collision_geoms}"

# [yaml] actuator
KP: dict[str, float] = {table(act.kp)}
KD: dict[str, float] = {table(act.kd)}
EFFORT: dict[str, float] = {table(act.torque_limit)}
ARMATURE: dict[str, float] = {table(act.armature)}
FRICTIONLOSS: dict[str, float] | None = {optional(act.frictionloss)}
VISCOUS_DAMPING: dict[str, float] | None = {optional(act.viscous_damping)}
ACTION_SCALE: dict[str, float] = {table(action_scale)}  # rad per unit action, as in the velocity tasks


def get_spec():
    import mujoco

    return mujoco.MjSpec.from_file(str(XML))


def _collision_cfg():
    """Every collision geom collides; feet with condim 6 and friction, the rest frictionless
    (mjlab's Go1 FULL_COLLISION, and what quadruped-locomotion's envs/builder.py sets)."""
    from mjlab.utils.spec_config import CollisionCfg

    return CollisionCfg(
        geom_names_expr=(_COLLISION_REGEX,),
        contype=1,
        conaffinity=1,
        solref=(0.01, 1),
        condim={{FOOT_GEOM_REGEX: 6, _COLLISION_REGEX: 1}},
        priority={{FOOT_GEOM_REGEX: 1, ".*": 0}},
        friction={{FOOT_GEOM_REGEX: (1, 5e-3, 5e-4)}},
    )


def get_entity_cfg(domain_randomization: bool = True, stiffness_scale: float = 1.0):
    """Fresh EntityCfg: one position actuator per joint with the YAML's gains.
    ``stiffness_scale`` multiplies every kp and keeps the damping ratio (kd x sqrt(scale));
    it is 1 except for the open-loop feasibility screen."""
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    del domain_randomization  # no action delay yet (spec.dr.action_delay_substeps)
    k, d = stiffness_scale, stiffness_scale**0.5
    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(re.escape(j),),
            stiffness=KP[j] * k,
            damping=KD[j] * d,
            effort_limit=EFFORT[j],
            armature=ARMATURE[j],
            frictionloss=None if FRICTIONLOSS is None else FRICTIONLOSS[j],
            viscous_damping=None if VISCOUS_DAMPING is None else VISCOUS_DAMPING[j],
        )
        for j in JOINT_NAMES
    )
    # Soft limit factor 1.0 as on the Rover: the clips are cut to the model's hard range.
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, STAND_Z + 0.01), joint_pos=dict(HOME), joint_vel={{".*": 0.0}}),
        collisions=(_collision_cfg(),),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=1.0),
    )


# Mirror plane x-z: FR <-> FL, RR <-> RL; the hip (axis x) negates, thigh and calf (axis y)
# keep their sign.
_PAIR = {{"FR": "FL", "FL": "FR", "RR": "RL", "RL": "RR"}}
MIRROR_JOINT_PERM: tuple[int, ...] = tuple(3 * LEGS.index(_PAIR[leg]) + k for leg in LEGS for k in range(3))
MIRROR_JOINT_SIGN: tuple[float, ...] = (-1.0, 1.0, 1.0) * 4


def _mirror_body_map() -> dict[str, str]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(XML))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    out = {{}}
    for n in names:
        leg = n[:2]
        partner = _PAIR[leg] + n[2:] if leg in _PAIR and n[2:3] == "_" else n
        assert partner in names, (n, partner)
        out[n] = partner
    return out


SPEC = RobotSpec(
    name="{name}",
    xml_path=XML,
    joint_names=JOINT_NAMES,
    root_body=ROOT_BODY,
    anchor_body=ROOT_BODY,
    tracked_bodies=TRACKED_BODIES,
    foot_geom_regex=FOOT_GEOM_REGEX,
    com_body=ROOT_BODY,
    action_scale=dict(ACTION_SCALE),
    entity_cfg_fn=get_entity_cfg,
    # Terminations. Height: 0.4 x standing height (Go1 0.12 / 0.288, Rover 0.15 / 0.391).
    # Tilt: 1 - cos(60 deg).
    root_height_threshold={_r(0.4 * stand, 2)},
    root_tilt_threshold=0.5,
    # World-model normalisation: {scales_note}.
    state_scales=StateScales(
        err_scale={scaled["err_scale"]},
        out_scale={scaled["out_scale"]},
        in_scale={scaled["in_scale"]},
        xi_friction=({_r((fr[0] + fr[1]) / 2)}, {_r((fr[1] - fr[0]) / 2)}),  # centre / half-width of the friction range
        xi_bias=(0.0, {bias}),
        xi_com=(0.0, {_r(max(abs(com[0]), abs(com[1])))}),
    ),
    obs_noise=ObsNoise(joint_pos=0.01, joint_vel=0.5, ang_vel=0.2, gravity=0.05),  # [paper] Table A.1
    dr=DomainRandomization(
        friction_range=({float(fr[0])}, {float(fr[1])}),  # [yaml] domain_randomization.friction
        encoder_bias={bias},  # [yaml]
        # [yaml] base_com_offset on x; y and z narrowed to 0.8 of it as on the Go1.
        com_offset=(({float(com[0])}, {float(com[1])}), ({_r(0.8 * com[0])}, {_r(0.8 * com[1])}), ({_r(0.8 * com[0])}, {_r(0.8 * com[1])})),
        action_delay_substeps=(0, 0),  # off, as on the Go1 and the Rover
        eval_friction=1.0,
    ),
    reset_noise=ResetNoise(  # [scaled]
        pose_range={{"x": (-{pose["x"]}, {pose["x"]}), "y": (-{pose["y"]}, {pose["y"]}), "z": (-{pose["z"]}, {pose["z"]}), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)}},
        velocity_range={{"x": (-{vel["x"]}, {vel["x"]}), "y": (-{vel["y"]}, {vel["y"]}), "z": (-{vel["z"]}, {vel["z"]}), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}},
        joint_position_range=(-0.1, 0.1),
    ),
    # Reward kernel stds: positions and linear velocity [scaled]; orientations and angular
    # velocity as on the Go1.
    reward_std_body_pos={_r(0.11 * s, 2)},
    reward_std_body_ori=0.4,
    reward_std_body_lin_vel={_r(0.43 * v, 2)},
    reward_std_body_ang_vel=3.14,
    reward_std_root_pos={_r(0.15 * s, 2)},
    reward_std_root_ori=0.4,
    reward_w_action_rate=-0.1,
    reward_w_joint_limit=-2.0,
    obs_term_scale=(1.0, 5.0, 2.0, 1.0, 1.0),
    goal_term_scale=(1.0, 5.0, 1.0, 2.0, 1.0, {_r(1.0 / stand, 2)}),  # root height x this -> O(1)
    # dt 0.005, decimation 4: the quadruped-locomotion tasks' 50 Hz control.
    sim=SimSettings(timestep=0.005, decimation=4, nconmax=35, njmax=250, iterations=10, ls_iterations=20),
    stand_height=STAND_Z,
    viewer_distance={_r(1.3 * s, 2)},
    # Get-up pushes [scaled].
    push_speed=({_r(2.4 * v, 2)}, {_r(3.6 * v, 2)}),
    push_ang_speed=3.0,
    push_down_speed=({_r(0.86 * v, 2)}, {_r(1.8 * v, 2)}),
    lie_height=({_r(0.06 * s, 2)}, {_r(0.15 * s, 2)}),
    mirror_joint_perm=MIRROR_JOINT_PERM,
    mirror_joint_sign=MIRROR_JOINT_SIGN,
    mirror_body_map=None,  # filled by {name}_spec() (needs the compiled model)
)


def {name}_spec(stiffness: float | None = None) -> RobotSpec:
    """The spec. ``stiffness`` (``robots.set_robot_overrides``, used by the feasibility screen)
    is the hip kp; every other kp keeps its ratio to it and every kd its damping ratio."""
    from dataclasses import replace
    from functools import partial

    spec = replace(SPEC, mirror_body_map=_mirror_body_map())
    if stiffness is not None:
        scale = float(stiffness) / KP["FR_hip_joint"]
        spec = replace(spec, entity_cfg_fn=partial(get_entity_cfg, stiffness_scale=scale))
    return spec
'''


def _insert_once(path: Path, pattern: str, line: str, present: str) -> bool:
    """Add ``line`` after the last line matching ``pattern`` unless ``present`` is already there."""
    text = path.read_text()
    if re.search(present, text, flags=re.MULTILINE):
        return False
    hits = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not hits:
        raise PortError(f"{path}: could not find where to add '{line.strip()}'; add it by hand")
    end = hits[-1].end()
    path.write_text(text[:end] + "\n" + line + text[end:])
    return True


def action_scales(robot, action_cfg: dict) -> list[float]:
    """Per-joint action scale of the velocity tasks (envs/builder.py make_action)."""
    scale = float(action_cfg.get("scale", 0.25))
    if action_cfg.get("scale_mode", "absolute") == "effort_relative":
        return [scale * tau / kp for tau, kp in zip(robot.actuator.torque_limit, robot.actuator.kp)]
    return [scale] * robot.num_joints


def sync(robot, raw: dict, supermimic_root: Path, action_cfg: dict | None = None) -> list[str]:
    """Write the robot into the SuperMimic repository. Returns the paths written.
    ``action_cfg``: the velocity task's ``action`` block, so both policies scale actions alike."""
    name = robot.name
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in ("g1", "rover", "go1", "spec"):
        raise PortError(f"robot name '{name}' cannot be used as a SuperMimic module name")
    root = Path(supermimic_root)
    robots_dir = root / "src" / "supermimic" / "robots"
    if not (robots_dir / "__init__.py").exists():
        raise PortError(f"{root} is not the SuperMimic repository (no src/supermimic/robots)")
    model = check_layout(robot)

    asset_dir = robots_dir / "assets" / name
    asset_dir.mkdir(parents=True, exist_ok=True)
    text = robot.model_file.read_text()
    meshdir = re.search(r'meshdir="([^"]*)"', text)
    src_meshes = (robot.model_file.parent / meshdir.group(1)).resolve() if meshdir else None
    files = sorted(set(re.findall(r'<mesh [^>]*file="([^"]+)"', text)))
    if files:
        (asset_dir / "meshes").mkdir(exist_ok=True)
        for f in files:
            shutil.copyfile(src_meshes / f, asset_dir / "meshes" / f)
        text = text.replace(meshdir.group(0), 'meshdir="meshes"')
    xml = asset_dir / f"{name}.xml"
    xml.write_text(text)

    cfg = root / "configs" / "robots" / f"{name}.yaml"
    cfg.write_text(retarget_yaml(robot, str(xml.relative_to(root))))
    # The retargeting's own checks (leg geometry, axes, foot sites) on what was just written.
    sys.path.insert(0, str(root / "src"))
    try:
        from quadmotion.robot import load_robot as load_retarget

        quad = load_retarget(cfg)
    except AssertionError as e:
        cfg.unlink()
        raise PortError(f"{name}: the motion retargeting cannot use this model: {e}") from e
    finally:
        sys.path.remove(str(root / "src"))
    if abs(quad.nominal_root_height() - robot.base_height_nominal) > 0.01:
        raise PortError(f"{name}: retargeting stance height {quad.nominal_root_height():.3f} m differs from "
                        f"base_height_nominal {robot.base_height_nominal} m (front and rear legs differ?)")

    spec = robots_dir / f"{name}.py"
    spec.write_text(spec_source(robot, model, action_scales(robot, action_cfg or {}), raw.get("supermimic") or {}))
    _insert_once(robots_dir / "__init__.py", r'^    "[a-z0-9_]+": "supermimic\.robots\.[a-z0-9_]+:[a-z0-9_]+",$',
                 f'    "{name}": "supermimic.robots.{name}:{name}_spec",', rf'^\s+"{name}": "supermimic\.robots\.')
    algo = REPO_ROOT / "configs" / "algorithms" / "supermimic.yaml"
    _insert_once(algo, r"^  robot_specs:.*(\n    [a-z0-9_]+: [a-z0-9_]+)*", f"    {name}: {name}",
                 rf"^    {name}: ")
    return [str(p) for p in (xml, cfg, spec, robots_dir / "__init__.py", algo)]

