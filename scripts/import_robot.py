"""Bring your own quadruped in: URDF -> the model and configs every script here trains on.

    python scripts/import_robot.py path/to/robot.urdf --name mydog \
        --kp 20 20 20 --kd 0.5 0.5 0.5 \
        --measurements data/sysid/processed/mydog.yaml
    python scripts/import_robot.py --sync mydog        # after editing configs/robots/mydog.yaml

From a URDF it writes (docs/robot_import.md is the guide):

    assets/robots/<name>/mjcf/<name>.xml   the model, in this project's joint convention
    assets/robots/<name>/meshes/           the meshes it uses
    assets/robots/<name>/urdf/             a copy of the URDF it came from
    assets/robots/<name>/import_report.md  what was found, changed and checked
    configs/robots/<name>.yaml             the robot config: train with --robot <name>

and, when the SuperMimic repository is next to this one, the robot's files there (retarget
config, training spec, registration), so `--algo supermimic --robot <name>` works too.

The URDF may use any link and joint names, joint frames, axis signs and zero pose. It must
be a 12-joint quadruped (per leg: hip abduction, thigh pitch, calf pitch) whose root link
has x forward, y left, z up, with a link at each foot. Joint angles in the result follow
the Go1 convention; the config records how they map back to the URDF's (``hardware:``).

Bench measurements (total mass, centre of mass, link masses, joint friction ...) go in a
small YAML passed with --measurements and replace the CAD values; see the guide.
"""

from __future__ import annotations

import argparse
import datetime
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

from quadruped_rl.robots import supermimic_port
from quadruped_rl.robots import urdf_import as ui
from quadruped_rl.robots.config import robot_from_dict
from quadruped_rl.utils.paths import ASSET_DIR, CONFIG_DIR, REPO_ROOT


def _triple(values: list[float]) -> list[float]:
    """Twelve per-joint values as [hip, thigh, calf] when every leg has the same."""
    v = [float(x) for x in values]
    return v[:3] if all(np.allclose(v[i:i + 3], v[:3]) for i in range(0, 12, 3)) else v


def _fmt(values, digits: int = 5) -> str:
    return "[" + ", ".join(f"{float(x):.{digits}g}" for x in values) + "]"


def _expand(values: list[float] | None) -> list[float] | None:
    return None if values is None else ui._per_joint(values, "option")


def parse_joint_ranges(items: list[list[str]]) -> dict[str, tuple[float, float]] | None:
    """--joint-range KIND LOWER UPPER (repeatable) -> {kind: (lower, upper)}."""
    out = {}
    for kind, lo, hi in items or []:
        if kind in out:
            raise ui.ImportError_(f"--joint-range {kind} given twice")
        try:
            out[kind] = (float(lo), float(hi))
        except ValueError:
            raise ui.ImportError_(f"--joint-range {kind} {lo} {hi}: LOWER and UPPER must be numbers (rad)") from None
    return out or None


def resolve_actuator(robot: ui.ImportedRobot, args) -> tuple[dict, list[str]]:
    """PD gains and limits per joint, and where each number came from."""
    joints = [robot.joints[j] for j in ui.JOINT_NAMES]
    notes = []
    armature = _expand(args.armature) or [j.armature for j in joints]
    torque = _expand(args.torque_limit) or [j.effort for j in joints]
    if any(t is None or t <= 0 for t in torque):
        raise ui.ImportError_("no torque limits: the URDF joints have no <limit effort=...>; pass "
                              "--torque-limit HIP THIGH CALF (N m at the joint)")
    notes.append("torque limits: " + ("--torque-limit" if args.torque_limit else "URDF <limit effort>"))
    velocity = _expand(args.velocity_limit) or [j.velocity for j in joints]
    if any(v is None or v <= 0 for v in velocity):
        raise ui.ImportError_("no velocity limits: the URDF joints have no <limit velocity=...>; pass "
                              "--velocity-limit HIP THIGH CALF (rad/s at the joint)")
    notes.append("velocity limits: " + ("--velocity-limit" if args.velocity_limit else "URDF <limit velocity>"))
    if args.kp or args.kd:
        if not (args.kp and args.kd):
            raise ui.ImportError_("pass both --kp and --kd")
        kp, kd = _expand(args.kp), _expand(args.kd)
        notes.append("PD gains: --kp / --kd")
    elif all(a > 0 for a in armature):
        wn = 2 * math.pi * args.natural_frequency
        kp = [a * wn * wn for a in armature]
        kd = [2 * args.damping_ratio * a * wn for a in armature]
        notes.append(f"PD gains: from the armature, natural frequency {args.natural_frequency} Hz, damping "
                     f"ratio {args.damping_ratio} (the rule mjlab uses for the Go1). A starting point: "
                     "set --kp / --kd to what the motor drivers will run")
    else:
        raise ui.ImportError_(
            "no PD gains: pass --kp HIP THIGH CALF and --kd HIP THIGH CALF (the gains the motor drivers "
            "will run, at the joint), or give the rotor inertia and gear ratio in the measurements file "
            "so they can be derived from the armature")
    if not all(a > 0 for a in armature):
        notes.append("armature: 0 (unknown). Geared motors need it: put rotor_inertia and gear_ratio, or "
                     "armature, under joint: in the measurements file")
    friction = [j.frictionloss for j in joints]
    damping = [j.damping for j in joints]
    continuous = _expand(getattr(args, "continuous_torque", None))
    if continuous:
        notes.append("continuous torque: --continuous-torque")
    return ({"kp": kp, "kd": kd, "torque_limit": torque, "velocity_limit": velocity, "armature": armature,
             "frictionloss": friction if any(friction) else None,
             "viscous_damping": damping if any(damping) else None,
             "continuous_torque": continuous}, notes)


def robot_yaml(robot: ui.ImportedRobot, act: dict, q0: list[float], height: float, model_path: str,
               meas: dict, source: str) -> str:
    """configs/robots/<name>.yaml as text (written by hand to keep the comments)."""
    mass = robot.total_mass
    scale = height / 0.278                                     # lengths relative to the Go1
    mu = meas.get("foot_friction")
    friction = [round(0.6 * mu, 2), round(1.4 * mu, 2)] if mu else [0.3, 1.5]
    com = round(0.025 * scale, 3)
    joints = [robot.joints[j] for j in ui.JOINT_NAMES]
    trunk_cols = [g.name for g in robot.trunk.geoms if g.collision]
    names = "\n".join(f"  - {j}" for j in ui.JOINT_NAMES)

    def opt(values):
        return "null" if values is None else _fmt(_triple(values))

    overrides = "".join(f"\n#   {n}" for n in robot.range_overrides)
    if overrides:
        overrides = "\n# Joint ranges set at import, not taken from the URDF:" + overrides
    return f"""# {robot.name}: imported from {source} by scripts/import_robot.py on {datetime.date.today().isoformat()}.
# Edit numbers here as measurements improve, then run
#   python scripts/import_robot.py --sync {robot.name}
# so the SuperMimic side follows. Report: {Path(model_path).parent.parent}/import_report.md{overrides}
name: {robot.name}
model_path: {model_path}

base_body: trunk
foot_geoms: [FR_foot_collision, FL_foot_collision, RR_foot_collision, RL_foot_collision]
foot_sites: [FR, FL, RR, RL]
collision_geoms: '.*_collision\\d*'
base_contact_geoms: [{", ".join(trunk_cols)}]       # ground contact ends the episode
undesired_contact_geoms: ['.*_thigh_collision\\d*', '.*_calf_collision\\d*']  # penalized

# Policy action / observation order. Must match the model's joint order; never reorder
# after training.
joint_names:
{names}

# Standing pose, rad, in this project's convention (hip 0 = leg vertical, thigh 0 = straight
# down, calf 0 = in line with the thigh). hardware: below maps it to the URDF's angles.
default_joint_pos: {_fmt(_triple(q0))}
base_height_nominal: {height:.4f}            # m, trunk height in that pose (also the spawn height)
mass_nominal: {mass:.4f}                   # kg, the model's total mass

actuator:
  model: ideal_pd
  kp: {_fmt(_triple(act["kp"]))}        # N*m/rad   [hip, thigh, calf] or 12 values
  kd: {_fmt(_triple(act["kd"]))}        # N*m*s/rad
  torque_limit: {_fmt(_triple(act["torque_limit"]))}   # N*m
  velocity_limit: {_fmt(_triple(act["velocity_limit"]))} # rad/s
  armature: {_fmt(_triple(act["armature"]))}  # kg*m^2, reflected rotor inertia N^2 * I_rotor
  frictionloss: {opt(act["frictionloss"])}                  # N*m per joint (null keeps the model's value)
  viscous_damping: {opt(act["viscous_damping"])}               # N*m*s/rad per joint
  continuous_torque: {opt(act.get("continuous_torque"))}             # N*m per joint held continuously (thermal); null: unknown
  params: {{}}

# Ranges for per-environment randomization: what the measurements leave uncertain.
domain_randomization:
  friction: {_fmt(friction)}                # foot sliding friction
  added_base_mass: {_fmt([round(-0.08 * mass, 2), round(0.16 * mass, 2)])}        # kg, point mass added at the trunk COM
  base_com_offset: {_fmt([-com, com])}    # m, each axis
  kp_scale: [0.9, 1.1]
  kd_scale: [0.9, 1.1]
  motor_strength: [0.9, 1.1]          # torque-limit scale
  encoder_bias: 0.015                 # rad, constant offset per joint (half-width)
  action_delay_steps: [0, 1]          # control steps of command latency
  push_interval_s: [5.0, 10.0]
  push_velocity: {round(0.5 * math.sqrt(scale), 2)}                  # m/s, planar velocity kick

# How this model's joint angles relate to the URDF's (and so to the motors, if the firmware
# uses the URDF's zero and direction):   q_urdf = offset + sign * q_sim   per joint, in
# joint_names order. The deployment code converts with this; verify it on the robot.
hardware:
  urdf_joint_names: [{", ".join(j.urdf_name for j in joints)}]
  sign: {_fmt([j.sign for j in joints])}
  offset: {_fmt([j.offset for j in joints], 8)}

# SuperMimic world-model scales, once measured (supermimic-quadruped scripts/calibrate_scales.py
# --robot {robot.name}); until then they are scaled from the Go1 by standing height.
# supermimic:
#   err_scale: [...]
#   out_scale: [...]
#   in_scale: [...]
"""


def report_md(robot: ui.ImportedRobot, act: dict, q0, height, notes: dict, drop: dict, error: float) -> str:
    joints = [robot.joints[j] for j in ui.JOINT_NAMES]
    lines = [f"# Import report: {robot.name}", "",
             f"Source: `{robot.source['urdf']}`, imported {datetime.datetime.now():%Y-%m-%d %H:%M}.", "",
             "## Legs", "",
             "| Leg | URDF joints (hip, thigh, calf) | hip to thigh, sideways (m) | thigh (m) | calf to foot centre (m) |",
             "|---|---|---|---|---|"]
    for leg in ui.LEGS:
        d, l1, l2 = robot.leg_lengths(leg)
        urdf = ", ".join(robot.joints[f"{leg}_{p}_joint"].urdf_name for p in ui.PARTS)
        lines.append(f"| {leg} | {urdf} | {d:+.4f} | {l1:.4f} | {l2:.4f} |")
    labels = (f"Foot radius {robot.foot_radius:.4f} m. Check the leg labels: FR is front right "
              "seen from behind the robot.")
    convention = ("`q_urdf = offset + sign * q_sim`. A sign of -1 means the joint turns the other way than in "
                  "the URDF; a non-zero offset means the URDF's zero is not the straight-down leg.")
    lines += ["", labels, "", "## Joint convention", "", convention, "",
              "| Joint | URDF joint | sign | offset (rad) | range here (rad) | standing pose here | standing pose, URDF angle |",
              "|---|---|---|---|---|---|---|"]
    for j, q in zip(joints, q0):
        lines.append(f"| {j.name} | {j.urdf_name} | {j.sign:+.0f} | {j.offset:+.4f} | "
                     f"{j.range[0]:+.3f} .. {j.range[1]:+.3f} | {q:+.3f} | {j.offset + j.sign * q:+.3f} |")
    if robot.range_overrides:
        lines += ["", "Joint ranges set at import, not taken from the URDF:", ""]
        lines += [f"- {n}" for n in robot.range_overrides]
    lines += ["", "## Mass", "", "| Body | mass (kg) | COM in the trunk frame, legs straight down (m) | URDF links merged |", "|---|---|---|---|"]
    for b in robot.bodies:
        lines.append(f"| {b.name} | {b.inertia.mass:.4f} | {np.round(b.inertia.com, 4).tolist()} | {', '.join(b.urdf_links)} |")
    lines += ["", f"Total {robot.total_mass:.4f} kg. Standing height {height:.4f} m.", ""]
    lines += ["## Actuators", ""] + [f"- {n}" for n in notes["actuator"]]
    lines += [(f"- kp {_fmt(_triple(act['kp']))}, kd {_fmt(_triple(act['kd']))}, torque limit "
               f"{_fmt(_triple(act['torque_limit']))} N m"), ""]
    if notes["measurements"]:
        lines += ["## Measurements applied", ""] + [f"- {n}" for n in notes["measurements"]] + [""]
    hits = ", ".join(f"{a} / {b}" for a, b in drop["self_collisions"]) or "none"
    lines += ["## Checks", "",
              ("- Rebuilt model against the URDF at 20 random poses (foot positions, centre of mass, mass): "
               f"largest difference {error:.2e} (m, kg)."),
              (f"- Standing for 2 s on a floor with these gains: {'stays up' if drop['stands'] else '**FALLS**'}; "
               f"trunk {drop['height_start']:.3f} -> {drop['height_end']:.3f} m, tilt {drop['tilt_deg']:.1f} deg, "
               f"largest joint sag {drop['joint_sag_rad']:.3f} rad, holding torque up to "
               f"{100 * drop['hold_torque_fraction']:.0f}% of the limit."),
              f"- Self-collisions in the standing pose: {hits}."]
    lines += ["", "## Motion retargeting (SuperMimic)", ""]
    lines += ([f"- **Not usable:** {p}" for p in robot.retarget_problems] or ["- Leg geometry is supported."])
    if notes.get("supermimic"):
        lines += [f"- {n}" for n in notes["supermimic"]]
    lines += ["", "## Warnings", ""] + ([f"- {w}" for w in robot.warnings] or ["- none"])
    return "\n".join(lines) + "\n"


def find_supermimic(arg: str | None) -> Path | None:
    if arg:
        return Path(arg).expanduser()
    from quadruped_rl.algorithms.supermimic import backend
    from quadruped_rl.utils.config import load_group

    return backend.find_root(load_group("algorithms", "supermimic"))


def do_sync(name: str, args) -> list[str]:
    from quadruped_rl.utils.config import load_group

    raw = load_group("robots", name)
    robot = robot_from_dict(raw)
    root = find_supermimic(args.supermimic_root)
    if root is None:
        raise supermimic_port.PortError(
            "the SuperMimic repository was not found (set SUPERMIMIC_ROOT or --supermimic-root)")
    written = supermimic_port.sync(robot, raw, root, load_group("tasks", "flat_velocity").get("action", {}))
    print(f"SuperMimic files for '{name}' written in {root}:")
    for p in written:
        print(f"  {p}")
    print(f"""Next, in {root} (see docs/robot_import.md, step 6):
  python -m pytest tests/test_quadmotion_robot.py -q
  python scripts/data/build_dataset.py --robot {name}
  python scripts/data/check_library.py --robot {name}
  python scripts/data/screen_clips.py --robot {name} --stiffness <kp>
  python scripts/data/assign_groups.py --robot {name}
  python scripts/calibrate_scales.py --robot {name} --npz-dir data/npz/{name}
then here:  python scripts/train.py --algo supermimic --robot {name}""")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("urdf", nargs="?", type=Path, help="the robot's URDF (not needed with --sync)")
    p.add_argument("--name", help="short lowercase name: becomes --robot <name>")
    p.add_argument("--sync", metavar="NAME", help="only regenerate the SuperMimic files from configs/robots/NAME.yaml")
    p.add_argument("--mesh-dir", action="append", default=[], type=Path,
                   help="folder to search for the URDF's meshes (repeatable; default: around the URDF)")
    p.add_argument("--foot-links", nargs=4, metavar="LINK", help="the four URDF links at the feet (default: found by name)")
    p.add_argument("--foot-radius", type=float, help="m; needed when the URDF feet are not spheres")
    p.add_argument("--joint-range", nargs=3, action="append", default=[], metavar=("KIND", "LOWER", "UPPER"),
                   help="replace the URDF's range of every hip, thigh or calf joint (KIND) with LOWER..UPPER, "
                        "URDF radians, the same numbers on all four legs; e.g. for a continuous knee (repeatable)")
    p.add_argument("--stance", nargs=2, type=float, metavar=("THIGH", "CALF"),
                   help="standing pose in this project's convention, rad (default: feet under the hips "
                        "at --stance-height of the leg length)")
    p.add_argument("--stance-height", type=float, default=0.62, help="default stance: fraction of the straight leg length")
    p.add_argument("--knees", choices=["back", "front"], default="back",
                   help="which way the knees bend in the default stance (SuperMimic retargeting supports one for all legs)")
    p.add_argument("--kp", nargs="+", type=float, help="PD stiffness at the joint, N m/rad: 3 [hip thigh calf] or 12")
    p.add_argument("--kd", nargs="+", type=float, help="PD damping at the joint, N m s/rad")
    p.add_argument("--natural-frequency", type=float, default=10.0, help="Hz, for gains derived from the armature")
    p.add_argument("--damping-ratio", type=float, default=2.0)
    p.add_argument("--torque-limit", nargs="+", type=float, help="N m (default: URDF <limit effort>)")
    p.add_argument("--velocity-limit", nargs="+", type=float, help="rad/s (default: URDF <limit velocity>)")
    p.add_argument("--armature", nargs="+", type=float, help="kg m^2 (default: measurements file, else 0)")
    p.add_argument("--continuous-torque", nargs="+", type=float,
                   help="N m each joint can hold continuously (thermal rating), only reported against "
                        "(default: kept from the existing config on a re-import, else unknown)")
    p.add_argument("--measurements", type=Path, help="YAML of bench measurements that replace CAD values")
    p.add_argument("--out-dir", type=Path, help="default: assets/robots/<name>")
    p.add_argument("--supermimic-root", help="SuperMimic repository (default: $SUPERMIMIC_ROOT, ../supermimic-quadruped)")
    p.add_argument("--no-supermimic", action="store_true", help="do not write the SuperMimic files")
    p.add_argument("--force", action="store_true", help="overwrite an existing configs/robots/<name>.yaml")
    args = p.parse_args()

    try:
        if args.sync:
            do_sync(args.sync, args)
            return
        if args.urdf is None or not args.name:
            p.error("give a URDF and --name (or --sync NAME)")
        run_import(args)
    except (ui.ImportError_, supermimic_port.PortError) as e:
        sys.exit(f"ERROR: {e}")


def run_import(args) -> None:
    import re

    name = args.name
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name.startswith("_"):
        raise ui.ImportError_(f"--name '{name}': use lowercase letters, digits and underscores")
    cfg_path = CONFIG_DIR / "robots" / f"{name}.yaml"
    if cfg_path.exists() and not args.force:
        raise ui.ImportError_(f"{cfg_path} exists; pass --force to replace it (its hand edits are lost), "
                              f"or --sync {name} to only regenerate the SuperMimic files")
    if not args.urdf.is_file():
        raise ui.ImportError_(f"no such file: {args.urdf}")
    out = (args.out_dir or ASSET_DIR / "robots" / name).resolve()
    mesh_dir = out / "meshes"
    meas = {}
    if args.measurements:
        meas = yaml.safe_load(args.measurements.read_text()) or {}

    robot = ui.import_urdf(args.urdf, name, mesh_dir, mesh_dirs=args.mesh_dir,
                           foot_links=args.foot_links, foot_radius=args.foot_radius,
                           joint_ranges=parse_joint_ranges(args.joint_range))
    error = ui.verify_against_urdf(robot, mesh_dir)
    if error > 1e-6:
        raise ui.ImportError_(f"internal check failed: the rebuilt model differs from the URDF by {error:.2e} "
                              "(m or kg). Nothing was written; please report this URDF.")
    notes = {"measurements": ui.apply_measurements(robot, meas)}
    knee = -1.0 if args.knees == "back" else 1.0
    stance = tuple(args.stance) if args.stance else ui.default_stance(robot, args.stance_height, knee)
    q0 = ui.stance_vector(robot, stance)
    if meas.get("com") is not None:
        notes["measurements"].append(ui.match_com(robot, mesh_dir, q0, list(meas["com"])))
    act, notes["actuator"] = resolve_actuator(robot, args)
    if act["continuous_torque"] is None and cfg_path.exists():      # a re-import keeps the rating
        old = (yaml.safe_load(cfg_path.read_text()) or {}).get("actuator") or {}
        kept = old.get("continuous_torque")
        if kept is not None:
            act["continuous_torque"] = _expand([kept] * 3 if isinstance(kept, (int, float)) else kept)
            notes["actuator"].append(f"continuous torque: kept from the previous {cfg_path.name}")
    for jn, a, f, d in zip(ui.JOINT_NAMES, act["armature"], act["frictionloss"] or [0] * 12,
                           act["viscous_damping"] or [0] * 12):
        robot.joints[jn].armature, robot.joints[jn].frictionloss, robot.joints[jn].damping = a, f, d
    model = ui.compile_mjcf(robot, mesh_dir)
    height = ui.standing_height(model, robot, q0)
    drop = ui.drop_test(robot, mesh_dir, q0, height, act["kp"], act["kd"], act["torque_limit"], act["armature"])

    (out / "mjcf").mkdir(parents=True, exist_ok=True)
    (out / "urdf").mkdir(exist_ok=True)
    xml = out / "mjcf" / f"{name}.xml"
    xml.write_text(ui.to_mjcf(robot, meshdir="../meshes"))
    if args.urdf.resolve().parent != (out / "urdf"):
        shutil.copyfile(args.urdf, out / "urdf" / args.urdf.name)
    try:
        model_path = str(xml.relative_to(REPO_ROOT))
    except ValueError:
        model_path = str(xml)
    text = robot_yaml(robot, act, q0, height, model_path, meas, args.urdf.name)
    robot_from_dict(yaml.safe_load(text))            # the config must load before it is written
    cfg_path.write_text(text)

    notes["supermimic"] = []
    if args.no_supermimic:
        notes["supermimic"].append("SuperMimic files not written (--no-supermimic).")
    elif robot.retarget_problems:
        notes["supermimic"].append("SuperMimic files not written: the leg geometry is not supported there.")
    else:
        try:
            do_sync(name, args)
            notes["supermimic"].append("SuperMimic retarget config and training spec written.")
        except supermimic_port.PortError as e:
            notes["supermimic"].append(f"SuperMimic files not written: {e}")
    (out / "import_report.md").write_text(report_md(robot, act, q0, height, notes, drop, error))

    print(f"\nImported '{name}': {robot.total_mass:.3f} kg, standing height {height:.3f} m, "
          f"legs {', '.join(f'{leg} {robot.leg_lengths(leg)[1]:.3f}+{robot.leg_lengths(leg)[2]:.3f} m' for leg in ui.LEGS)}")
    print(f"  model   {xml}\n  config  {cfg_path}\n  report  {out / 'import_report.md'}")
    print(f"  check against the URDF: largest difference {error:.1e}")
    print(f"  standing test: {'stays up' if drop['stands'] else 'FALLS'}, holding torque up to "
          f"{100 * drop['hold_torque_fraction']:.0f}% of the limit, sag {drop['joint_sag_rad']:.3f} rad")
    for n in notes["supermimic"]:
        print(f"  {n}")
    problems = robot.warnings + [f"motion retargeting: {x}" for x in robot.retarget_problems]
    if drop["self_collisions"]:
        problems.append(f"{len(drop['self_collisions'])} self-collisions in the standing pose (see the report)")
    if not drop["stands"]:
        problems.append("the robot does not hold the standing pose with these gains and torque limits")
    if problems:
        print("\nWarnings:")
        for w in problems:
            print(f"  - {w}")
    print(f"\nNext: read the report, then  python scripts/train.py --algo ppo --robot {name} --max-iterations 5")


if __name__ == "__main__":
    main()
