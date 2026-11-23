"""Holding torque of candidate standing poses: settle the robot on a flat floor and measure.

    python scripts/dev/stance_sweep.py                       # Astra, the default grid
    python scripts/dev/stance_sweep.py --robot astra --thigh 0.7 0.8 --calf -1.4 -1.6

Each pose is held for --seconds (3) under the config's PD gains, torque limit, armature
and joint frictionloss, in plain CPU MuJoCo on the generated MJCF with a floor added,
the same way scripts/import_robot.py runs its standing test (urdf_import.drop_test):
2 ms physics steps, the PD torque applied every step, the robot dropped from 5 mm above
the floor. The holding torque is the mean |PD torque| over the last --average (1) s, the
largest over the four legs, per joint kind. Each pose runs twice, with the joints'
frictionloss and without it, because in MuJoCo dry friction can carry static load that a
real joint, dithering under a walking controller, may not.

Grid (this project's joint angles, rad): thigh x calf with hip abduction 0, then at the
pair whose largest frictionless torque is smallest, hip abduction that moves each foot
inward towards the line under its abduction axis, up to the angle that puts it there.
The sign per leg is found from the model (the foot's sideways offset from the axis).

Writes a Markdown table sorted by the largest frictionless joint torque (default
outputs/debug/astra/stance_sweep.md) and prints it.
"""

from __future__ import annotations

import argparse
import datetime
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from quadruped_rl.robots import load_robot
from quadruped_rl.robots.config import LEGS
from quadruped_rl.utils.paths import REPO_ROOT

KINDS = ("hip", "thigh", "calf")


def build_model(cfg) -> mujoco.MjModel:
    """The robot's MJCF with a floor plane, mesh paths made absolute."""
    path = Path(cfg.model_file)
    root = ET.parse(path).getroot()
    for comp in root.iter("compiler"):
        for key in ("meshdir", "assetdir", "texturedir"):
            if comp.get(key) is not None:
                comp.set(key, str((path.parent / comp.get(key)).resolve()))
    ET.SubElement(root.find("worldbody"), "geom", name="floor", type="plane", size="0 0 1",
                  contype="1", conaffinity="1")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class Sweep:
    def __init__(self, cfg, timestep: float = 0.002):
        self.cfg = cfg
        self.m = build_model(cfg)
        self.m.opt.timestep = timestep
        J, G = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_GEOM
        jids = [mujoco.mj_name2id(self.m, J, n) for n in cfg.joint_names]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in jids])
        act = cfg.actuator
        self.kp, self.kd, self.lim = (np.array(x, float) for x in (act.kp, act.kd, act.torque_limit))
        self.m.dof_armature[self.dadr] = act.armature
        self.friction = (np.array(act.frictionloss, float) if act.frictionloss is not None
                         else self.m.dof_frictionloss[self.dadr].copy())
        self.floor = mujoco.mj_name2id(self.m, G, "floor")
        self.feet = [mujoco.mj_name2id(self.m, G, g) for g in cfg.foot_geoms]
        self.hips = [jids[3 * i] for i in range(len(LEGS))]

    def standing_height(self, q: np.ndarray) -> float:
        """Trunk height with the lowest foot touching the floor."""
        d = mujoco.MjData(self.m)
        d.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        return float(-min(d.geom_xpos[g][2] - self.m.geom_size[g][0] for g in self.feet))

    def foot_offsets(self, q: np.ndarray) -> np.ndarray:
        """Per leg, the foot centre's sideways (y) offset from its abduction axis, trunk level."""
        d = mujoco.MjData(self.m)
        d.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        return np.array([d.geom_xpos[f][1] - d.xanchor[h][1] for f, h in zip(self.feet, self.hips)])

    def inward_signs(self, thigh: float, calf: float) -> np.ndarray:
        """Per leg, the sign of hip abduction that moves the foot towards its axis."""
        q = np.array([0.0, thigh, calf] * len(LEGS))
        base = np.abs(self.foot_offsets(q))
        q[0::3] = 0.05
        return np.where(np.abs(self.foot_offsets(q)) < base, 1.0, -1.0)

    def under_axis_angle(self, thigh: float, calf: float, signs: np.ndarray) -> float:
        """Inward abduction (same on all legs) that puts the feet under their axes, by bisection."""
        lo, hi = 0.0, 1.2
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            q = np.array([0.0, thigh, calf] * len(LEGS))
            q[0::3] = signs * mid
            off = np.abs(self.foot_offsets(q)).mean()
            q[0::3] = signs * (mid + 1e-4)
            lo, hi = (mid, hi) if np.abs(self.foot_offsets(q)).mean() < off else (lo, mid)
        return 0.5 * (lo + hi)

    def settle(self, q0: np.ndarray, friction: bool, seconds: float, average_s: float) -> dict:
        m, d = self.m, mujoco.MjData(self.m)
        m.dof_frictionloss[self.dadr] = self.friction if friction else 0.0
        height = self.standing_height(q0)
        d.qpos[:7] = (0, 0, height + 0.005, 1, 0, 0, 0)
        d.qpos[self.qadr] = q0
        mujoco.mj_forward(m, d)
        steps, keep = int(round(seconds / m.opt.timestep)), int(round(average_s / m.opt.timestep))
        tau_log = np.zeros((keep, len(q0)))
        for i in range(steps):
            tau = np.clip(self.kp * (q0 - d.qpos[self.qadr]) - self.kd * d.qvel[self.dadr], -self.lim, self.lim)
            d.qfrc_applied[self.dadr] = tau
            mujoco.mj_step(m, d)
            if i >= steps - keep:
                tau_log[i - (steps - keep)] = tau
        hold = np.abs(tau_log).mean(0)
        up = float(d.xmat[1].reshape(3, 3)[2, 2])
        touching = set()
        for c in d.contact[: d.ncon]:
            if self.floor in (c.geom1, c.geom2):
                touching.add(c.geom2 if c.geom1 == self.floor else c.geom1)
        feet_only = touching == set(self.feet)
        return {"hold": {k: float(hold[i::3].max()) for i, k in enumerate(KINDS)},
                "height_start": height, "height_end": float(d.qpos[2]),
                "tilt_deg": float(np.degrees(np.arccos(np.clip(up, -1, 1)))),
                "sag_rad": float(np.abs(d.qpos[self.qadr] - q0).max()),
                "stands": bool(up > 0.95 and d.qpos[2] > 0.7 * height), "feet_only": feet_only,
                "touching": sorted(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
                                   for g in touching)}


def fmt_row(label: str, hip: float, thigh: float, calf: float, nf: dict, wf: dict) -> str:
    def side(r):
        h = r["hold"]
        ok = ("up" if r["stands"] else "**FALLS**") + ("" if r["feet_only"] else ", **not feet only**")
        return (f"{h['hip']:.2f} | {h['thigh']:.2f} | {h['calf']:.2f} | **{max(h.values()):.2f}** | "
                f"{r['height_end']:.3f} | {ok}")
    return f"| {label} | {hip:+.3f} | {thigh:.2f} | {calf:.2f} | {side(nf)} | {side(wf)} |"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="astra")
    p.add_argument("--thigh", nargs="+", type=float, default=[0.55, 0.65, 0.75, 0.85, 0.95])
    p.add_argument("--calf", nargs="+", type=float, default=[-1.1, -1.3, -1.5, -1.7])
    p.add_argument("--hip-inward", nargs="+", type=float, default=[0.1, 0.2, 0.3],
                   help="rad of inward abduction tried at the best pair (plus the under-axis angle)")
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--average", type=float, default=1.0, help="s at the end averaged for the torque")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "debug" / "astra" / "stance_sweep.md")
    args = p.parse_args()

    cfg = load_robot(args.robot)
    sw = Sweep(cfg)
    current = np.array(cfg.default_joint_pos, float)

    # The importer's standing test: 2 s with the config's friction, same start.
    check = sw.settle(current, friction=True, seconds=2.0, average_s=args.average)
    print(f"current stance, 2 s as the importer's standing test: trunk {check['height_start']:.3f} -> "
          f"{check['height_end']:.3f} m, tilt {check['tilt_deg']:.1f} deg, sag {check['sag_rad']:.3f} rad")

    rows = []                       # (label, hip, thigh, calf, no friction, with friction)

    def run(label, hip_signed, thigh, calf, hip_mag):
        q = np.array([0.0, thigh, calf] * len(LEGS))
        q[0::3] = hip_signed
        nf = sw.settle(q, False, args.seconds, args.average)
        wf = sw.settle(q, True, args.seconds, args.average)
        rows.append((label, hip_mag, thigh, calf, nf, wf))
        print(f"{label:<14} hip {hip_mag:+.3f} thigh {thigh:.2f} calf {calf:.2f}  "
              f"no friction {nf['hold']}  with {wf['hold']}  h {nf['height_end']:.3f}/{wf['height_end']:.3f}")
        return nf

    run("current", current[0::3], float(current[1]), float(current[2]), 0.0)
    best = None
    for thigh in args.thigh:
        for calf in args.calf:
            nf = run("grid", np.zeros(len(LEGS)), thigh, calf, 0.0)
            score = max(nf["hold"].values()) if nf["stands"] and nf["feet_only"] else np.inf
            if best is None or score < best[0]:
                best = (score, thigh, calf)
    _, bt, bc = best
    signs = sw.inward_signs(bt, bc)
    under = sw.under_axis_angle(bt, bc, signs)
    hip_rng = max(abs(cfg_r) for cfg_r in sw.m.jnt_range[sw.hips[0]]) if sw.m.jnt_limited[sw.hips[0]] else np.inf
    for mag in sorted(set([x for x in args.hip_inward if x < under] + [under])):
        mag_c = min(mag, hip_rng)
        run("hip inward" + (" (under axis)" if mag == under else ""), signs * mag_c, bt, bc, mag_c)

    rows.sort(key=lambda r: max(r[4]["hold"].values()) if r[4]["stands"] else np.inf)
    act = cfg.actuator
    trip = lambda v: "[" + ", ".join(f"{x:g}" for x in v[:3]) + "]"   # noqa: E731
    lines = [f"# Stance sweep: {cfg.name}", "",
             f"Written {datetime.datetime.now():%Y-%m-%d %H:%M} by `scripts/dev/stance_sweep.py`.", "",
             (f"Plain CPU MuJoCo on `{cfg.model_path}` with a floor; PD kp {trip(act.kp)}, kd {trip(act.kd)}, "
              f"torque limit {trip(act.torque_limit)} N m, armature {trip(act.armature)} kg m^2, "
              f"frictionloss {trip(list(sw.friction))} N m (\"with friction\") or 0 (\"no friction\"). "
              f"Each pose settles for {args.seconds:g} s from 5 mm above the floor, 2 ms steps."), "",
             (f"Holding torque: mean |PD torque| over the last {args.average:g} s, largest over the four legs, "
              "N m. Max is the largest of hip, thigh, calf. Height: trunk height at the end, m. "
              "\"up\": upright (tilt under ~18 deg) and above 70% of the start height; \"not feet only\": "
              "something other than the four feet touches the floor (or a foot does not)."), "",
             (f"Check: the current stance for 2 s with friction, as the importer's standing test: trunk "
              f"{check['height_start']:.3f} -> {check['height_end']:.3f} m, tilt {check['tilt_deg']:.1f} deg, "
              f"sag {check['sag_rad']:.3f} rad."), "",
             (f"Best thigh/calf pair with hip 0 (smallest frictionless max): thigh {bt:g}, calf {bc:g}. "
              f"Inward abduction sign per leg ({', '.join(LEGS)}): {', '.join(f'{s:+.0f}' for s in signs)}; "
              f"feet under their abduction axes at {under:.3f} rad (hip range +-{hip_rng:.3f})."), "",
             "Sorted by the largest frictionless joint torque. Hip column: inward abduction magnitude, rad.", "",
             "| pose | hip | thigh | calf | hip | thigh | calf | max | height | state "
             "| hip | thigh | calf | max | height | state |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
             "| | | | | **no friction** | | | | | | **with friction** | | | | | |"]
    lines += [fmt_row(*r) for r in rows]
    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print("\n" + text + f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
