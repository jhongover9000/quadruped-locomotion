"""Inspect the reference-motion library a tracking task trains on (no GPU needed).

    python scripts/motions.py                         # groups and totals
    python scripts/motions.py --group holdout         # the clips of one group
    python scripts/motions.py --check 50              # validate the first 50 clip files
    python scripts/motions.py --clip ai4a_D1_004_KAN01_001_s0

Clip names listed here can be given to  scripts/record.py <supermimic run> --clips ...
"""

from __future__ import annotations

import argparse

from quadruped_rl.algorithms.supermimic import backend
from quadruped_rl.motions.format import validate_clip
from quadruped_rl.motions.motion_lib import MotionLibrary
from quadruped_rl.robots import load_robot
from quadruped_rl.utils.config import available, load_group


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="motion_tracking", choices=available("tasks"))
    p.add_argument("--robot", default="go1", choices=available("robots"))
    p.add_argument("--group", help="list the clips of these manifest groups, e.g. holdout or train,hard")
    p.add_argument("--clip", help="describe one clip")
    p.add_argument("--check", type=int, metavar="N", help="load and validate the first N clips")
    args = p.parse_args()

    algo = load_group("algorithms", "supermimic")
    motions = load_group("tasks", args.task)["motions"]
    robot = load_robot(args.robot)
    spec = backend.spec_name(args.robot, algo)
    root = backend.find_root(algo)
    lib = MotionLibrary(backend.resolve_path(motions["dir"], root, spec),
                        backend.resolve_path(motions.get("manifest"), root, spec))
    kept = lib.kept()
    print(f"Library: {lib.directory}\n{len(lib.rows)} clips in the manifest, {len(kept)} kept")
    for group, n in sorted(lib.groups().items()):
        role = ("training" if group in motions["train_groups"].split(",")
                else "evaluation" if group in (motions.get("eval_groups") or "").split(",") else "unused")
        print(f"  {group:<10}{n:5d} clips   ({role}; incl. mirrored copies)")

    if args.clip:
        c = lib.load(args.clip)
        validate_clip(c, robot.joint_names)
        print(f"\n{c.name}: {c.frames} frames at {c.fps:.0f} Hz = {c.duration_s:.1f} s, "
              f"mean speed {c.mean_speed(robot.base_body):.2f} m/s, {len(c.body_names)} bodies")
    if args.group:
        rows = lib.summary(robot.base_body, args.group)
        print(f"\n{len(rows)} clips in '{args.group}' (mirrored copies not listed):")
        for r in rows:
            print(f"  {r['name']:<44}{r['seconds']:7.1f} s  {r['mean_speed_mps']:6.2f} m/s")
        print(f"  total {sum(r['seconds'] for r in rows) / 60:.1f} min")
    if args.check:
        names = lib.names()[: args.check]
        for name in names:
            validate_clip(lib.load(name), robot.joint_names)
        print(f"\n{len(names)} clips validated: shapes, unit quaternions, finite values, "
              f"joint order = configs/robots/{args.robot}.yaml")


if __name__ == "__main__":
    main()
