"""Record a policy: video, diagnostics plot, summary, and a replayable rollout file.

    python scripts/record.py outputs/ppo/flat_velocity/<run>/                 # "tour" of commands
    python scripts/record.py outputs/ppo/flat_velocity/<run>/ --command 1.0 0 0 --duration 10
    python scripts/record.py outputs/ppo/flat_velocity/<run>/ --scenario random --duration 30
    python scripts/record.py outputs/ppo/flat_velocity/<run>/ --task rough_velocity   # other terrain
    python scripts/record.py outputs/ppo/flat_velocity/<run>/ --robots 0 3 --gif     # more videos
    python scripts/record.py --robot go1 --scenario stand                             # no policy
    python scripts/record.py outputs/supermimic/motion_tracking/<run>/        # tracking: holdout clips
    python scripts/record.py <supermimic run>/ --clips tencent_dog_quad_walk_002_s4 --gif

Scenarios (evaluation/rollout.py): "tour" (stand, forward, backward, sideways, turns, stand),
"forward" (0.5 -> 1.5 m/s), "stand", or "random" (the task's own command sampling).
Every robot gets the same command; --robots picks which ones to film.

A tracking run (SuperMimic) has no scenario: each robot plays one reference clip from its
first frame (--clips / --groups choose them; default the task's eval_groups). The video
shows the reference as a blue ghost, the plots compare the robot's base velocity with the
reference's, and the summary adds joint-angle and trunk-position error per clip.

Writes a recording directory (default <run>/recordings/<time>_<scenario>/):

    video.mp4       follow camera, command/actual readout, velocity arrows, fall banner
    plots.png       velocity tracking, joints, torques vs limits, actions, gait diagram
    summary.txt     falls, tracking error, torques and gait per segment, smoothness metrics
    rollout.npz, model.mjb.gz, meta.json   replay offline: python scripts/replay.py <dir>

Runs on the GPU like play.py; rendering is offscreen (no window), so it also works over SSH
or on a cluster node (MUJOCO_GL=egl is chosen automatically without a display).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

from quadruped_rl.evaluation.render import setup_gl

setup_gl()  # must run before mujoco is imported

from quadruped_rl.evaluation import render
from quadruped_rl.evaluation.plots import plot_rollout
from quadruped_rl.evaluation.rollout import SCENARIOS, load_run, make_schedule, record_rollout
from quadruped_rl.robots import load_robot
from quadruped_rl.utils.config import available


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", type=Path, help="training run directory (omit: zero actions)")
    p.add_argument("--checkpoint", help="checkpoint file name or path (default: latest)")
    p.add_argument("--robot", default="go1", choices=available("robots"), help="without a run only")
    p.add_argument("--task", choices=available("tasks"), help="default: the run's task")
    p.add_argument("--scenario", default="tour", choices=sorted(SCENARIOS) + ["random"])
    p.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "WZ"),
                   help="one fixed command for the whole rollout (overrides --scenario)")
    p.add_argument("--duration", type=float,
                   help="s, for --command and random (default 20); tracking runs: default is "
                        "the longest clip, at most 30")
    p.add_argument("--clips", nargs="+", help="tracking runs: clip names to play")
    p.add_argument("--groups", help="tracking runs: manifest groups to draw clips from "
                                    "(default: the task's eval_groups)")
    p.add_argument("--no-terminations", action="store_true",
                   help="tracking runs: do not reset when the robot strays from the reference")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--robots", type=int, nargs="+", default=[0], help="robots to film")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--neighbors", type=int, default=2, help="nearby robots drawn in each video")
    p.add_argument("--distance", type=float, default=1.8, help="camera distance, m")
    p.add_argument("--elevation", type=float, default=-15.0, help="camera elevation, deg")
    p.add_argument("--azimuth", type=float, default=135.0, help="camera azimuth, deg")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (0.25 = slow motion)")
    p.add_argument("--gif", action="store_true",
                   help="also write a GIF per video (640 px wide, 12.5 frames/s, ~1 MB per s)")
    p.add_argument("--no-video", action="store_true", help="only rollout, plots and summary")
    p.add_argument("--out", type=Path, help="recording directory")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if any(not 0 <= r < args.num_envs for r in args.robots):
        sys.exit(f"--robots must be between 0 and {args.num_envs - 1}")

    t_start = time.perf_counter()
    run = load_run(args.run_dir, args.checkpoint, robot_name=args.robot, task_name=args.task,
                   num_envs=args.num_envs, device=args.device, seed=args.seed,
                   clips=args.clips, groups=args.groups, terminations=not args.no_terminations)
    if run.kind == "tracking":
        schedule, name, duration = None, "clips", args.duration
        args.num_envs = run.env.num_envs
        args.robots = [r for r in args.robots if r < args.num_envs] or [0]
        print(f"Recording {args.num_envs} reference clips, one per robot, checkpoint {run.checkpoint}")
    else:
        duration = 20.0 if args.duration is None else args.duration
        schedule = make_schedule(args.scenario, args.command, duration)
        name = "command" if args.command else args.scenario
        print(f"Recording {args.num_envs} robots, scenario '{name}'"
              + (f", checkpoint {run.checkpoint}" if run.checkpoint else ", zero actions"))
    rollout = record_rollout(run, schedule, duration)

    model = run.env.sim.mj_model
    import mujoco

    root = f"robot/{run.robot.base_body}"
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
    limits = [float(model.actuator_forcerange[i, 1]) if model.actuator_forcelimited[i] else None
              for i in range(model.nu)]
    # Continuous (thermal) torque rating to report against. A run saved before the field
    # existed has none, so fall back to the robot's current config: it is a hardware
    # rating, not something the policy was trained with.
    continuous = run.robot.actuator.continuous_torque
    if continuous is None:
        try:
            continuous = load_robot(run.robot.name).actuator.continuous_torque
        except Exception:  # noqa: BLE001 - no such config any more: report without a rating
            continuous = None
    segments, start = [], 0.0
    for s in schedule or []:
        segments.append({"label": s.label, "command": list(s.command), "start_s": start,
                         "end_s": start + s.duration_s})
        start += s.duration_s
    meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run.run_dir) if run.run_dir else None,
        "run_name": "/".join(run.run_dir.parts[-3:]) if run.run_dir else "no policy (zero actions)",
        "checkpoint": str(run.checkpoint) if run.checkpoint else None,
        "checkpoint_name": run.checkpoint.name if run.checkpoint else "",
        "task_name": run.task["name"], "robot_name": run.robot.name, "scenario": name,
        "segments": segments, "dt": float(rollout["dt"]), "num_envs": args.num_envs,
        "seed": args.seed, "root_body": root, "mass": float(model.body_subtreemass[root_id]),
        "joint_names": list(run.robot.joint_names), "foot_names": list(run.robot.foot_sites),
        "torque_limits": limits, "command": args.command, "kind": run.kind,
        "continuous_torque": continuous,
    }

    if args.out:
        out = args.out
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = run.run_dir / "recordings" if run.run_dir else Path("outputs") / "recordings"
        out = base / (f"{stamp}_{name}" + ("" if args.task is None else f"_{args.task}"))
    render.save_recording(out, rollout, model, meta)
    run.env.close()

    summary = render.summarize(rollout, meta)
    text = render.format_summary(summary, meta)
    (out / "summary.txt").write_text(text)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + text)

    for r in args.robots:
        suffix = "" if r == 0 and len(args.robots) == 1 else f"_robot{r}"
        plot_rollout(rollout, meta, r, out / f"plots{suffix}.png")
        if not args.no_video:
            view = {"robot": r, "speed": args.speed, "width": args.width, "height": args.height,
                    "neighbors": args.neighbors, "distance": args.distance,
                    "elevation": args.elevation, "azimuth": args.azimuth}
            paths = [out / f"video{suffix}.mp4"] + ([out / f"video{suffix}.gif"] if args.gif else [])
            render.render_video(model, rollout, meta, paths, **view)

    print(f"\nSaved to {out}/  ({time.perf_counter() - t_start:.0f} s)")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:<18} {f.stat().st_size / 1e6:7.1f} MB")
    print(f"Replay offline:  python scripts/replay.py {out}")


if __name__ == "__main__":
    main()
