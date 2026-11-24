"""Run velocity-command policies on a fixed evaluation course and compare them.

    python scripts/course.py <run> [<run> ...] --course gauntlet          # one lane per robot
    python scripts/course.py <run> --course beacon --seeds 1000 1001 1002  # walk to a beacon
    python scripts/course.py <run> --course gauntlet --video --robots 0 3  # + follow-camera videos
    python scripts/course.py --robot astra_stance2 --course gauntlet --preview   # course PNGs only
    python scripts/course.py --robot go1 --course beacon --num-envs 2 --time-limit 5   # zero actions

Courses (configs/courses/<name>.yaml; envs/courses.py builds them):

    gauntlet   a straight ~55 m track of sections (low friction, ramps of 10 and 20 deg,
               rough, stairs, blocks, curbs, a slippery decline, waves), one lane per robot.
               A lane-following controller commands vx = --speed and steers back to the lane.
    beacon     a 60 x 60 m field of rocky, hilly ground with friction patches; each robot
               walks from its start to a goal 20 m away with a shared homing controller.
               --seeds choose the start and goal pairs (default: the course's default_seeds,
               disjoint from training seeds); the field itself is fixed by the YAML.

The policy only does locomotion; the controller is the same for every policy. No domain
randomization, no observation noise, deterministic actions, exact start poses. A robot's
attempt ends at its first fall (or when it leaves the terrain); later steps are not scored.
Velocity-command runs only (ppo, ppo_rslrl, amp, mpc).

Writes <out>/ (default outputs/courses/<time>_<course>/):

    results.csv    one row per robot and section (gauntlet) or per attempt (beacon), with
                   run and seed columns
    summary.txt    aggregate tables per run, then a side-by-side comparison
    meta.json      the resolved course, seeds, git hash, checkpoints, evaluation conditions
    recordings/<run>/seed<k>/   with --video: video_robot<r>.mp4 and rollout.npz,
                   model.mjb.gz, meta.json (python scripts/replay.py <dir> works on it)
    *.png          with --preview: gauntlet_top/side.png or beacon_top.png

Runs on the GPU like record.py, or on the CPU with --device cpu (slow: use few robots and
--time-limit).
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

from quadruped_rl.evaluation.render import setup_gl

setup_gl()  # must run before mujoco is imported

import numpy as np  # noqa: E402

from quadruped_rl.envs import courses as C  # noqa: E402
from quadruped_rl.evaluation import courses as E  # noqa: E402
from quadruped_rl.utils.config import available  # noqa: E402
from quadruped_rl.utils.paths import OUTPUT_DIR, REPO_ROOT  # noqa: E402


def git_hash() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=REPO_ROOT, capture_output=True, text=True, timeout=10).stdout
        return head + ("-dirty" if dirty.strip() else "")
    except Exception:  # noqa: BLE001 - not a git checkout
        return "unknown"


def run_label(run_dir: Path | None, robot_name: str) -> str:
    return run_dir.name if run_dir else f"zero_actions_{robot_name}"


def saved_algo_and_task(run_dir: Path) -> tuple[str, dict]:
    import yaml

    with open(run_dir / "config.yaml", encoding="utf-8") as f:
        saved = yaml.safe_load(f)
    return saved["algo"]["name"], saved["task"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", type=Path, help="training run directories (none: zero actions)")
    p.add_argument("--course", required=True,
                   help=f"course name ({', '.join(available('courses'))}) or a YAML path")
    p.add_argument("--num-envs", type=int, default=16, help="robots (gauntlet: lanes)")
    p.add_argument("--seeds", type=int, nargs="+", help="default: the course's default_seeds")
    p.add_argument("--speed", type=float, help="gauntlet target speed, m/s (default: the YAML's)")
    p.add_argument("--time-limit", type=float, help="s, overrides the course's time limit")
    p.add_argument("--checkpoint", nargs="+",
                   help="checkpoint per run (one name for all runs, or one per run; default latest)")
    p.add_argument("--robot", default="go1", choices=available("robots"), help="without a run only")
    p.add_argument("--task", default="flat_velocity", choices=available("tasks"),
                   help="without a run only: the task whose commands and observations to use")
    p.add_argument("--device", default=None)
    p.add_argument("--video", action="store_true", help="follow-camera videos of --robots")
    p.add_argument("--robots", type=int, nargs="+", default=[0], help="robots to film")
    p.add_argument("--preview", action="store_true", help="only draw the course (no policy, no sim)")
    p.add_argument("--out", type=Path, help="output directory")
    args = p.parse_args()

    course = C.load_course(args.course)
    kind = course["kind"]
    seeds = args.seeds or [int(s) for s in course.get("default_seeds", [0])]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or OUTPUT_DIR / "courses" / f"{stamp}_{kind}{'_preview' if args.preview else ''}"

    if args.preview:
        paths = (E.preview_gauntlet(course, out) if kind == "gauntlet" else
                 E.preview_beacon(course, out, args.num_envs, seeds[0]))
        for path in paths:
            print(f"Saved {path}")
        return

    if any(not 0 <= r < args.num_envs for r in args.robots):
        sys.exit(f"--robots must be between 0 and {args.num_envs - 1}")
    runs = args.runs or [None]
    ckpts = args.checkpoint or [None]
    if len(ckpts) not in (1, len(runs)):
        sys.exit("--checkpoint takes one name for all runs or one per run")
    ckpts = ckpts * len(runs) if len(ckpts) == 1 else ckpts
    for run_dir in args.runs:                         # refuse before building anything
        algo, task = saved_algo_and_task(run_dir)
        try:
            E.check_compatible(algo, task)
        except ValueError as e:
            sys.exit(f"{run_dir}: {e}")

    t_start = time.perf_counter()
    rows_all, per_run_rows, blocks, run_meta = [], {}, [], []
    for run_dir, ckpt in zip(runs, ckpts):
        label = run_label(run_dir, args.robot)
        rows_run, robots_run = run_course(args, course, run_dir, ckpt, seeds, label, out, run_meta)
        for r in rows_run:
            rows_all.append({"run": label, **r})
        per_run_rows[label] = rows_run
        blocks.append(E.format_gauntlet(label, rows_run, robots_run) if kind == "gauntlet"
                      else E.format_beacon(label, rows_run))

    out.mkdir(parents=True, exist_ok=True)
    E.write_csv(rows_all, out / "results.csv")
    header = (f"Course: {kind} ({args.course})   seeds: {seeds}   robots per seed: {args.num_envs}\n"
              f"Conditions: no domain randomization, no observation noise, deterministic actions, "
              f"exact start poses; an attempt ends at its first fall.\n")
    compare = (E.compare_gauntlet(per_run_rows) if kind == "gauntlet" else E.compare_beacon(per_run_rows))
    text = header + "\n" + "\n".join(blocks) + "\n" + compare
    (out / "summary.txt").write_text(text)
    meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "git": git_hash(),
            "course_name": args.course, "course": course, "seeds": seeds, "num_envs": args.num_envs,
            "speed": args.speed, "time_limit_override_s": args.time_limit, "runs": run_meta,
            "command": " ".join(sys.argv)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("\n" + text)
    print(f"Saved to {out}/  ({time.perf_counter() - t_start:.0f} s)")


def run_course(args, course: dict, run_dir: Path | None, ckpt: str | None, seeds: list[int],
               label: str, out: Path, run_meta: list) -> tuple[list[dict], list[dict]]:
    import mujoco

    from quadruped_rl.evaluation import render
    from quadruped_rl.evaluation.rollout import load_run, record_rollout
    from quadruped_rl.robots import load_robot

    kind = course["kind"]
    run = load_run(run_dir, ckpt, robot_name=args.robot, task_name=None if run_dir else args.task,
                   num_envs=args.num_envs, device=args.device, seed=seeds[0],
                   task_override=lambda t: E.course_task(t, course, args.num_envs),
                   robot_override=E.eval_robot)
    if run.kind != "velocity":
        sys.exit(f"{run_dir}: not a velocity-command run")
    env = run.env
    model = env.sim.mj_model
    root = f"robot/{run.robot.base_body}"
    mass = float(model.body_subtreemass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)])
    legs = list(run.robot.foot_sites)
    continuous = run.robot.actuator.continuous_torque
    if continuous is None:
        try:
            continuous = load_robot(run.robot.name).actuator.continuous_torque
        except Exception:  # noqa: BLE001 - no such config any more
            continuous = None
    ranges = E.command_ranges(run.task)
    dt = float(env.step_dt)
    frame = C.course_frame(course, args.num_envs)
    fld = C.beacon_field(course) if kind == "beacon" else None
    print(f"{label}: {kind}, {args.num_envs} robots, seeds {seeds}"
          + (f", checkpoint {run.checkpoint}" if run.checkpoint else ", zero actions"))

    rows, robots_all = [], []
    for seed in seeds:
        if kind == "gauntlet":
            ctrl = E.GauntletController(course, args.num_envs, ranges, dt, speed=args.speed,
                                        time_limit_s=args.time_limit)
            goals_w = None
        else:
            starts, goals = C.beacon_pairs(course, args.num_envs, seed)
            corner = np.array([frame["x0"], frame["y0"]])
            starts_w = np.column_stack([starts + corner, C.spawn_height(fld, starts)])
            goals_w = np.column_stack([goals + corner, fld.height(goals[:, 0], goals[:, 1])])
            import torch

            env.scene.terrain.env_origins[:] = torch.tensor(starts_w, dtype=torch.float,
                                                            device=env.device)
            ctrl = E.BeaconController(course, starts_w, goals_w, ranges, dt,
                                      time_limit_s=args.time_limit)
        driver = E.EnvDriver(ctrl, run.robot.base_height_nominal)
        print(f"  seed {seed}: time limit {ctrl.time_limit_s:.1f} s")
        rollout = record_rollout(run, None, ctrl.time_limit_s, controller=driver)
        if kind == "gauntlet":
            seed_rows, robots = E.gauntlet_results(rollout, ctrl, mass, legs, continuous)
            robots_all += [{"seed": seed, **r} for r in robots]
        else:
            seed_rows = E.beacon_results(rollout, ctrl, fld, frame, mass, legs, continuous)
        rows += [{"seed": seed, **r} for r in seed_rows]

        if args.video:
            rec = out / "recordings" / label / f"seed{seed}"
            meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
                    "run_dir": str(run.run_dir) if run.run_dir else None, "run_name": label,
                    "checkpoint": str(run.checkpoint) if run.checkpoint else None,
                    "checkpoint_name": run.checkpoint.name if run.checkpoint else "",
                    "task_name": run.task.get("name"), "robot_name": run.robot.name,
                    "scenario": f"course {kind}", "course": kind, "segments": [],
                    "dt": dt, "num_envs": args.num_envs, "seed": seed, "root_body": root,
                    "mass": mass, "joint_names": list(run.robot.joint_names), "foot_names": legs,
                    "continuous_torque": continuous, "kind": run.kind,
                    "goals_w": goals_w.tolist() if goals_w is not None else None,
                    "outcomes": [str(s) for s in ctrl.attempts.status]}
            render.save_recording(rec, rollout, model, meta)
            for r in args.robots:
                render.render_video(model, rollout, meta, [rec / f"video_robot{r}.mp4"], robot=r)
            print(f"  recording: {rec}  (python scripts/replay.py {rec})")

    run_meta.append({"run": label, "run_dir": str(run_dir) if run_dir else None,
                     "checkpoint": str(run.checkpoint) if run.checkpoint else None,
                     "robot": run.robot.name, "task": run.task.get("name"),
                     "command_ranges": ranges, "dt": dt, "mass": mass,
                     "conditions": E.eval_conditions(run.robot)})
    env.close()
    return rows, robots_all


if __name__ == "__main__":
    main()
