"""Evaluate trained policies with one protocol per kind of task, into one table.

    python scripts/evaluate.py outputs/ppo/flat_velocity/<run>/
    python scripts/evaluate.py outputs/*/*/*/ --out outputs/evaluations/all.csv
    python scripts/evaluate.py outputs/supermimic/motion_tracking/<run>/ --checkpoint ckpt_0020000.pt
    python scripts/evaluate.py <run>/ --seeds 1 --num-envs 16        # quick look

Velocity runs (PPO): every command in configs/eval/<eval>.yaml's command_set is held for
--segment-s seconds by --num-envs robots, for each seed. Reported: falls, velocity
tracking error per command, and the smoothness / efficiency metrics of evaluation/metrics.py.

Tracking runs (SuperMimic): the reference benchmark (every clip of the task's train and
eval groups played once, deterministically: clips failed, normalized return), plus the
same velocity-error and smoothness metrics measured while following eval clips, where
the "command" is the reference clip's base velocity.

Each run gets <run>/evaluation/<eval>_<checkpoint>.json; one row per run goes to --out
(default outputs/evaluations/<time>.csv). Smoothness columns are comparable across
kinds; returns and falls are not (different tasks).

Not covered yet from configs/eval/default.yaml: terrain_levels, max_speed_test and
perturbations (phase 5).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import time
from pathlib import Path

import numpy as np

from quadruped_rl.evaluation.render import setup_gl

setup_gl()

from quadruped_rl.evaluation import render
from quadruped_rl.evaluation.rollout import Segment, load_run, record_rollout
from quadruped_rl.utils import watchdog
from quadruped_rl.utils.config import available, load_group
from quadruped_rl.utils.paths import OUTPUT_DIR

COMMON = ["run", "algo", "task", "kind", "checkpoint", "robots", "sim_seconds", "falls",
          "falls_per_robot_minute", "velocity_rmse_xy", "velocity_rmse_wz"]


def _meta(run, rollout, segments) -> dict:
    import mujoco

    model = run.env.sim.mj_model
    root = f"robot/{run.robot.base_body}"
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
    return {"dt": float(rollout["dt"]), "mass": float(model.body_subtreemass[root_id]),
            "segments": segments, "root_body": root, "kind": run.kind}


def _rollout_stats(summaries: list[dict]) -> dict:
    """Pool per-seed rollout summaries into one set of numbers."""
    robots = sum(s["robots"] for s in summaries)
    seconds = sum(s["robots"] * s["duration_s"] for s in summaries)
    falls = sum(s["total_falls"] for s in summaries)
    seg_rows = [r for s in summaries for r in s["segments"] if "rmse_xy" in r]
    out = {"robots": robots, "sim_seconds": seconds, "falls": falls,
           "falls_per_robot_minute": 60.0 * falls / max(seconds, 1e-9)}
    if seg_rows:
        out["velocity_rmse_xy"] = float(np.sqrt(np.mean([r["rmse_xy"] ** 2 for r in seg_rows])))
        out["velocity_rmse_wz"] = float(np.sqrt(np.mean([r["rmse_wz"] ** 2 for r in seg_rows])))
    weights = [s["robots_without_falls"] for s in summaries]
    keys = {k for s in summaries for k in s["metrics_mean_over_robots_without_falls"]}
    for k in sorted(keys):
        pairs = [(s["metrics_mean_over_robots_without_falls"][k], w)
                 for s, w in zip(summaries, weights) if w and k in s["metrics_mean_over_robots_without_falls"]]
        if pairs:
            out[k] = float(sum(v * w for v, w in pairs) / sum(w for _, w in pairs))
    return out


def evaluate_velocity(run_dir: Path, args, eval_cfg: dict) -> dict:
    commands = eval_cfg["command_set"]
    schedule = [Segment(args.segment_s, tuple(float(c) for c in cmd),
                        f"vx {cmd[0]:+.1f} vy {cmd[1]:+.1f} wz {cmd[2]:+.1f}") for cmd in commands]
    segments, start = [], 0.0
    for s in schedule:
        segments.append({"label": s.label, "command": list(s.command), "start_s": start,
                         "end_s": start + s.duration_s})
        start += s.duration_s
    summaries, per_command, info = [], {}, {}
    for seed in eval_cfg["seeds"][: args.seeds]:
        run = load_run(run_dir, args.checkpoint, num_envs=args.num_envs, device=args.device, seed=seed)
        watchdog.watch_env(run.env)
        rollout = record_rollout(run, schedule, start, progress=False)
        summary = render.summarize(rollout, _meta(run, rollout, segments))
        info = {"checkpoint": run.checkpoint.name, "task": run.task["name"], "kind": run.kind}
        run.env.close()
        summaries.append(summary)
        for row in summary["segments"]:
            per_command.setdefault(row["label"], []).append(row)
    result = {**info, **_rollout_stats(summaries), "commands": [
        {"command": rows[0]["command"], "falls": sum(r["falls"] for r in rows),
         "rmse_xy": float(np.mean([r["rmse_xy"] for r in rows if "rmse_xy" in r] or [np.nan])),
         "rmse_wz": float(np.mean([r["rmse_wz"] for r in rows if "rmse_wz" in r] or [np.nan])),
         "mean_vx": float(np.mean([r["mean_vx"] for r in rows if "mean_vx" in r] or [np.nan]))}
        for rows in per_command.values()]}
    return result


def evaluate_tracking(run_dir: Path, args, eval_cfg: dict) -> dict:
    run = load_run(run_dir, args.checkpoint, num_envs=args.num_envs, device=args.device, seed=0)
    watchdog.watch_env(run.env)
    rollout = record_rollout(run, None, None, progress=False)
    summary = render.summarize(rollout, _meta(run, rollout, []))
    agent, ckpt, motions = run.agent, run.checkpoint, run.task["motions"]
    result = {"checkpoint": ckpt.name, "task": run.task["name"], "kind": run.kind,
              **_rollout_stats([summary]),
              "joint_rmse_rad": summary["tracking"]["joint_rmse_rad"],
              "root_pos_err_m": summary["tracking"]["root_pos_err_m"]}
    run.env.close()
    out_dir = run_dir / "evaluation"
    out_dir.mkdir(exist_ok=True)                 # the benchmark writes its JSON here
    for label, groups in (("eval", motions.get("eval_groups")), ("train", motions.get("train_groups"))):
        if not groups:
            continue
        b = agent.benchmark(ckpt, groups, num_envs=args.benchmark_envs,
                            out=out_dir / f"benchmark_{ckpt.stem}_{label}.json")
        result[f"{label}_clips"] = b["clips"]
        result[f"{label}_clips_failed"] = b["failed"]
        result[f"{label}_norm_return"] = b["mean_norm_return"]
        result[f"{label}_norm_return_weighted"] = b["step_weighted_norm_return"]
        result[f"{label}_worst_clips"] = b["worst"]
    return result


def describe(r: dict) -> str:
    line = (f"  {r['robots']} robots, {r['sim_seconds']:.0f} simulated s: {r['falls']} falls "
            f"({r['falls_per_robot_minute']:.2f} per robot-minute)")
    if "velocity_rmse_xy" in r:
        line += f"; velocity error xy {r['velocity_rmse_xy']:.3f} m/s, yaw {r['velocity_rmse_wz']:.3f} rad/s"
    lines = [line]
    for c in r.get("commands", []):
        cmd = c["command"]
        lines.append(f"    command {cmd[0]:+.1f} {cmd[1]:+.1f} {cmd[2]:+.1f}: mean vx {c['mean_vx']:+.2f}, "
                     f"rmse xy {c['rmse_xy']:.3f}, falls {c['falls']}")
    for label in ("eval", "train"):
        if f"{label}_clips" in r:
            lines.append(f"    {label} clips: {r[f'{label}_clips_failed']}/{r[f'{label}_clips']} failed, "
                         f"normalized return {r[f'{label}_norm_return']:.3f}")
    if r.get("joint_rmse_rad") is not None:
        lines.append(f"    joint angle RMSE {r['joint_rmse_rad']:.3f} rad, trunk position error "
                     f"{r['root_pos_err_m']:.3f} m")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="training run directories")
    p.add_argument("--eval", default="default", choices=available("eval"))
    p.add_argument("--checkpoint", help="checkpoint file name (default: each run's latest)")
    p.add_argument("--num-envs", type=int, help="robots per rollout (default: episodes_per_seed)")
    p.add_argument("--seeds", type=int, help="how many of the eval config's seeds (default: all)")
    p.add_argument("--segment-s", type=float, default=8.0, help="velocity: seconds per command")
    p.add_argument("--benchmark-envs", type=int, default=1024, help="tracking: envs for the clip benchmark")
    p.add_argument("--out", type=Path, help="CSV with one row per run")
    p.add_argument("--device", default=None)
    p.add_argument("--no-timeout", action="store_true",
                   help="no stall stack dumps (sets QUADRUPED_NO_TIMEOUT=1)")
    args = p.parse_args()

    if args.no_timeout:
        watchdog.disable_timeouts()
    eval_cfg = load_group("eval", args.eval)
    args.num_envs = args.num_envs or int(eval_cfg["episodes_per_seed"])
    args.seeds = args.seeds or len(eval_cfg["seeds"])
    watchdog.install()

    rows = []
    for run_dir in args.runs:
        run_dir = run_dir.resolve()
        if not (run_dir / "config.yaml").exists():
            print(f"skipping {run_dir}: no config.yaml (not a run directory)")
            continue
        import yaml

        with open(run_dir / "config.yaml", encoding="utf-8") as f:
            saved = yaml.safe_load(f)
        kind = saved["task"].get("kind", "velocity")
        print(f"\n{run_dir}  [{saved['algo']['name']}, {kind}]", flush=True)
        t0 = time.perf_counter()
        try:
            result = (evaluate_tracking if kind == "tracking" else evaluate_velocity)(run_dir, args, eval_cfg)
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
            continue
        result = {"run": str(run_dir), "algo": saved["algo"]["name"], **result,
                  "eval": args.eval, "evaluated": datetime.datetime.now().isoformat(timespec="seconds")}
        out_dir = run_dir / "evaluation"
        out_dir.mkdir(exist_ok=True)
        stem = Path(result["checkpoint"]).stem
        (out_dir / f"{args.eval}_{stem}.json").write_text(json.dumps(result, indent=2))
        print(describe(result) + f"\n  ({time.perf_counter() - t0:.0f} s) -> {out_dir / f'{args.eval}_{stem}.json'}")
        rows.append(result)

    if not rows:
        raise SystemExit("Nothing evaluated.")
    out = args.out or OUTPUT_DIR / "evaluations" / f"{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    def scalar(v) -> bool:
        return isinstance(v, (int, float, str)) or v is None

    extra = sorted({k for r in rows for k, v in r.items() if scalar(v) and k not in COMMON})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COMMON + extra, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nTable: {out}")


if __name__ == "__main__":
    main()
