"""Train a policy.

    python scripts/train.py --algo ppo                       # default task for the algorithm
    python scripts/train.py --algo ppo --task rough_velocity --seed 1
    python scripts/train.py --algo ppo --set rewards.action_rate=-0.02
    python scripts/train.py --algo ppo_rslrl                 # reference PPO, same environment
    python scripts/train.py --algo supermimic                # motion tracking, world model + BPTT
    python scripts/train.py --algo ppo --num-envs 64 --max-iterations 5   # quick smoke test
    python scripts/train.py --algo ppo --resume outputs/ppo/flat_velocity/<run>/checkpoints/model_500.pt
    python scripts/train.py --continue-run outputs/supermimic/motion_tracking/<run>/
    python scripts/train.py --algo ppo --dry-run             # print the merged config, then exit
    python scripts/train.py --list                           # which algorithm runs on which task

Each run writes to outputs/<algo>/<task>/<timestamp>_<robot>_s<seed>/ with the fully
resolved config saved next to its checkpoints, so every result can be reproduced. When
training ends (or stops at --max-hours) the run also gets training_curves.png and
status.json. Watch progress live with:  tensorboard --logdir outputs

--resume starts a new run from a checkpoint and trains --max-iterations more.
--continue-run picks an interrupted run up in place, with its saved config, and trains
only what is left of its max_iterations (used by the cluster pipeline in hpc/).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import yaml

from quadruped_rl.algorithms import (
    IncompatibleError,
    check_compatible,
    compatibility_table,
    get_algorithm,
)
from quadruped_rl.robots import load_robot, robot_from_dict
from quadruped_rl.utils.config import apply_overrides, available, load_group
from quadruped_rl.utils.paths import OUTPUT_DIR
from quadruped_rl.utils.seeding import seed_everything


def default_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def write_status(run_dir: Path, agent_cls, target: int, started: float, error: str | None = None) -> dict:
    """status.json: how far the run got. The cluster pipeline reads ``finished``."""
    from quadruped_rl.algorithms.checkpoints import checkpoint_number, list_checkpoints

    ckpts = list_checkpoints(run_dir, agent_cls.checkpoint_patterns)
    done = checkpoint_number(ckpts[-1]) if ckpts else 0
    status = {
        "iteration": done, "target_iterations": target,
        # rsl_rl numbers its last checkpoint target - 1.
        "finished": error is None and done >= target - 1,
        "latest_checkpoint": str(ckpts[-1]) if ckpts else None,
        "last_session_hours": round((time.time() - started) / 3600.0, 3),
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        "error": error,
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    return status


def plot_curves(run_dir: Path) -> None:
    """training_curves.png + progress_all.csv from whatever the trainer logged."""
    try:
        from quadruped_rl.evaluation.curves import export_csv, load_curves, plot_dashboard

        curves = load_curves(run_dir)
        plot_dashboard(curves, run_dir / "training_curves.png")
        export_csv(curves, run_dir / "progress_all.csv")
        print(f"Training curves: {run_dir / 'training_curves.png'}")
    except Exception as e:  # noqa: BLE001  (a plot must never fail a finished run)
        print(f"(could not plot training curves: {type(e).__name__}: {e})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=available("algorithms"))
    p.add_argument("--task", choices=available("tasks"), help="defaults to the algorithm's default_task")
    p.add_argument("--robot", default="go1", choices=available("robots"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override task config entries, e.g. rewards.action_rate=-0.02")
    p.add_argument("--algo-set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override algorithm config entries, e.g. policy_trainer.horizon=16")
    p.add_argument("--num-envs", type=int, help="override num_envs from the algorithm config")
    p.add_argument("--max-iterations", type=int, help="override max_iterations")
    p.add_argument("--max-hours", type=float,
                   help="stop cleanly after this many hours (continue later with --continue-run)")
    p.add_argument("--stall-timeout", type=float, metavar="SECONDS",
                   help="end the run (exit code 75) if it makes no progress for this long; "
                        "continue it with --continue-run. Default: only report stalls")
    p.add_argument("--device", help="cuda:0 or cpu (default: cuda:0 if available)")
    p.add_argument("--resume", type=Path, help="checkpoint to start a new run from")
    p.add_argument("--continue-run", type=Path, metavar="RUN_DIR",
                   help="continue an interrupted run in place, from its latest checkpoint")
    p.add_argument("--run-dir", type=Path, help="write the run here instead of outputs/<algo>/<task>/...")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", action="store_true", help="print the algorithm x task table and exit")
    args = p.parse_args()

    if args.list:
        print(compatibility_table())
        return

    from quadruped_rl.utils import watchdog

    if not args.dry_run:
        watchdog.supervise(args.stall_timeout)   # returns in the training process only

    resume, iterations = args.resume, None
    if args.continue_run is not None:
        run_dir = args.continue_run.resolve()
        with open(run_dir / "config.yaml", encoding="utf-8") as f:
            resolved = yaml.safe_load(f)
        algo_cfg, task_cfg = resolved["algo"], resolved["task"]
        algo_name, seed = algo_cfg["name"], int(resolved.get("seed", 0))
        robot = robot_from_dict(resolved["robot"])
        target = int(algo_cfg["max_iterations"])
        found = get_algorithm(algo_name).resume_checkpoint(run_dir)
        if found:
            resume, done = found
            iterations = target - done
            if iterations <= 1:
                print(f"{run_dir} already reached {done}/{target} iterations.")
                write_status(run_dir, get_algorithm(algo_name), target, time.time())
                plot_curves(run_dir)
                return
            print(f"Continuing {run_dir} from {resume.name}: {iterations} iterations left")
        else:
            print(f"Continuing {run_dir}: no checkpoint yet, starting from scratch")
        if args.max_hours is not None:
            algo_cfg["max_hours"] = args.max_hours
    else:
        if args.algo is None:
            p.error("--algo is required (or --continue-run RUN_DIR, or --list)")
        algo_name, seed = args.algo, args.seed
        algo_cfg = apply_overrides(load_group("algorithms", algo_name), args.algo_set)
        if args.num_envs:
            algo_cfg["num_envs"] = args.num_envs
        if args.max_iterations:
            algo_cfg["max_iterations"] = args.max_iterations
        if args.max_hours is not None:
            algo_cfg["max_hours"] = args.max_hours
        task_name = args.task or algo_cfg["default_task"]
        task_cfg = apply_overrides(load_group("tasks", task_name), args.set)
        robot = load_robot(args.robot)
        resolved = {
            "algo": algo_cfg,
            "task": task_cfg,
            "robot": load_group("robots", args.robot),
            "seed": seed,
            "overrides": args.set,
        }
        target = int(algo_cfg.get("max_iterations", 0))

    try:                                        # before any directory is created
        check_compatible(algo_name, task_cfg)
    except IncompatibleError as e:
        sys.exit(f"Cannot train: {e}\n\n{compatibility_table()}")
    if args.dry_run:
        print(yaml.safe_dump(resolved, sort_keys=False))
        return

    device = args.device or default_device()
    if args.continue_run is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = args.run_dir or OUTPUT_DIR / algo_name / task_cfg["name"] / f"{stamp}_{args.robot}_s{seed}"
        run_dir = Path(run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=args.run_dir is not None)
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(resolved, f, sort_keys=False)
    print(f"Run directory: {run_dir}\nDevice: {device}", flush=True)

    watchdog.install()                          # prints stacks if the run stops progressing
    seed_everything(seed)
    agent_cls = get_algorithm(algo_name)
    agent = agent_cls(robot=robot, task_cfg=task_cfg, algo_cfg=algo_cfg, run_dir=run_dir,
                      device=device, seed=seed)
    started = time.time()
    try:
        agent.train(max_iterations=iterations, resume=resume)
    except BaseException as e:
        write_status(run_dir, agent_cls, target, started, error=f"{type(e).__name__}: {e}")
        raise
    status = write_status(run_dir, agent_cls, target, started)
    plot_curves(run_dir)
    print(f"{'Finished' if status['finished'] else 'Stopped'} at iteration "
          f"{status['iteration']}/{target}. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
