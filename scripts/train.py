"""Train a policy.

    python scripts/train.py --algo ppo                       # default task for the algorithm
    python scripts/train.py --algo amp --task rough_velocity --robot go1 --seed 1
    python scripts/train.py --algo ppo --set rewards.action_rate=-0.02
    python scripts/train.py --algo ppo --dry-run             # print merged config, then exit

Each run writes to outputs/<algo>/<task>/<timestamp>_<robot>_s<seed>/ with the fully
resolved config saved next to its checkpoints, so every result can be reproduced.
"""

from __future__ import annotations

import argparse
import datetime

import yaml

from quadruped_rl.algorithms import get_algorithm
from quadruped_rl.robots import load_robot
from quadruped_rl.utils.config import apply_overrides, available, load_group
from quadruped_rl.utils.paths import OUTPUT_DIR
from quadruped_rl.utils.seeding import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", required=True, choices=available("algorithms"))
    p.add_argument("--task", choices=available("tasks"), help="defaults to the algorithm's default_task")
    p.add_argument("--robot", default="go1", choices=available("robots"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override task config entries, e.g. rewards.action_rate=-0.02")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    algo_cfg = load_group("algorithms", args.algo)
    task_name = args.task or algo_cfg["default_task"]
    task_cfg = apply_overrides(load_group("tasks", task_name), args.set)
    robot = load_robot(args.robot)

    resolved = {
        "algo": algo_cfg,
        "task": task_cfg,
        "robot": load_group("robots", args.robot),
        "seed": args.seed,
        "overrides": args.set,
    }
    if args.dry_run:
        print(yaml.safe_dump(resolved, sort_keys=False))
        return

    if not robot.model_path.exists():
        raise SystemExit(f"Robot model not found: {robot.model_path}\n"
                         "For the stand-in, run: python scripts/setup/download_stand_in.py")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / args.algo / task_name / f"{stamp}_{args.robot}_s{args.seed}"
    run_dir.mkdir(parents=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)

    seed_everything(args.seed)
    agent = get_algorithm(args.algo)(robot=robot, task_cfg=task_cfg, algo_cfg=algo_cfg,
                                     run_dir=run_dir)
    agent.train()


if __name__ == "__main__":
    main()
