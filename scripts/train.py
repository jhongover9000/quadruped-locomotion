"""Train a policy.

    python scripts/train.py --algo ppo                       # default task for the algorithm
    python scripts/train.py --algo ppo --task rough_velocity --seed 1
    python scripts/train.py --algo ppo --set rewards.action_rate=-0.02
    python scripts/train.py --algo ppo_rslrl                 # reference PPO, same environment
    python scripts/train.py --algo ppo --num-envs 64 --max-iterations 5   # quick smoke test
    python scripts/train.py --algo ppo --resume outputs/ppo/flat_velocity/<run>/checkpoints/model_500.pt
    python scripts/train.py --algo ppo --dry-run             # print the merged config, then exit

Each run writes to outputs/<algo>/<task>/<timestamp>_<robot>_s<seed>/ with the fully
resolved config saved next to its checkpoints, so every result can be reproduced.
Watch progress with:  tensorboard --logdir outputs
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import yaml

from quadruped_rl.algorithms import get_algorithm
from quadruped_rl.robots import load_robot
from quadruped_rl.utils.config import apply_overrides, available, load_group
from quadruped_rl.utils.paths import OUTPUT_DIR
from quadruped_rl.utils.seeding import seed_everything


def default_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", required=True, choices=available("algorithms"))
    p.add_argument("--task", choices=available("tasks"), help="defaults to the algorithm's default_task")
    p.add_argument("--robot", default="go1", choices=available("robots"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override task config entries, e.g. rewards.action_rate=-0.02")
    p.add_argument("--num-envs", type=int, help="override num_envs from the algorithm config")
    p.add_argument("--max-iterations", type=int, help="override max_iterations")
    p.add_argument("--device", help="cuda:0 or cpu (default: cuda:0 if available)")
    p.add_argument("--resume", type=Path, help="checkpoint to continue from")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    algo_cfg = load_group("algorithms", args.algo)
    if args.num_envs:
        algo_cfg["num_envs"] = args.num_envs
    if args.max_iterations:
        algo_cfg["max_iterations"] = args.max_iterations
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

    device = args.device or default_device()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / args.algo / task_name / f"{stamp}_{args.robot}_s{args.seed}"
    run_dir.mkdir(parents=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    print(f"Run directory: {run_dir}\nDevice: {device}")

    seed_everything(args.seed)
    agent = get_algorithm(args.algo)(robot=robot, task_cfg=task_cfg, algo_cfg=algo_cfg,
                                     run_dir=run_dir, device=device, seed=args.seed)
    agent.train(resume=args.resume)


if __name__ == "__main__":
    main()
