"""Watch a policy in the viewer.

    python scripts/play.py outputs/ppo/flat_velocity/<run>/              # latest checkpoint
    python scripts/play.py outputs/ppo/flat_velocity/<run>/ --checkpoint model_500.pt
    python scripts/play.py --robot go1 --task flat_velocity              # no policy: zero actions

The zero-action mode is the first check for a new robot: it should load, settle, and
stand in its default pose under PD control.

Viewers: "native" opens a MuJoCo window (needs a display; works in WSL2 through WSLg);
"viser" serves an interactive 3D view at http://localhost:8080 in your browser.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import yaml

from quadruped_rl.algorithms import get_algorithm
from quadruped_rl.robots import load_robot, robot_from_dict
from quadruped_rl.utils.config import available, load_group


def latest_checkpoint(run_dir: Path) -> Path:
    found = list(run_dir.glob("checkpoints/model_*.pt")) + list(run_dir.glob("model_*.pt"))
    if not found:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    return max(found, key=lambda p: int(re.search(r"model_(\d+)", p.name).group(1)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", type=Path, help="training run directory")
    p.add_argument("--checkpoint", help="checkpoint file name or path (default: latest)")
    p.add_argument("--robot", default="go1", choices=available("robots"))
    p.add_argument("--task", default="flat_velocity", choices=available("tasks"))
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--viewer", choices=["auto", "native", "viser"], default="auto")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    import torch
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    from quadruped_rl.envs.builder import make_env

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.run_dir is not None:
        saved = yaml.safe_load(open(args.run_dir / "config.yaml", encoding="utf-8"))
        robot, task, algo_cfg = robot_from_dict(saved["robot"]), saved["task"], saved["algo"]
    else:
        robot, task, algo_cfg = load_robot(args.robot), load_group("tasks", args.task), None

    env = make_env(robot, task, num_envs=args.num_envs, device=device, play=True)

    if algo_cfg is None:
        num_actions = env.action_manager.total_action_dim
        viewer_env = env

        def policy(obs):
            return torch.zeros(env.num_envs, num_actions, device=device)
    else:
        ckpt = latest_checkpoint(args.run_dir) if args.checkpoint is None else Path(args.checkpoint)
        if not ckpt.is_absolute() and not ckpt.exists():
            ckpt = args.run_dir / "checkpoints" / ckpt
        print(f"Loading {ckpt}")
        agent = get_algorithm(algo_cfg["name"])(robot=robot, task_cfg=task, algo_cfg=algo_cfg,
                                                run_dir=args.run_dir, device=device)
        viewer_env, policy = agent.make_policy(env, ckpt)

    viewer = args.viewer
    if viewer == "auto":
        viewer = "native" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "viser"
    if viewer == "native":
        NativeMujocoViewer(viewer_env, policy).run()
    else:
        print("Open http://localhost:8080 in your browser.")
        ViserPlayViewer(viewer_env, policy).run()


if __name__ == "__main__":
    main()
