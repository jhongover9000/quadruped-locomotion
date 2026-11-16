"""Watch a policy in the viewer.

    python scripts/play.py outputs/ppo/flat_velocity/<run>/              # latest checkpoint
    python scripts/play.py outputs/ppo/flat_velocity/<run>/ --checkpoint model_500.pt
    python scripts/play.py outputs/ppo/flat_velocity/<run>/ --command 1.0 0 0   # every robot forward
    python scripts/play.py outputs/ppo/flat_velocity/<run>/ --task rough_velocity
    python scripts/play.py --robot go1 --task flat_velocity              # no policy: zero actions
    python scripts/play.py outputs/supermimic/motion_tracking/<run>/     # tracking: holdout clips
    python scripts/play.py <supermimic run>/ --clips tencent_dog_quad_walk_002_s4

Without --command, each robot follows the task's random velocity commands (new ones every
resample_interval_s), so some walk backwards or sideways. In the viewer, the blue arrow is
the command and the cyan arrow the robot's actual velocity.

A tracking run (SuperMimic) plays reference clips instead: one clip per robot, restarted
when it ends or the robot falls (scripts/motions.py lists clip names).

The zero-action mode is the first check for a new robot: it should load, settle, and
stand in its default pose under PD control.

Viewers: "native" opens a MuJoCo window (needs a display; works in WSL2 through WSLg);
"viser" serves an interactive 3D view at http://localhost:8080 in your browser.
To save videos or watch offline instead, use scripts/record.py and scripts/replay.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from quadruped_rl.evaluation.rollout import fixed_command_task, load_run
from quadruped_rl.utils.config import available


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", type=Path, help="training run directory")
    p.add_argument("--checkpoint", help="checkpoint file name or path (default: latest)")
    p.add_argument("--robot", default="go1", choices=available("robots"))
    p.add_argument("--task", choices=available("tasks"),
                   help="default: the run's task (flat_velocity without a run)")
    p.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "WZ"),
                   help="give every robot this fixed command (m/s, m/s, rad/s)")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--clips", nargs="+", help="tracking runs: clip names to play")
    p.add_argument("--groups", help="tracking runs: manifest groups (default: the task's eval_groups)")
    p.add_argument("--viewer", choices=["auto", "native", "viser"], default="auto")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    override = (lambda task: fixed_command_task(task, args.command)) if args.command else None
    run = load_run(args.run_dir, args.checkpoint, robot_name=args.robot, task_name=args.task,
                   num_envs=args.num_envs, device=args.device, task_override=override,
                   clips=args.clips, groups=args.groups)
    if run.checkpoint is not None:
        print(f"Loading {run.checkpoint}")

    viewer = args.viewer
    if viewer == "auto":
        viewer = "native" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "viser"
    if viewer == "native":
        NativeMujocoViewer(run.viewer_env, run.policy).run()
    else:
        print("Open http://localhost:8080 in your browser.")
        ViserPlayViewer(run.viewer_env, run.policy).run()


if __name__ == "__main__":
    main()
