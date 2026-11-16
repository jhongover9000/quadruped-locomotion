"""Replay a recording from scripts/record.py: no policy, no GPU, no mjlab needed.

    python scripts/replay.py <recording_dir>                      # interactive viewer
    python scripts/replay.py <recording_dir> --robot 5            # start on robot 5
    python scripts/replay.py <recording_dir> --video robot5.mp4 --robot 5 --start 10 --end 16
    python scripts/replay.py <recording_dir> --video slow.mp4 --speed 0.25 --azimuth 90
    python scripts/replay.py <recording_dir> --plots --robot 5    # diagnostics for robot 5
    python scripts/replay.py <recording_dir> --summary            # print summary.txt again

Interactive viewer keys (the window must have focus):

    Space        pause / resume
    Right, Left  step one frame forward / back (pauses)
    Up, Down     playback speed x2 / /2
    PgDn, PgUp   next / previous robot (the camera follows it)
    Home         back to the start
    mouse        orbit / pan / zoom as in the MuJoCo viewer (the camera keeps tracking)

The viewer re-poses the saved model from the recorded states; nothing is re-simulated.
Relative paths given to --video and --plots are written inside the recording directory.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from quadruped_rl.evaluation.render import setup_gl

setup_gl()  # must run before mujoco is imported

import numpy as np

from quadruped_rl.evaluation import render

KEY_SPACE, KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 32, 262, 263, 264, 265
KEY_PAGE_UP, KEY_PAGE_DOWN, KEY_HOME = 266, 267, 268


def interactive(model, rollout: dict, meta: dict, robot: int, speed: float, neighbors: int) -> None:
    import mujoco.viewer

    T, N = rollout["qpos"].shape[:2]
    dt = meta["dt"]
    state = {"t": 0.0, "paused": False, "speed": speed, "robot": robot}

    def on_key(key: int) -> None:
        if key == KEY_SPACE:
            state["paused"] = not state["paused"]
        elif key in (KEY_RIGHT, KEY_LEFT):
            state["paused"] = True
            state["t"] = float(np.clip(int(state["t"]) + (1 if key == KEY_RIGHT else -1), 0, T - 1))
        elif key == KEY_UP:
            state["speed"] = min(state["speed"] * 2, 16.0)
        elif key == KEY_DOWN:
            state["speed"] = max(state["speed"] / 2, 1 / 16)
        elif key in (KEY_PAGE_DOWN, KEY_PAGE_UP):
            state["robot"] = (state["robot"] + (1 if key == KEY_PAGE_DOWN else -1)) % N
        elif key == KEY_HOME:
            state["t"] = 0.0

    data = mujoco.MjData(model)
    other = mujoco.MjData(model)
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, meta["root_body"])
    print(__doc__.split("Interactive viewer keys")[1].split("The viewer re-poses")[0])

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = root_id
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1.8, -15.0, 135.0
        last = time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            if not state["paused"]:
                state["t"] += (now - last) * state["speed"] / dt
                if state["t"] >= T:
                    state["t"] = 0.0          # loop
            last = now
            t, r = int(state["t"]), state["robot"]

            with viewer.lock():
                data.qpos[:] = rollout["qpos"][t, r]
                mujoco.mj_forward(model, data)
                scn = viewer.user_scn
                scn.ngeom = 0
                render.add_neighbors(model, other, opt, pert, scn, rollout, t, r, neighbors)
                render.add_velocity_arrows(scn, rollout, t, r)
                lines = render.hud_lines(rollout, meta, t, r)
                fell_at = render.last_fall(rollout, t, r)
                if fell_at is not None and (t - fell_at) * dt < render.FELL_BANNER_S:
                    lines.append("*** FELL - robot reset ***")
                status = (f"{'PAUSED' if state['paused'] else 'playing'}  x{state['speed']:g}   "
                          "[Space] pause  [<- ->] step  [Up/Down] speed  [PgUp/PgDn] robot  [Home] restart")
                viewer.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "\n".join(lines), None),
                                  (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, status, None)])
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - now)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", type=Path, help="recording directory (or its rollout.npz)")
    p.add_argument("--robot", type=int, default=0)
    p.add_argument("--video", type=Path, help="render a video (.mp4 or .gif) instead of the viewer")
    p.add_argument("--plots", action="store_true", help="write plots_robot<N>.png instead")
    p.add_argument("--summary", action="store_true", help="print the recording's summary")
    p.add_argument("--start", type=float, default=0.0, help="s, for --video")
    p.add_argument("--end", type=float, help="s, for --video")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--neighbors", type=int, default=2)
    p.add_argument("--distance", type=float, default=1.8)
    p.add_argument("--elevation", type=float, default=-15.0)
    p.add_argument("--azimuth", type=float, default=135.0)
    args = p.parse_args()

    rec_dir = args.recording.parent if args.recording.is_file() else args.recording
    model, rollout, meta = render.load_recording(rec_dir)
    N = rollout["qpos"].shape[1]
    if not 0 <= args.robot < N:
        raise SystemExit(f"--robot must be between 0 and {N - 1}")

    if args.summary:
        print(render.format_summary(render.summarize(rollout, meta), meta))
    elif args.plots:
        from quadruped_rl.evaluation.plots import plot_rollout

        out = rec_dir / f"plots_robot{args.robot}.png"
        plot_rollout(rollout, meta, args.robot, out)
        print(f"Saved {out}")
    elif args.video:
        out = args.video if args.video.is_absolute() else rec_dir / args.video
        render.render_video(model, rollout, meta, out, args.robot, start_s=args.start, end_s=args.end,
                            speed=args.speed, width=args.width, height=args.height,
                            neighbors=args.neighbors, distance=args.distance,
                            elevation=args.elevation, azimuth=args.azimuth)
        print(f"Saved {out}")
    else:
        interactive(model, rollout, meta, args.robot, args.speed, args.neighbors)


if __name__ == "__main__":
    main()
