"""Save, load, render and summarize recorded rollouts (from evaluation/rollout.py).

Needs only mujoco, numpy, Pillow and imageio (no torch, mjlab or GPU), so a recording can
be replayed or re-rendered on any machine with this repo installed.

A recording is a directory:

    rollout.npz    every robot's state and signals per control step (arrays [T, N, ...])
    model.mjb.gz   the compiled MuJoCo scene the rollout ran in (terrain included)
    meta.json      run, checkpoint, task, schedule segments, joint names, ...
    video*.mp4     follow-camera videos           (scripts/record.py, scripts/replay.py --video)
    plots*.png     diagnostics for one robot      (scripts/record.py, scripts/replay.py --plots)
    summary.txt    falls, tracking error, torques and gait per segment, smoothness metrics
                   (also summary.json)

Rendering re-poses the saved model from ``qpos``; it does not re-simulate, so what you see
is exactly what happened in the rollout.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path

import numpy as np

CMD_RGBA = (0.15, 0.35, 1.0, 0.9)      # commanded planar velocity
ACT_RGBA = (0.0, 0.85, 1.0, 0.9)       # actual planar velocity
FELL_BANNER_S = 1.0


def setup_gl() -> None:
    """Choose an OpenGL backend for offscreen rendering before mujoco is imported.

    GLFW works under WSL2 (through WSLg) and on desktops; EGL on headless GPU machines
    such as a cluster node. An explicit MUJOCO_GL is left alone.
    """
    if "MUJOCO_GL" not in os.environ:
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        os.environ["MUJOCO_GL"] = "glfw" if has_display else "egl"


# ---- files --------------------------------------------------------------------------

def save_recording(out_dir: Path, rollout: dict, model, meta: dict) -> None:
    import mujoco

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "rollout.npz", **rollout)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.mjb"
        mujoco.mj_saveModel(model, str(path), None)
        with open(path, "rb") as src, gzip.open(out_dir / "model.mjb.gz", "wb", compresslevel=6) as dst:
            dst.write(src.read())
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_recording(path: Path):
    """Returns (model, rollout dict, meta dict). ``path`` is the directory or its rollout.npz."""
    import mujoco

    rec_dir = Path(path)
    if rec_dir.is_file():
        rec_dir = rec_dir.parent
    with np.load(rec_dir / "rollout.npz") as f:
        rollout = {k: f[k] for k in f.files}
    meta = json.loads((rec_dir / "meta.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        mjb = Path(tmp) / "model.mjb"
        with gzip.open(rec_dir / "model.mjb.gz", "rb") as src:
            mjb.write_bytes(src.read())
        model = mujoco.MjModel.from_binary_path(str(mjb))
    return model, rollout, meta


def segment_at(meta: dict, t: int) -> dict | None:
    time_s = t * meta["dt"]
    for seg in meta.get("segments") or []:
        if seg["start_s"] <= time_s < seg["end_s"]:
            return seg
    segs = meta.get("segments") or []
    return segs[-1] if segs else None


def last_fall(rollout: dict, t: int, robot: int) -> int | None:
    """Step of the most recent fall of ``robot`` at or before ``t``, or None."""
    falls = np.flatnonzero(rollout["fell"][: t + 1, robot])
    return int(falls[-1]) if falls.size else None


# ---- rendering ----------------------------------------------------------------------

def _yaw_rotate(quat_wxyz: np.ndarray, v_xy: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v_xy[0] - s * v_xy[1], s * v_xy[0] + c * v_xy[1], 0.0])


def add_arrow(scene, p0, p1, rgba, width: float = 0.012) -> None:
    import mujoco

    if scene.ngeom >= scene.maxgeom or np.linalg.norm(np.subtract(p1, p0)) < 1e-3:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(p0, float),
                         np.asarray(p1, float))
    scene.ngeom += 1


NEIGHBOR_ALPHA = 0.3     # robots run in separate simulation worlds and pass through each other
REFERENCE_RGBA = (0.18, 0.48, 0.98, 0.35)   # tracking: the reference motion, as a blue ghost


def add_neighbors(model, data, opt, pert, scene, rollout: dict, t: int, robot: int, count: int) -> None:
    """Draw the ``count`` robots nearest to ``robot`` as translucent ghosts, for context.
    In a tracking recording, draw the robot's reference pose instead (blue ghost)."""
    import mujoco

    if "ref_qpos" in rollout:       # tracking: the ghost is this robot's reference pose
        data.qpos[:] = rollout["ref_qpos"][t, robot]
        mujoco.mj_kinematics(model, data)
        first = scene.ngeom
        mujoco.mjv_addGeoms(model, data, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC.value, scene)
        for g in range(first, scene.ngeom):
            scene.geoms[g].rgba[:] = REFERENCE_RGBA
        return
    if count <= 0:
        return
    xy = rollout["qpos"][t, :, :2]
    order = np.argsort(np.linalg.norm(xy - xy[robot], axis=1))
    for other in [i for i in order if i != robot][:count]:
        data.qpos[:] = rollout["qpos"][t, other]
        mujoco.mj_kinematics(model, data)
        first = scene.ngeom
        mujoco.mjv_addGeoms(model, data, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC.value, scene)
        for g in range(first, scene.ngeom):
            scene.geoms[g].rgba[3] = min(scene.geoms[g].rgba[3], NEIGHBOR_ALPHA)


def add_velocity_arrows(scene, rollout: dict, t: int, robot: int) -> None:
    """Commanded (blue) and actual (cyan) planar base velocity, drawn above the robot;
    1 m of arrow per 1 m/s."""
    qpos = rollout["qpos"][t, robot]
    base = qpos[:3] + np.array([0.0, 0.0, 0.25])
    cmd = _yaw_rotate(qpos[3:7], rollout["command"][t, robot, :2])
    act = _yaw_rotate(qpos[3:7], rollout["base_lin_vel_b"][t, robot, :2])
    add_arrow(scene, base, base + cmd, CMD_RGBA)
    add_arrow(scene, base + [0, 0, 0.02], base + [0, 0, 0.02] + act, ACT_RGBA)


def hud_lines(rollout: dict, meta: dict, t: int, robot: int) -> list[str]:
    cmd = rollout["command"][t, robot]
    v, w = rollout["base_lin_vel_b"][t, robot], rollout["base_ang_vel_b"][t, robot]
    falls = int(rollout["fell"][: t + 1, robot].sum())
    head = [
        f"{meta.get('run_name', 'no policy')}   {meta.get('checkpoint_name', '')}",
        f"t {t * meta['dt']:6.2f} s   robot {robot}/{rollout['qpos'].shape[1]}   falls so far {falls}",
    ]
    if "ref_qpos" in rollout:
        name = str(rollout["clip_names"][robot]) if "clip_names" in rollout else "?"
        frame, length = int(rollout["clip_frame"][t, robot]), int(rollout["clip_length"][robot])
        return head + [
            f"clip: {name}   frame {frame}/{length}",
            f"reference  vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}",
            f"actual     vx {v[0]:+.2f}  vy {v[1]:+.2f}  wz {w[2]:+.2f}",
            "blue ghost = reference motion;  arrows: blue = reference velocity, cyan = actual",
        ]
    seg = segment_at(meta, t)
    return head + [
        f"segment: {seg['label'] if seg else 'task commands (random)'}",
        f"command  vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}",
        f"actual   vx {v[0]:+.2f}  vy {v[1]:+.2f}  wz {w[2]:+.2f}",
        "arrows: blue = command, cyan = actual;  ghosts = nearby robots",
    ]


class RolloutRenderer:
    """Offscreen renderer: camera tracks one robot, nearby robots are drawn for context."""

    def __init__(self, model, rollout: dict, meta: dict, width: int = 1280, height: int = 720,
                 neighbors: int = 2, distance: float = 1.8, elevation: float = -15.0,
                 azimuth: float = 135.0, hud: bool = True, arrows: bool = True):
        import mujoco

        self.mj = mujoco
        self.model, self.rollout, self.meta = model, rollout, meta
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.data = mujoco.MjData(model)
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, meta["root_body"])
        self.cam.distance, self.cam.elevation, self.cam.azimuth = distance, elevation, azimuth
        self.neighbors, self.hud, self.arrows = neighbors, hud, arrows
        self.width, self.height = width, height
        self._font = None

    def _pose(self, t: int, robot: int) -> None:
        self.data.qpos[:] = self.rollout["qpos"][t, robot]
        self.data.qvel[:] = 0.0
        self.mj.mj_forward(self.model, self.data)

    def frame(self, t: int, robot: int) -> np.ndarray:
        self._pose(t, robot)
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        scene = self.renderer.scene
        add_neighbors(self.model, self.data, self.opt, self.pert, scene, self.rollout, t, robot,
                      self.neighbors)
        if self.arrows:
            add_velocity_arrows(scene, self.rollout, t, robot)
        img = self.renderer.render()
        return self._overlay(img, t, robot) if self.hud else img

    def _overlay(self, img: np.ndarray, t: int, robot: int) -> np.ndarray:
        from PIL import Image, ImageDraw, ImageFont

        if self._font is None:
            size = max(12, self.height // 40)
            try:
                self._font = ImageFont.load_default(size=size)
            except TypeError:  # Pillow < 10.1
                self._font = ImageFont.load_default()
            self._line_h = size + 4
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im, "RGBA")
        lines = hud_lines(self.rollout, self.meta, t, robot)
        box_w = int(self.width * 0.46)
        draw.rectangle([8, 8, 8 + box_w, 14 + self._line_h * len(lines)], fill=(0, 0, 0, 140))
        for i, line in enumerate(lines):
            draw.text((16, 12 + i * self._line_h), line, font=self._font, fill=(255, 255, 255, 255))
        fell_at = last_fall(self.rollout, t, robot)
        if fell_at is not None and (t - fell_at) * self.meta["dt"] < FELL_BANNER_S:
            msg = "FELL - robot reset"
            x = self.width // 2 - 100
            draw.rectangle([x - 10, 12, x + 230, 16 + self._line_h], fill=(200, 0, 0, 200))
            draw.text((x, 14), msg, font=self._font, fill=(255, 255, 255, 255))
        # Timeline: segment boundaries and the current time.
        T = self.rollout["qpos"].shape[0]
        y0, y1 = self.height - 14, self.height - 6
        draw.rectangle([8, y0, self.width - 8, y1], fill=(0, 0, 0, 120))
        for seg in self.meta.get("segments") or []:
            x = 8 + (self.width - 16) * seg["start_s"] / (T * self.meta["dt"])
            draw.line([x, y0 - 3, x, y1 + 3], fill=(255, 255, 255, 200), width=1)
        draw.rectangle([8, y0, 8 + (self.width - 16) * (t + 1) / T, y1], fill=(80, 170, 255, 220))
        return np.asarray(im)

    def close(self) -> None:
        self.renderer.close()


GIF_WIDTH = 640          # px; the HUD stays legible. GIFs are for chat and slides, MP4s for detail
GIF_STRIDE = 4           # every 4th control step: 12.5 frames/s of a 50 Hz rollout


class _VideoSink:
    """One output file. MP4: every frame at full size. GIF: every GIF_STRIDE-th frame,
    scaled to GIF_WIDTH (roughly 1 MB per second of rollout; cut clips with
    scripts/replay.py --start/--end)."""

    def __init__(self, path: Path, fps: float):
        import imageio.v2 as imageio

        self.path = Path(path)
        self.gif = self.path.suffix.lower() == ".gif"
        self.stride = GIF_STRIDE if self.gif else 1
        if self.gif:
            self.writer = imageio.get_writer(self.path, mode="I", duration=1000.0 * self.stride / fps,
                                             loop=0)
        else:
            self.writer = imageio.get_writer(self.path, fps=fps, codec="libx264", quality=7,
                                             pixelformat="yuv420p", macro_block_size=8)

    def add(self, i: int, frame: np.ndarray) -> None:
        if i % self.stride:
            return
        if self.gif:
            from PIL import Image

            h = round(frame.shape[0] * GIF_WIDTH / frame.shape[1])
            frame = np.asarray(Image.fromarray(frame).resize((GIF_WIDTH, h), Image.LANCZOS))
        self.writer.append_data(frame)

    def close(self) -> None:
        self.writer.close()


def render_video(model, rollout: dict, meta: dict, paths, robot: int = 0, *,
                 start_s: float = 0.0, end_s: float | None = None, speed: float = 1.0,
                 width: int = 1280, height: int = 720, neighbors: int = 2,
                 distance: float = 1.8, elevation: float = -15.0, azimuth: float = 135.0,
                 progress: bool = True) -> None:
    """Follow-camera video of one robot into one or more files (.mp4 / .gif), rendering
    each frame once. ``speed`` < 1 plays in slow motion."""
    paths = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    dt = meta["dt"]
    T = rollout["qpos"].shape[0]
    t0 = int(start_s / dt)
    t1 = T if end_s is None else min(T, int(end_s / dt))
    fps = max(1.0, speed / dt)
    r = RolloutRenderer(model, rollout, meta, width, height, neighbors, distance, elevation, azimuth)
    sinks = [_VideoSink(p, fps) for p in paths]
    names = ", ".join(p.name for p in paths)
    try:
        n = t1 - t0
        for i, t in enumerate(range(t0, t1)):
            if progress and (i % 250 == 0 or i == n - 1):
                print(f"  rendering {names}: frame {i + 1}/{n}", flush=True)
            frame = r.frame(t, robot)
            for s in sinks:
                s.add(i, frame)
    finally:
        for s in sinks:
            s.close()
        r.close()


# ---- summary ------------------------------------------------------------------------

def valid_mask(rollout: dict, meta: dict, settle_s: float = 1.0) -> np.ndarray:
    """[T, N] True where a robot has been in the current segment and alive for settle_s,
    so tracking errors don't count the speed-up after a command change or a reset."""
    T, N = rollout["fell"].shape
    if "ref_qpos" in rollout:       # tracking episodes start on the reference: nothing to settle
        settle_s = min(settle_s, 0.1)
    k = round(settle_s / meta["dt"])
    ok = np.ones((T, N), dtype=bool)
    starts = [round(s["start_s"] / meta["dt"]) for s in meta.get("segments") or []] or [0]
    for s in starts:
        ok[s: s + k] = False
    resets = rollout["fell"].astype(bool)
    if "timeout" in rollout:        # clip restarts, walking off the generated terrain
        resets = resets | rollout["timeout"].astype(bool)
    for n in range(N):
        for f in np.flatnonzero(resets[:, n]):
            ok[f: f + k + 1, n] = False
    return ok


def tracking_errors(rollout: dict, meta: dict) -> dict:
    """How closely each robot followed its clip: joint-angle and trunk-position error
    against the reference, over the steps ``valid_mask`` keeps."""
    ok = valid_mask(rollout, meta)
    ref, qpos = rollout["ref_qpos"], rollout["qpos"]
    joint_err = rollout["joint_pos"] - ref[:, :, 7:]
    root_err = np.linalg.norm(qpos[:, :, :3] - ref[:, :, :3], axis=-1)
    fell = rollout["fell"].astype(bool)
    clips = []
    for n in range(ref.shape[1]):
        m = ok[:, n]
        first_fall = np.flatnonzero(fell[:, n])
        clips.append({
            "clip": str(rollout["clip_names"][n]), "frames": int(rollout["clip_length"][n]),
            "falls": int(fell[:, n].sum()),
            "first_fall_s": float(first_fall[0] * meta["dt"]) if len(first_fall) else None,
            "joint_rmse_rad": float(np.sqrt((joint_err[m, n] ** 2).mean())) if m.any() else None,
            "root_pos_err_m": float(root_err[m, n].mean()) if m.any() else None,
        })
    return {
        "joint_rmse_rad": float(np.sqrt((joint_err[ok] ** 2).mean())) if ok.any() else None,
        "root_pos_err_m": float(root_err[ok].mean()) if ok.any() else None,
        "clips_without_falls": sum(c["falls"] == 0 for c in clips), "clips": clips,
    }


JOINT_KINDS = ("hip", "thigh", "calf")


def torque_stats(torques: np.ndarray, mask: np.ndarray, legs: list[str],
                 continuous: list[float] | None = None) -> dict | None:
    """RMS and peak |torque| per joint kind over the steps ``mask`` keeps, from
    ``torques`` [T, N, 12] in the project's joint order (legs x hip, thigh, calf).
    Each kind gives the largest value over the legs and the per-leg values; with a
    per-joint ``continuous`` rating, ``over_continuous`` marks an RMS above it."""
    if torques.shape[-1] != 3 * len(legs) or not mask.any():
        return None
    tau = np.abs(torques[mask].astype(np.float64))          # [steps, 12]
    rms, peak = np.sqrt((tau ** 2).mean(0)), tau.max(0)
    out = {}
    for k, kind in enumerate(JOINT_KINDS):
        idx = [3 * i + k for i in range(len(legs))]
        over = [bool(continuous is not None and rms[j] > continuous[j]) for j in idx]
        out[kind] = {"rms": float(rms[idx].max()), "peak": float(peak[idx].max()),
                     "per_leg": {leg: {"rms": float(rms[j]), "peak": float(peak[j]),
                                       **({"over_continuous": o} if continuous is not None else {})}
                                 for leg, j, o in zip(legs, idx, over)}}
        if continuous is not None:
            out[kind]["over_continuous"] = any(over)
    return out


def _true_runs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) indices of the runs of True in a 1-D bool array."""
    d = np.diff(np.concatenate([[0], x.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def gait_stats(rollout: dict, a: int, b: int, mask: np.ndarray, dt: float) -> dict | None:
    """Gait of one segment [a, b) over the steps ``mask`` [b - a, N] keeps (as for rmse).

    clearance_mm_median / _p90: peak height of the foot's lowest point (``foot_pos_w`` z)
      per swing, over swings that lie wholly inside the kept steps (None without
      ``foot_pos_w``, e.g. recordings made before it existed). Height above z = 0: flat
      ground only.
    thigh_excursion_rad / calf_excursion_rad: p95 - p5 of the joint angle per robot and
      leg, mean over both.   calf_mean_rad: mean calf angle.   trunk_height_m: mean base z.
    stride_hz: touchdowns per foot per second of kept steps.   duty_factor: fraction of foot-steps in
      contact.   power_W: mean of sum over joints |tau * qdot|.
    """
    if not mask.any():
        return None
    q = rollout["joint_pos"][a:b].astype(np.float64)
    T, N = mask.shape
    out = {}
    for name, k in (("thigh", 1), ("calf", 2)):
        exc = []
        for n in range(N):
            x = q[mask[:, n], n, k::3]                     # [steps, legs]
            if len(x):
                exc.extend(np.percentile(x, 95, axis=0) - np.percentile(x, 5, axis=0))
        out[f"{name}_excursion_rad"] = float(np.mean(exc))
    out["calf_mean_rad"] = float(q[..., 2::3][mask].mean())
    out["trunk_height_m"] = float(rollout["qpos"][a:b, :, 2][mask].mean())

    contact = rollout["foot_contact"][a:b].astype(bool)  # [T, N, 4]
    both = mask[1:] & mask[:-1]                          # transitions between kept steps
    touchdowns = (contact[1:] & ~contact[:-1] & both[..., None]).sum()
    out["stride_hz"] = float(touchdowns / (contact.shape[-1] * max(int(both.sum()), 1) * dt))
    out["duty_factor"] = float(contact[mask].mean())

    if "torques" in rollout:
        p = np.abs(rollout["torques"][a:b].astype(np.float64) * rollout["joint_vel"][a:b]).sum(-1)
        out["power_W"] = float(p[mask].mean())

    peaks = []
    if "foot_pos_w" in rollout:
        z = rollout["foot_pos_w"][a:b, ..., 2]           # [T, N, 4]
        for n in range(N):
            for f in range(contact.shape[-1]):
                for s, e in zip(*_true_runs(~contact[:, n, f])):
                    if s > 0 and e < T and mask[s - 1:e + 1, n].all():   # complete swing
                        peaks.append(z[s:e, n, f].max())
    out["swings"] = len(peaks)
    out["clearance_mm_median"] = float(np.median(peaks) * 1000) if peaks else None
    out["clearance_mm_p90"] = float(np.percentile(peaks, 90) * 1000) if peaks else None
    return out


def summarize(rollout: dict, meta: dict) -> dict:
    from quadruped_rl.evaluation import metrics

    dt = meta["dt"]
    fell = rollout["fell"].astype(bool)
    N = fell.shape[1]
    ok = valid_mask(rollout, meta)
    cmd, v, w = rollout["command"], rollout["base_lin_vel_b"], rollout["base_ang_vel_b"]
    legs = list(meta.get("foot_names") or ("FR", "FL", "RR", "RL"))
    continuous = meta.get("continuous_torque")

    segments = []
    for seg in meta.get("segments") or [{"label": "whole rollout", "start_s": 0.0,
                                         "end_s": fell.shape[0] * dt, "command": None}]:
        a, b = round(seg["start_s"] / dt), round(seg["end_s"] / dt)
        m = ok[a:b]
        row = {"label": seg["label"], "command": seg.get("command"),
               "falls": int(fell[a:b].sum()), "robots_fell": int(fell[a:b].any(0).sum())}
        if m.any():
            e_xy = np.linalg.norm(cmd[a:b, :, :2] - v[a:b, :, :2], axis=-1)[m]
            e_wz = np.abs(cmd[a:b, :, 2] - w[a:b, :, 2])[m]
            row.update(mean_vx=float(v[a:b, :, 0][m].mean()), mean_vy=float(v[a:b, :, 1][m].mean()),
                       mean_wz=float(w[a:b, :, 2][m].mean()),
                       rmse_xy=float(np.sqrt((e_xy ** 2).mean())),
                       rmse_wz=float(np.sqrt((e_wz ** 2).mean())))
        if "torques" in rollout:
            row["torque_Nm"] = torque_stats(rollout["torques"][a:b], m, legs, continuous)
        row["gait"] = gait_stats(rollout, a, b, m, dt)
        segments.append(row)

    clean = [n for n in range(N) if not fell[:, n].any()]
    per_robot = {}
    for n in clean:
        log = {"command": cmd[:, n], "base_lin_vel": v[:, n], "joint_vel": rollout["joint_vel"][:, n],
               "actions": rollout["actions"][:, n], "torques": rollout["torques"][:, n],
               "foot_vel": rollout["foot_vel_w"][:, n],
               "foot_contact": rollout["foot_contact"][:, n].astype(bool)}
        per_robot[n] = metrics.summarize(log, dt, meta["mass"])
    smooth = ({k: float(np.mean([r[k] for r in per_robot.values()])) for k in next(iter(per_robot.values()))}
              if per_robot else {})
    out = {"robots": N, "duration_s": fell.shape[0] * dt, "total_falls": int(fell.sum()),
           "robots_that_fell": int(fell.any(0).sum()),
           "falls_per_robot": fell.sum(0).astype(int).tolist(),
           "segments": segments, "metrics_mean_over_robots_without_falls": smooth,
           "robots_without_falls": len(clean), "continuous_torque_Nm": continuous}
    if "ref_qpos" in rollout:
        out["tracking"] = tracking_errors(rollout, meta)
    return out


def format_summary(s: dict, meta: dict) -> str:
    lines = [f"Run: {meta.get('run_dir') or 'no policy (zero actions)'}",
             f"Checkpoint: {meta.get('checkpoint') or '-'}",
             (f"Task: {meta.get('task_name')}   robot: {meta.get('robot_name')}   "
              f"scenario: {meta.get('scenario')}"),
             (f"{s['robots']} robots x {s['duration_s']:.1f} s;  falls: {s['total_falls']} "
              f"({s['robots_that_fell']} robots fell at least once)"), ""]
    lines.append(f"{'segment':<28}{'command':>22}{'actual (mean)':>24}{'rmse xy':>9}"
                 f"{'rmse wz':>9}{'falls':>7}")
    for r in s["segments"]:
        c = r.get("command")
        cs = f"{c[0]:+.2f} {c[1]:+.2f} {c[2]:+.2f}" if c else "-"
        if "mean_vx" in r:
            acts = f"{r['mean_vx']:+.2f} {r['mean_vy']:+.2f} {r['mean_wz']:+.2f}"
            lines.append(f"{r['label']:<28}{cs:>22}{acts:>24}{r['rmse_xy']:>9.3f}"
                         f"{r['rmse_wz']:>9.3f}{r['falls']:>7}")
        else:
            lines.append(f"{r['label']:<28}{cs:>22}{'(no valid steps)':>24}{'':>18}{r['falls']:>7}")
    lines += ["", ("Columns: command and actual are vx m/s, vy m/s, wz rad/s; rmse excludes the "
                   "first 1 s of each segment and after each fall.")]
    if any(r.get("torque_Nm") for r in s["segments"]):
        lines += ["", "Joint torque, N m: RMS and peak |torque|, largest over the four legs (same steps as rmse)",
                  f"{'segment':<28}" + "".join(f"{k + ' rms':>11}" for k in JOINT_KINDS)
                  + "".join(f"{k + ' peak':>12}" for k in JOINT_KINDS)]
        for r in s["segments"]:
            t = r.get("torque_Nm")
            if not t:
                lines.append(f"{r['label']:<28}{'(no valid steps)':>33}")
                continue
            rms = "".join(f"{t[k]['rms']:>10.2f}" + ("*" if t[k].get("over_continuous") else " ")
                          for k in JOINT_KINDS)
            lines.append(f"{r['label']:<28}{rms}" + "".join(f"{t[k]['peak']:>12.2f}" for k in JOINT_KINDS))
        cont = s.get("continuous_torque_Nm")
        if cont:
            rating = ", ".join(f"{kind} {max(cont[k::3]):g}" for k, kind in enumerate(JOINT_KINDS))
            lines.append(f"* RMS above the continuous torque rating ({rating} N m; robot config "
                         "actuator.continuous_torque) on at least one leg.")
    if any(r.get("gait") for r in s["segments"]):
        lines += ["", ("Gait (same steps as rmse): swing foot clearance median / p90 mm, thigh and "
                       "calf excursion p95-p5 rad, mean calf rad, trunk m, stride Hz, duty, power W"),
                  f"{'segment':<28}{'clear':>7}{'p90':>6}{'thigh':>7}{'calf':>7}{'calf q':>8}"
                  f"{'trunk':>7}{'stride':>8}{'duty':>6}{'power':>7}"]
        for r in s["segments"]:
            g = r.get("gait")
            if not g:
                lines.append(f"{r['label']:<28}{'(no valid steps)':>33}")
                continue
            cl = ("".join(f"{g[k]:>{w}.0f}" for k, w in (("clearance_mm_median", 7), ("clearance_mm_p90", 6)))
                  if g["clearance_mm_median"] is not None else f"{'-':>7}{'-':>6}")
            pw = f"{g['power_W']:>7.0f}" if "power_W" in g else f"{'-':>7}"
            lines.append(f"{r['label']:<28}{cl}{g['thigh_excursion_rad']:>7.2f}{g['calf_excursion_rad']:>7.2f}"
                         f"{g['calf_mean_rad']:>8.2f}{g['trunk_height_m']:>7.3f}{g['stride_hz']:>8.2f}"
                         f"{g['duty_factor']:>6.2f}{pw}")
    tr = s.get("tracking")
    if tr:
        lines += ["", "Tracking (command = the reference clip's base velocity):",
                  f"  joint angle RMSE   {tr['joint_rmse_rad']:.3f} rad" if tr["joint_rmse_rad"] is not None else "",
                  f"  trunk position err {tr['root_pos_err_m']:.3f} m" if tr["root_pos_err_m"] is not None else "",
                  f"  clips without a fall: {tr['clips_without_falls']}/{len(tr['clips'])}", "",
                  f"  {'robot':<6}{'clip':<44}{'falls':>6}{'first fall':>12}{'joint rmse':>12}{'trunk err':>11}"]
        for i, c in enumerate(tr["clips"]):
            ff = f"{c['first_fall_s']:.1f} s" if c["first_fall_s"] is not None else "-"
            jr = f"{c['joint_rmse_rad']:.3f}" if c["joint_rmse_rad"] is not None else "-"
            re_ = f"{c['root_pos_err_m']:.3f}" if c["root_pos_err_m"] is not None else "-"
            lines.append(f"  {i:<6}{c['clip'][:42]:<44}{c['falls']:>6}{ff:>12}{jr:>12}{re_:>11}")
        lines = [x for i, x in enumerate(lines) if x != "" or (i > 0 and lines[i - 1] != "")]
    m = s["metrics_mean_over_robots_without_falls"]
    if m:
        lines += ["", (f"Smoothness and efficiency (mean over {s['robots_without_falls']} robots "
                       "without falls; see evaluation/metrics.py):")]
        lines += [f"  {k:<26}{val:.4g}" for k, val in m.items()]
    return "\n".join(lines) + "\n"
