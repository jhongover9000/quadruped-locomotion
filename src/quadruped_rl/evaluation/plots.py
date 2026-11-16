"""Plots.

Rollout diagnostics (``plot_rollout``): one robot's commanded vs actual velocity, joint
angles, torques against their limits, actions, and a foot-contact gait diagram, with
schedule segments and falls marked. Written by scripts/record.py and scripts/replay.py.

Comparison plots (learning curves, Pareto fronts, per-terrain success, torque spectra):
TODO(phase 5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LEG_COLORS = {"hip": "tab:blue", "thigh": "tab:orange", "calf": "tab:green"}


def _joint_color(name: str) -> str:
    for key, color in LEG_COLORS.items():
        if key in name:
            return color
    return "tab:gray"


def plot_rollout(rollout: dict, meta: dict, robot: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt = meta["dt"]
    T = rollout["qpos"].shape[0]
    t = np.arange(T) * dt
    names = meta["joint_names"]
    cmd, v, w = rollout["command"][:, robot], rollout["base_lin_vel_b"][:, robot], rollout["base_ang_vel_b"][:, robot]
    falls = np.flatnonzero(rollout["fell"][:, robot]) * dt

    fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.4, 1.4, 1.4, 0.8]})
    for ax, (label, c, a) in zip(axes[:3], [("vx [m/s]", cmd[:, 0], v[:, 0]),
                                             ("vy [m/s]", cmd[:, 1], v[:, 1]),
                                             ("wz [rad/s]", cmd[:, 2], w[:, 2])]):
        ax.plot(t, c, color="tab:blue", lw=1.5, label="command")
        ax.plot(t, a, color="tab:cyan", lw=1.0, label="actual")
        ax.set_ylabel(label)
    axes[0].legend(loc="upper right", fontsize=8)

    for j, name in enumerate(names):
        col = _joint_color(name)
        axes[3].plot(t, rollout["joint_pos"][:, robot, j], color=col, lw=0.7)
        axes[4].plot(t, rollout["torques"][:, robot, j], color=col, lw=0.7)
        axes[5].plot(t, rollout["actions"][:, robot, j], color=col, lw=0.7)
    axes[3].set_ylabel("joint angle [rad]")
    axes[4].set_ylabel("torque [N m]")
    axes[5].set_ylabel("action")
    for lim in {abs(x) for x in meta.get("torque_limits") or [] if x}:
        axes[4].axhline(lim, color="red", ls="--", lw=0.6)
        axes[4].axhline(-lim, color="red", ls="--", lw=0.6)
    handles = [plt.Line2D([], [], color=c, label=k) for k, c in LEG_COLORS.items()]
    axes[3].legend(handles=handles, loc="upper right", fontsize=8, ncol=3)
    if meta.get("torque_limits"):
        axes[4].text(0.005, 0.92, "red dashed: actuator torque limits", transform=axes[4].transAxes,
                     fontsize=8, color="red")

    contact = rollout["foot_contact"][:, robot].astype(bool)
    for k, foot in enumerate(meta["foot_names"]):
        on = contact[:, k]
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            axes[6].broken_barh([(s * dt, (e - s) * dt)], (k - 0.4, 0.8), color="black")
    axes[6].set_yticks(range(len(meta["foot_names"])), meta["foot_names"])
    axes[6].set_ylabel("contact")
    axes[6].set_xlabel("time [s]")

    for seg in meta.get("segments") or []:
        for ax in axes:
            ax.axvline(seg["start_s"], color="gray", lw=0.6, ls=":")
        axes[0].text(seg["start_s"] + 0.05, 1.02, seg["label"], transform=axes[0].get_xaxis_transform(),
                     fontsize=7, rotation=20, va="bottom")
    for f in falls:
        for ax in axes:
            ax.axvline(f, color="red", lw=1.0)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlim(0, T * dt)

    title = f"{meta.get('run_name', 'no policy')}  {meta.get('checkpoint_name', '')}  robot {robot}"
    if len(falls):
        title += f"   ({len(falls)} falls, red lines)"
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
