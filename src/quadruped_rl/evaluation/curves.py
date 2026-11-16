"""Training curves for any run, whichever trainer wrote it.

Our PPO, rsl_rl's PPO and the SuperMimic backend each log to TensorBoard under different
tag names and count "iterations" differently. This module reads a run's event files,
maps the tags onto common quantities, and puts every run on the same x-axis (simulated
environment steps, or wall-clock hours), so runs can be plotted side by side.

    curves = load_curves(run_dir)
    plot_dashboard(curves, run_dir / "training_curves.png")       # one run, everything logged
    plot_comparison([curves_a, curves_b], "comparison.png")       # several runs, common panels
    export_csv(curves, run_dir / "progress_all.csv")

Used by scripts/plot_training.py and, at the end of every training run, scripts/train.py.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Common quantity -> the tag each trainer uses for it (ours, rsl_rl, SuperMimic).
ALIASES: dict[str, tuple[str, ...]] = {
    "return": ("Train/mean_return", "Train/mean_reward", "train/episode_norm_return"),
    "episode_length": ("Train/mean_episode_length", "train/episode_len"),
    "policy_loss": ("Loss/policy_loss", "Loss/surrogate", "policy/imag_objective"),
    "value_loss": ("Loss/value_loss", "Loss/value", "policy/critic_loss"),
    "entropy": ("Loss/entropy",),
    "learning_rate": ("Policy/learning_rate", "Loss/learning_rate", "lr"),
    "action_std": ("Policy/action_std", "Policy/mean_std"),
    "steps_per_s": ("Perf/steps_per_s", "Perf/total_fps"),
}
LABELS = {
    "return": "Episode return", "episode_length": "Episode length (control steps)",
    "policy_loss": "Policy loss", "value_loss": "Value loss", "entropy": "Policy entropy",
    "learning_rate": "Learning rate", "action_std": "Action noise std",
    "steps_per_s": "Throughput (env steps/s)",
}
LOG_SCALE = {"learning_rate", "value_loss"}

# Validated categorical palette (fixed order, never cycled) and neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, GRID, OTHER = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b5b4ae"
MAX_SERIES = len(SERIES)


@dataclass
class Series:
    step: np.ndarray        # trainer's own iteration counter
    value: np.ndarray
    wall: np.ndarray        # seconds since the run's first event


@dataclass
class RunCurves:
    run_dir: Path
    label: str
    algo: str
    task: str
    env_steps_per_iteration: float      # simulated control steps per trainer iteration
    series: dict[str, Series] = field(default_factory=dict)   # by original tag

    def tag_for(self, quantity: str) -> str | None:
        return next((t for t in ALIASES.get(quantity, (quantity,)) if t in self.series), None)

    def get(self, quantity: str) -> Series | None:
        tag = self.tag_for(quantity)
        if tag is not None:
            return self.series[tag]
        if quantity == "steps_per_s":
            return self._supermimic_throughput()
        return None

    def _supermimic_throughput(self) -> Series | None:
        parts = [self.series.get(f"time/{k}_s_per_iter") for k in ("sim", "wm", "policy")]
        if any(p is None for p in parts):
            return None
        n = min(len(p.step) for p in parts)
        total = sum(p.value[:n] for p in parts)
        return Series(parts[0].step[:n], self.env_steps_per_iteration / np.maximum(total, 1e-9),
                      parts[0].wall[:n])

    def x(self, s: Series, axis: str) -> np.ndarray:
        if axis == "iteration":
            return s.step
        if axis == "hours":
            return s.wall / 3600.0
        return s.step * self.env_steps_per_iteration


# ---- loading ------------------------------------------------------------------------

def _run_info(run_dir: Path) -> tuple[str, str, float]:
    """(algorithm, task, env steps per iteration) from the run's saved config."""
    cfg_yaml, cfg_json = run_dir / "config.yaml", run_dir / "config.json"
    if cfg_yaml.exists():
        import yaml

        with open(cfg_yaml, encoding="utf-8") as f:
            saved = yaml.safe_load(f)
        algo, task = saved["algo"], saved.get("task") or {}
        name = algo.get("name", "?")
        per_iter = float(algo.get("num_envs", 1))
        if "steps_per_env" in algo and name != "supermimic":
            per_iter *= float(algo["steps_per_env"])
        return name, task.get("name", "?"), per_iter
    if cfg_json.exists():       # a run made directly with the SuperMimic reference trainer
        saved = json.loads(cfg_json.read_text())
        return "supermimic", "motion_tracking", float(saved["env"]["num_envs"])
    return "?", "?", 1.0


def load_curves(run_dir: str | Path, label: str | None = None) -> RunCurves:
    """Read every scalar a run logged (all event files under it, resumed runs included)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir = Path(run_dir)
    files = sorted(run_dir.rglob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files under {run_dir}")
    algo, task, per_iter = _run_info(run_dir)
    raw: dict[str, list[tuple[int, float, float]]] = {}
    for f in files:
        acc = EventAccumulator(str(f), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            raw.setdefault(tag, []).extend((e.step, e.wall_time, e.value) for e in acc.Scalars(tag))
    t0 = min(w for events in raw.values() for _, w, _ in events)
    curves = RunCurves(run_dir, label or _default_label(run_dir, algo, task), algo, task, per_iter)
    for tag, events in raw.items():
        events.sort(key=lambda e: (e[0], e[1]))
        latest = {step: (wall, value) for step, wall, value in events}   # resumed runs: last wins
        steps = np.array(sorted(latest), dtype=np.float64)
        curves.series[tag] = Series(
            steps, np.array([latest[s][1] for s in steps]),
            np.array([latest[s][0] - t0 for s in steps]))
    return curves


def _default_label(run_dir: Path, algo: str, task: str) -> str:
    parts = run_dir.resolve().parts
    return f"{algo} / {task} / {parts[-1]}" if algo != "?" else "/".join(parts[-2:])


def export_csv(curves: RunCurves, path: Path) -> None:
    """One row per logged iteration, one column per tag (blank where a tag was not logged)."""
    tags = sorted(curves.series)
    steps = sorted({s for t in tags for s in curves.series[t].step})
    lookup = {t: dict(zip(curves.series[t].step, curves.series[t].value)) for t in tags}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "env_steps", *tags])
        for s in steps:
            w.writerow([int(s), int(s * curves.env_steps_per_iteration),
                        *(lookup[t].get(s, "") for t in tags)])


# ---- plotting -----------------------------------------------------------------------

def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average that shrinks at the ends (no lag, no edge artifacts)."""
    if window <= 1 or len(values) < 3:
        return values
    half = window // 2
    csum = np.concatenate([[0.0], np.cumsum(values)])
    idx = np.arange(len(values))
    lo, hi = np.maximum(idx - half, 0), np.minimum(idx + half + 1, len(values))
    return (csum[hi] - csum[lo]) / (hi - lo)


def _style(ax, title: str, xlabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=6)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK_2)


def _axis_label(axis: str) -> str:
    return {"env_steps": "environment steps", "hours": "wall-clock hours",
            "iteration": "trainer iteration"}[axis]


def _line(ax, x, y, color, label, window, lw=2.0):
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return False
    if window > 1 and len(y) > 2 * window:
        ax.plot(x, y, color=color, linewidth=0.8, alpha=0.25)
        y = smooth(y, window)
    ax.plot(x, y, color=color, linewidth=lw, label=label,
            marker="o" if len(x) < 3 else None, markersize=4)
    return True


def _legend(ax, n: int) -> None:
    """Up to four series: inside, where matplotlib finds room. More: below the axis, so the
    legend never covers data."""
    if n <= 1:
        return
    if n <= 4:
        ax.legend(fontsize=7, frameon=False, labelcolor=INK_2, loc="best")
    else:
        ax.legend(fontsize=7, frameon=False, labelcolor=INK_2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.2), ncol=3, columnspacing=1.0, handlelength=1.4)


def _safe_log(ax, values) -> None:
    v = np.concatenate([np.asarray(a)[np.isfinite(a)] for a in values]) if values else np.array([])
    if len(v) and v.min() > 0 and v.max() / v.min() > 20:
        ax.set_yscale("log")


def _group_panel(ax, curves: RunCurves, prefix: str, title: str, axis: str, window: int,
                 strip: str | None = None) -> bool:
    """All tags under ``prefix`` in one panel; past the palette's size the smallest fold
    into grey "other" lines rather than getting invented colours."""
    tags = [t for t in curves.series if t.startswith(prefix) and not t.endswith("/time")]
    if not tags:
        return False
    tags.sort(key=lambda t: -float(np.nanmax(np.abs(curves.series[t].value))))
    shown, rest = tags[:MAX_SERIES - 1 if len(tags) > MAX_SERIES else MAX_SERIES], []
    rest = tags[len(shown):]
    for t in rest:
        s = curves.series[t]
        _line(ax, curves.x(s, axis), s.value, OTHER, None, window, lw=1.0)
    for color, t in zip(SERIES, sorted(shown)):
        s = curves.series[t]
        _line(ax, curves.x(s, axis), s.value, color, t[len(strip or prefix):], window)
    if rest:
        ax.plot([], [], color=OTHER, linewidth=1.0, label=f"{len(rest)} smaller terms")
    _style(ax, title, _axis_label(axis))
    _legend(ax, len(shown) + bool(rest))
    return True


# Dashboard panels beyond the common ones: (title, tag prefix, prefix stripped from labels).
GROUPS = [
    ("Reward per term (episode mean)", "Episode_Reward/", None),
    ("Episode endings per cause", "Episode_Termination/", None),
    ("Task metrics", "Metrics/", None),
    ("Terrain curriculum level", "Curriculum/terrain_levels/", None),
    ("Imagined reward per term", "policy/imag_r/", None),
]
# SuperMimic panels made of named tags: title -> {tag: legend label}.
NAMED = [
    ("World-model error (normalized, lower is better)",
     {"wm/online_nse_1step": "1-step, fresh data", "wm/online_nse_8step": "8-step, fresh data",
      "wm/nse_1step": "1-step, training batch", "wm/nse_kstep": "k-step, training batch"}, True),
    ("Imagined rollouts", {"policy/imag_return_undisc": "return over the horizon",
                           "policy/critic_value_mean": "critic value"}, False),
    ("Imagined survival to the horizon", {"policy/imag_survival_H": "survival",
                                          "wm/term_acc": "termination-head accuracy"}, False),
    ("Seconds per iteration", {"time/sim_s_per_iter": "simulation", "time/wm_s_per_iter": "world model",
                               "time/policy_s_per_iter": "policy", "Perf/collection_s": "collection",
                               "Perf/learning_s": "learning", "Perf/collection_time": "collection",
                               "Perf/learning_time": "learning"}, False),
    ("Clip sampler: sampling entropy", {"sampler/entropy": "entropy"}, False),
    ("Clip sampler: mean return per clip bin", {"sampler/bin_return_mean": "bin return"}, False),
]
# What "return" means depends on the trainer's tag.
RETURN_LABELS = {"train/episode_norm_return": "Episode return (normalized: 1 = perfect tracking)",
                 "Train/mean_return": "Episode return (sum of rewards)",
                 "Train/mean_reward": "Episode return (sum of rewards)"}


def plot_dashboard(curves: RunCurves, path: Path, axis: str = "env_steps", window: int = 0) -> Path:
    """Everything one run logged: the common panels, then whatever its trainer adds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []      # callables drawing into an axis
    for q in ALIASES:
        s = curves.get(q)
        if s is not None and np.isfinite(s.value).any():
            panels.append(("common", q, s))
    for title, prefix, strip in GROUPS:
        if any(t.startswith(prefix) for t in curves.series):
            panels.append(("group", title, (prefix, strip)))
    for title, tags, log in NAMED:
        present = {t: lab for t, lab in tags.items() if t in curves.series}
        if present:
            panels.append(("named", title, (present, log)))

    cols = 3
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.5 * rows), squeeze=False,
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.12, h_pad=0.12)
    fig.patch.set_facecolor(SURFACE)
    if window <= 0:
        n = max((len(s.step) for s in curves.series.values()), default=0)
        window = max(1, n // 40)
    for ax, (kind, title, payload) in zip(axes.flat, panels):
        if kind == "common":
            _line(ax, curves.x(payload, axis), payload.value, SERIES[0], None, window)
            label = RETURN_LABELS.get(curves.tag_for(title), LABELS[title]) if title == "return" else LABELS[title]
            _style(ax, label, _axis_label(axis))
            if title in LOG_SCALE:
                _safe_log(ax, [payload.value])
        elif kind == "group":
            _group_panel(ax, curves, payload[0], title, axis, window, payload[1])
        else:
            tags, log = payload
            for color, (t, lab) in zip(SERIES, tags.items()):
                s = curves.series[t]
                _line(ax, curves.x(s, axis), s.value, color, lab, window)
            _style(ax, title, _axis_label(axis))
            if log:
                _safe_log(ax, [curves.series[t].value for t in tags])
            _legend(ax, len(tags))
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f"Training curves: {curves.label}", fontsize=12, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return Path(path)


def plot_comparison(runs: list[RunCurves], path: Path, axis: str = "env_steps",
                    window: int = 0, quantities: list[str] | None = None) -> Path:
    """Several runs on shared panels: one line per run, the same colour in every panel.

    ``quantities`` may name common quantities (see ALIASES) or raw TensorBoard tags.
    Returns differ in meaning across tasks (velocity: reward sum; tracking: normalized
    0-1), so compare returns only between runs of the same task.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(runs) > MAX_SERIES:
        raise ValueError(f"At most {MAX_SERIES} runs per comparison (got {len(runs)}); "
                         "split them into several plots")
    quantities = quantities or [q for q in ALIASES if any(r.get(q) is not None for r in runs)]
    cols = min(3, len(quantities))
    rows = -(-len(quantities) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.5 * rows + 0.3 * len(runs)),
                             squeeze=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.12, h_pad=0.12)
    fig.patch.set_facecolor(SURFACE)
    for ax, q in zip(axes.flat, quantities):
        drawn = []
        for color, run in zip(SERIES, runs):
            s = run.get(q)
            if s is None:
                continue
            w = window if window > 0 else max(1, len(s.step) // 40)
            if _line(ax, run.x(s, axis), s.value, color, run.label, w):
                drawn.append(s.value)
        _style(ax, LABELS.get(q, q), _axis_label(axis))
        if q in LOG_SCALE:
            _safe_log(ax, drawn)
        if not drawn:
            ax.text(0.5, 0.5, "not logged by these runs", transform=ax.transAxes, ha="center",
                    va="center", fontsize=8, color=INK_2)
    for ax in axes.flat[len(quantities):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=c, linewidth=2, label=r.label) for c, r in zip(SERIES, runs)]
    fig.legend(handles=handles, loc="outside upper left", fontsize=8, frameon=False,
               labelcolor=INK_2, ncol=1)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return Path(path)


def summarize(curves: RunCurves) -> dict:
    """Last-10% means of the common quantities: a one-line state of the run."""
    out = {"algo": curves.algo, "task": curves.task}
    for q in ALIASES:
        s = curves.get(q)
        if s is None or not len(s.value):
            continue
        tail = s.value[-max(1, len(s.value) // 10):]
        out[q] = float(np.nanmean(tail))
    longest = max(curves.series.values(), key=lambda s: len(s.step))
    out["iterations"] = int(longest.step[-1])
    out["env_steps"] = int(longest.step[-1] * curves.env_steps_per_iteration)
    out["hours"] = float(max(s.wall[-1] for s in curves.series.values()) / 3600.0)
    return out
