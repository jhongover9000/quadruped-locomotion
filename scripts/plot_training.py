"""Plot training curves of one run, or compare several: any algorithm, one set of axes.

    python scripts/plot_training.py outputs/ppo/flat_velocity/<run>/            # dashboard
    python scripts/plot_training.py <ppo run>/ <ppo_rslrl run>/                 # comparison
    python scripts/plot_training.py outputs/*/flat_velocity/*/ --x hours --out flat.png
    python scripts/plot_training.py <run>/ --tags Episode_Reward/track_lin_vel_xy wm/nse_1step
    python scripts/plot_training.py <run>/ --list-tags

One run: a dashboard of everything its trainer logged (returns, losses, reward terms,
terminations, curriculum; for SuperMimic also world-model error and imagined rollouts),
saved as <run>/training_curves.png with <run>/progress_all.csv next to it.

Several runs: the quantities they share (return, episode length, losses, learning rate,
action noise, throughput), one line per run, saved as outputs/comparisons/<time>.png.

Each trainer counts iterations differently (PPO: 24 steps in every environment;
SuperMimic: one), so the default x-axis is simulated environment steps. --x hours shows
wall-clock time instead. Returns are comparable only between runs of the same task.
Needs no GPU: it only reads the TensorBoard files in the run directories.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from quadruped_rl.evaluation import curves as C
from quadruped_rl.utils.paths import OUTPUT_DIR


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="run directories")
    p.add_argument("--labels", nargs="+", help="legend label per run (default: algo / task / run)")
    p.add_argument("--x", choices=["env_steps", "hours", "iteration"], default="env_steps")
    p.add_argument("--smooth", type=int, default=0, help="moving-average window in log points (0: auto)")
    p.add_argument("--tags", nargs="+", help="comparison: plot these quantities or raw tags instead")
    p.add_argument("--out", type=Path, help="output image")
    p.add_argument("--list-tags", action="store_true", help="print what each run logged and exit")
    args = p.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        p.error("--labels needs one label per run")
    runs = []
    for i, run_dir in enumerate(args.runs):
        try:
            runs.append(C.load_curves(run_dir, args.labels[i] if args.labels else None))
        except FileNotFoundError as e:
            print(f"skipping {run_dir}: {e}")
    if not runs:
        raise SystemExit("No runs with TensorBoard logs.")

    if args.list_tags:
        for r in runs:
            print(f"\n{r.label}  ({r.env_steps_per_iteration:.0f} env steps per iteration)")
            for tag in sorted(r.series):
                s = r.series[tag]
                print(f"  {tag:<44} {len(s.step):5d} points, last {s.value[-1]:.4g}")
        return

    for r in runs:
        s = C.summarize(r)
        print(f"{r.label}: {s['iterations']} iterations, {s['env_steps']:,} env steps, "
              f"{s['hours']:.2f} h" + "".join(f", {k} {s[k]:.4g}" for k in ("return", "episode_length") if k in s))

    if len(runs) == 1 and not args.tags:
        r = runs[0]
        out = args.out or r.run_dir / "training_curves.png"
        C.plot_dashboard(r, out, axis=args.x, window=args.smooth)
        C.export_csv(r, out.with_name("progress_all.csv"))
        (out.with_name("training_summary.json")).write_text(json.dumps(C.summarize(r), indent=2))
        print(f"Saved {out}\n      {out.with_name('progress_all.csv')}")
    else:
        if args.out:
            out = args.out
        else:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = OUTPUT_DIR / "comparisons" / f"{stamp}_training.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        C.plot_comparison(runs, out, axis=args.x, window=args.smooth, quantities=args.tags)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
