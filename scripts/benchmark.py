"""Measure training throughput on this machine and project run and sweep times.

    python scripts/benchmark.py                                  # flat task, 1024/2048/4096 envs
    python scripts/benchmark.py --task rough_velocity --num-envs 2048 4096
    python scripts/benchmark.py --iterations 10                  # longer, steadier measurement

For each environment count, runs a few real PPO iterations (simulation + learning) in a
fresh process, then reports steps/s, GPU memory, and how long a full run and each sweep
in configs/sweeps/ would take at that rate. Results are saved to outputs/benchmarks/.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from quadruped_rl.utils.config import available, load_group
from quadruped_rl.utils.paths import CONFIG_DIR, OUTPUT_DIR


def measure(task: str, robot: str, num_envs: int, iterations: int, device: str) -> dict:
    """One measurement (runs inside a subprocess so GPU memory is released after)."""
    import torch

    from quadruped_rl.algorithms.ppo.agent import PPOAgent
    from quadruped_rl.envs.builder import make_env
    from quadruped_rl.robots import load_robot

    algo = load_group("algorithms", "ppo")
    cuda = device.startswith("cuda")
    free_before = torch.cuda.mem_get_info()[0] if cuda else 0

    t0 = time.perf_counter()
    env = make_env(load_robot(robot), load_group("tasks", task), num_envs, device, seed=0)
    build_s = time.perf_counter() - t0

    with tempfile.TemporaryDirectory() as tmp:
        agent = PPOAgent(robot=None, task_cfg={}, algo_cfg=algo, run_dir=Path(tmp), device=device)
        agent.learn(env, max_iterations=1)          # warm-up: kernel compilation, allocation
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        agent.learn(env, max_iterations=iterations)
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    steps = iterations * int(algo["steps_per_env"]) * num_envs
    result = {"num_envs": num_envs, "steps_per_s": steps / elapsed,
              "s_per_iteration": elapsed / iterations, "build_s": build_s}
    if cuda:
        free_after, total = torch.cuda.mem_get_info()
        result["gpu_mem_used_gb"] = (free_before - free_after) / 1e9
        result["gpu_mem_total_gb"] = total / 1e9
        result["gpu"] = torch.cuda.get_device_name(0)
    return result


def sweep_runs() -> dict[str, int]:
    runs = {}
    for path in sorted((CONFIG_DIR / "sweeps").glob("*.yaml")):
        cfg = load_group("sweeps", path.stem)
        n = len(cfg.get("algorithms", [None])) * len(cfg.get("seeds", [0]))
        for values in (cfg.get("grid") or {}).values():
            n *= len(values)
        runs[path.stem] = n
    return runs


def hours(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds < 3600 else f"{seconds / 3600:.1f} h"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="flat_velocity", choices=available("tasks"))
    p.add_argument("--robot", default="go1", choices=available("robots"))
    p.add_argument("--num-envs", type=int, nargs="+", default=[1024, 2048, 4096])
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--single", type=int, help=argparse.SUPPRESS)  # internal: one measurement
    args = p.parse_args()

    if args.device is None:
        import torch

        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.single is not None:
        print("RESULT " + json.dumps(measure(args.task, args.robot, args.single, args.iterations,
                                              args.device)))
        return

    results = []
    for n in args.num_envs:
        print(f"Measuring {n} envs ...", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, "--task", args.task, "--robot", args.robot,
             "--iterations", str(args.iterations), "--device", args.device, "--single", str(n)],
            capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            print(f"  failed (likely out of memory):\n{proc.stderr[-2000:]}")
            continue
        results.append(json.loads(line[len("RESULT "):]))

    if not results:
        sys.exit("No successful measurements.")
    algo = load_group("algorithms", "ppo")
    max_it = int(algo["max_iterations"])
    print(f"\nTask {args.task}, robot {args.robot}, device {results[0].get('gpu', args.device)}")
    print(f"{'envs':>6} {'steps/s':>10} {'s/iter':>7} {'GPU mem':>9} {'full run (' + str(max_it) + ' it)':>20}")
    for r in results:
        mem = f"{r['gpu_mem_used_gb']:.1f} GB" if "gpu_mem_used_gb" in r else "-"
        print(f"{r['num_envs']:>6} {r['steps_per_s']:>10,.0f} {r['s_per_iteration']:>7.2f} "
              f"{mem:>9} {hours(r['s_per_iteration'] * max_it):>20}")

    best = max(results, key=lambda r: r["steps_per_s"])
    run_s = best["s_per_iteration"] * max_it
    print(f"\nAt {best['num_envs']} envs, one {max_it}-iteration run takes ~{hours(run_s)}.")
    for name, n in sweep_runs().items():
        print(f"  sweep '{name}': {n} runs -> ~{hours(n * run_s)} back to back")
    print("(Projections hold the iteration count fixed, so they are realistic only at the\n"
          f" configured num_envs ({algo['num_envs']}). Sweeps count every algorithm in the grid.)")

    out = OUTPUT_DIR / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"{stamp}_{args.task}.json"
    path.write_text(json.dumps({"task": args.task, "robot": args.robot, "results": results}, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
