"""Learning methods and baselines. All share the Agent interface in base.py.

Imports are lazy so lightweight commands (``train.py --dry-run``, tests) don't need
torch or a GPU backend installed.
"""

from __future__ import annotations

import importlib

_REGISTRY = {
    "ppo": "quadruped_rl.algorithms.ppo.agent:PPOAgent",
    "ppo_rslrl": "quadruped_rl.algorithms.ppo.reference:RslRlPPOAgent",
    "deepmimic": "quadruped_rl.algorithms.deepmimic.agent:DeepMimicAgent",
    "amp": "quadruped_rl.algorithms.amp.agent:AMPAgent",
    "supermimic": "quadruped_rl.algorithms.supermimic.agent:SuperMimicAgent",
    "mpc": "quadruped_rl.algorithms.mpc_baseline.controller:ConvexMPC",
}


class IncompatibleError(ValueError):
    """The requested algorithm cannot train on the requested task."""


def get_algorithm(name: str):
    """Return the Agent class registered under ``name``."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown algorithm '{name}'. Available: {sorted(_REGISTRY)}")
    module, cls = _REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), cls)


def algorithm_names() -> list[str]:
    return list(_REGISTRY)


def check_compatible(algo_name: str, task_cfg: dict) -> None:
    """Raise before any work is done if ``algo_name`` cannot train on this task."""
    from quadruped_rl.algorithms.base import task_kind

    cls = get_algorithm(algo_name)
    if not cls.implemented:
        raise IncompatibleError(
            f"'{algo_name}' is not implemented yet ({getattr(cls, 'phase', 'planned')}). "
            f"Implemented: {[n for n in _REGISTRY if get_algorithm(n).implemented]}")
    kind = task_kind(task_cfg)
    if kind not in cls.task_kinds:
        fits = [n for n in _REGISTRY
                if get_algorithm(n).implemented and kind in get_algorithm(n).task_kinds]
        raise IncompatibleError(
            f"'{algo_name}' trains on {' / '.join(cls.task_kinds)} tasks, but "
            f"'{task_cfg.get('name', '?')}' is a {kind} task. Algorithms for it: {fits or 'none yet'}")


def compatibility_table(task_names: list[str] | None = None) -> str:
    """Which algorithm runs on which task, as text (scripts/train.py --list)."""
    from quadruped_rl.algorithms.base import task_kind
    from quadruped_rl.utils.config import available, load_group

    tasks = {t: load_group("tasks", t) for t in (task_names or available("tasks"))}
    width = max(len(n) for n in _REGISTRY) + 2
    lines = ["algorithm".ljust(width) + "  ".join(t.ljust(16) for t in tasks) + "  status"]
    for name in _REGISTRY:
        cls = get_algorithm(name)
        cells = ["yes" if cls.implemented and task_kind(cfg) in cls.task_kinds else "-"
                 for cfg in tasks.values()]
        status = "implemented" if cls.implemented else f"planned ({getattr(cls, 'phase', '')})"
        lines.append(name.ljust(width) + "  ".join(c.ljust(16) for c in cells) + "  " + status)
    return "\n".join(lines)
