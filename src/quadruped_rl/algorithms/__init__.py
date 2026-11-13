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


def get_algorithm(name: str):
    """Return the Agent class registered under ``name``."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown algorithm '{name}'. Available: {sorted(_REGISTRY)}")
    module, cls = _REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), cls)
