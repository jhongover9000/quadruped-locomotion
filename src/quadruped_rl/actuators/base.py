"""Actuator interface, used outside the GPU simulator (CPU sim2sim, analysis, tests).

    tau = actuator.compute(q_target, q, qd)

Inside mjlab training, the ideal PD loop runs as MuJoCo's built-in position actuator
(configured in envs/builder.py from the same robot YAML). Stateful models (delay,
actuator networks) keep their own history; call ``reset(env_ids)`` when envs reset.
"""

from __future__ import annotations

import numpy as np


class Actuator:
    def __init__(self, spec):
        self.spec = spec
        self.kp = np.asarray(spec.kp, dtype=float)
        self.kd = np.asarray(spec.kd, dtype=float)
        self.torque_limit = np.asarray(spec.torque_limit, dtype=float)

    def reset(self, env_ids=None) -> None:
        """Clear internal history for the given environments (no-op if stateless)."""

    def compute(self, q_target: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        raise NotImplementedError
