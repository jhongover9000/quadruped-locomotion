"""Actuator interface. In simulation this replaces the ideal torque source.

    tau = actuator.compute(q_target, q, qd)

Stateful models (delay, actuator networks) keep their own history; call ``reset(env_ids)``
when environments reset.
"""

from __future__ import annotations

import numpy as np


class Actuator:
    def __init__(self, spec):
        self.spec = spec
        self.kp = spec.kp
        self.kd = spec.kd
        self.torque_limit = np.asarray(spec.torque_limit, dtype=float)

    def reset(self, env_ids=None) -> None:
        """Clear internal history for the given environments (no-op if stateless)."""

    def compute(self, q_target: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        raise NotImplementedError
