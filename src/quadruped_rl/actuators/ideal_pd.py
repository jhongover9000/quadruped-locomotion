"""Ideal PD actuator: the default before any system identification.

    tau = clip( Kp (q* - q) - Kd qd,  -tau_max, +tau_max )

Instant, frictionless, and exact. Policies trained only on this tend to exploit it,
which is why it gets replaced by the analytic model or an actuator network later.
"""

from __future__ import annotations

import numpy as np

from quadruped_rl.actuators.base import Actuator


class IdealPD(Actuator):
    def compute(self, q_target: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        tau = self.kp * (q_target - q) - self.kd * qd
        return np.clip(tau, -self.torque_limit, self.torque_limit)
