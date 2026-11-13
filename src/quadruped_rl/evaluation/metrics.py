"""Evaluation metrics, computed identically for every method.

Every function takes arrays logged from one rollout with time along axis 0, sampled
every ``dt`` seconds (the control period). On hardware, low-pass filter velocities
before differentiating; finite differences amplify sensor noise.
"""

from __future__ import annotations

import numpy as np

G = 9.81


def _diff(x: np.ndarray, dt: float, order: int) -> np.ndarray:
    for _ in range(order):
        x = np.diff(x, axis=0) / dt
    return x


def joint_jerk(joint_vel: np.ndarray, dt: float) -> float:
    """Mean squared joint jerk, (1/T) * integral of ||d^3 q / dt^3||^2 dt.

    Computed as the second derivative of joint velocity [T, n_joints].
    """
    jerk = _diff(np.asarray(joint_vel, dtype=float), dt, 2)
    return float(np.mean(np.sum(jerk**2, axis=-1)))


def action_smoothness(actions: np.ndarray) -> float:
    """Mean squared second difference of actions, ||a_t - 2 a_{t-1} + a_{t-2}||^2."""
    a = np.asarray(actions, dtype=float)
    d2 = a[2:] - 2.0 * a[1:-1] + a[:-2]
    return float(np.mean(np.sum(d2**2, axis=-1)))


def high_freq_fraction(signal: np.ndarray, dt: float, cutoff_hz: float = 10.0) -> float:
    """Fraction of a signal's power above ``cutoff_hz`` (shaky policies score high).

    Applied to joint torques [T, n_joints]; the spectrum is summed over joints.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    x = x - x.mean(axis=0)
    power = np.abs(np.fft.rfft(x, axis=0)) ** 2
    freqs = np.fft.rfftfreq(x.shape[0], dt)
    total = power.sum()
    return float(power[freqs >= cutoff_hz].sum() / total) if total > 0 else 0.0


def cost_of_transport(torques: np.ndarray, joint_vel: np.ndarray, base_vel_xy: np.ndarray,
                      mass: float) -> float:
    """Mechanical cost of transport, mean(sum_j |tau_j qd_j|) / (m g mean(|v_xy|))."""
    power = np.sum(np.abs(np.asarray(torques) * np.asarray(joint_vel)), axis=-1)
    speed = np.linalg.norm(np.asarray(base_vel_xy), axis=-1)
    mean_speed = float(np.mean(speed))
    if mean_speed < 1e-3:
        return float("inf")
    return float(np.mean(power) / (mass * G * mean_speed))


def velocity_tracking_error(command_xy: np.ndarray, base_vel_xy: np.ndarray) -> float:
    """RMS error between commanded and actual planar base velocity, m/s."""
    err = np.asarray(command_xy) - np.asarray(base_vel_xy)
    return float(np.sqrt(np.mean(np.sum(err**2, axis=-1))))


def foot_slip(foot_vel_xy: np.ndarray, foot_contact: np.ndarray) -> float:
    """Mean horizontal speed of feet while in contact [T, 4, 2] and [T, 4], m/s."""
    speed = np.linalg.norm(np.asarray(foot_vel_xy), axis=-1)
    contact = np.asarray(foot_contact, dtype=bool)
    return float(speed[contact].mean()) if contact.any() else 0.0


def summarize(log: dict, dt: float, mass: float) -> dict[str, float]:
    """All per-rollout metrics from a log dict with the keys used below."""
    return {
        "velocity_tracking_error": velocity_tracking_error(log["command"][:, :2],
                                                           log["base_lin_vel"][:, :2]),
        "joint_jerk": joint_jerk(log["joint_vel"], dt),
        "action_smoothness": action_smoothness(log["actions"]),
        "torque_hf_fraction": high_freq_fraction(log["torques"], dt),
        "cost_of_transport": cost_of_transport(log["torques"], log["joint_vel"],
                                               log["base_lin_vel"][:, :2], mass),
        "foot_slip": foot_slip(log["foot_vel"][..., :2], log["foot_contact"]),
    }
