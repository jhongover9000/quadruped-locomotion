"""Mass-property formulas for bench measurements (see docs/measurement_protocol.md)."""

from __future__ import annotations

import numpy as np

G = 9.81


def com_from_scales(forces: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, float]:
    """Horizontal COM from scale readings under known support points.

    x_com = sum(F_i x_i) / sum(F_i), same for y.
    forces [k] in N (or kg, consistently); positions [k, 2] in m.
    Returns (com_xy, total_force). total_force also checks the total mass.
    """
    f = np.asarray(forces, dtype=float)
    p = np.asarray(positions, dtype=float)
    total = float(f.sum())
    return (f[:, None] * p).sum(axis=0) / total, total


def com_height_from_tilt(weight: float, span: float, f_level: float, f_tilted: float,
                         theta_rad: float) -> float:
    """COM height above the line joining two supports, from a tilt test.

    Rest the body on two supports ``span`` apart and read the support at the end you
    will raise (``f_level``). Raise that end by ``theta_rad`` and read it again
    (``f_tilted``). Then h = (F - F') L / (W tan(theta)). Use 20-30 degrees.
    """
    return (f_level - f_tilted) * span / (weight * np.tan(theta_rad))


def inertia_compound_pendulum(mass: float, pivot_to_com: float, period: float) -> float:
    """Inertia about the COM from a link swinging freely about a pivot.

    I_pivot = m g d T^2 / (4 pi^2);  I_com = I_pivot - m d^2  (parallel-axis theorem).
    """
    i_pivot = mass * G * pivot_to_com * period**2 / (4 * np.pi**2)
    return i_pivot - mass * pivot_to_com**2


def inertia_bifilar(mass: float, string_spacing: float, string_length: float,
                    period: float) -> float:
    """Inertia about the vertical axis from a bifilar (two-string) torsional pendulum.

    I = m g d^2 T^2 / (16 pi^2 L)
    """
    return mass * G * string_spacing**2 * period**2 / (16 * np.pi**2 * string_length)


def friction_from_slip_angle(theta_rad: float) -> float:
    """Static friction coefficient from the tilt angle where a foot starts to slide."""
    return float(np.tan(theta_rad))


def reflected_inertia(rotor_inertia: float, gear_ratio: float) -> float:
    """Armature: rotor inertia seen at the joint, N^2 * I_rotor."""
    return gear_ratio**2 * rotor_inertia
