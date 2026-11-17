"""Batched convex QP solver in torch (ADMM, the iteration OSQP uses).

    minimize    0.5 x'Px + q'x
    subject to  l <= Ax <= u

for a batch of problems at once, on the CPU or the GPU. Each problem is small and
dense (the MPC has 120 unknowns), so instead of OSQP's sparse factorization the
ADMM system matrix is inverted once per solve and every iteration is two batched
matrix products. The iteration count is fixed: no data-dependent control flow, so a
batch costs the same every control step.

Before iterating, unknowns are scaled to give P a unit diagonal and constraint rows
are scaled to unit norm (a one-pass version of OSQP's Ruiz equilibration). Rows with
l == u are equalities and get a larger penalty, as in OSQP.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QPSolution:
    x: torch.Tensor          # [B, n] primal solution
    y: torch.Tensor          # [B, m] constraint multipliers (unscaled)
    z: torch.Tensor          # [B, m] A x projected onto [l, u] (unscaled)
    primal_residual: torch.Tensor   # [B] max |Ax - z|
    dual_residual: torch.Tensor     # [B] max |Px + q + A'y|


def solve_qp(P: torch.Tensor, q: torch.Tensor, A: torch.Tensor, lower: torch.Tensor,
             upper: torch.Tensor, iterations: int = 100, rho: float = 0.1, sigma: float = 1e-6,
             alpha: float = 1.6, warm_start: tuple[torch.Tensor, torch.Tensor] | None = None
             ) -> QPSolution:
    """Solve a batch of QPs. Shapes: P [B, n, n] (symmetric positive semidefinite),
    q [B, n], A [B, m, n] or [m, n], lower/upper [B, m] (+-inf for no bound).

    ``warm_start`` is (x, y) from an earlier solution of a similar problem.
    """
    batch, n = q.shape
    if A.dim() == 2:
        A = A.expand(batch, *A.shape)

    # Scaling: x = D xs, constraint rows multiplied by E.
    d = torch.diagonal(P, dim1=1, dim2=2).clamp_min(1e-8).rsqrt()                # [B, n]
    Ps = P * d[:, :, None] * d[:, None, :]
    qs = q * d
    As = A * d[:, None, :]
    e = torch.linalg.vector_norm(As, dim=2).clamp_min(1e-8).reciprocal()          # [B, m]
    As = As * e[:, :, None]
    ls, us = lower * e, upper * e

    rho_vec = torch.full_like(ls, rho)
    rho_vec = torch.where((upper - lower).abs() < 1e-9, rho * 1e3, rho_vec)
    eye = torch.eye(n, dtype=P.dtype, device=P.device)
    K = Ps + sigma * eye + As.transpose(1, 2) @ (rho_vec[:, :, None] * As)
    K_inv = torch.cholesky_inverse(torch.linalg.cholesky(K))
    M_x = sigma * K_inv                                   # x~ = M_x x + M_v (rho z - y) + c
    M_v = K_inv @ As.transpose(1, 2)
    c = -(K_inv @ qs[:, :, None]).squeeze(2)

    if warm_start is None:
        x = torch.zeros_like(qs)
        y = torch.zeros_like(ls)
    else:
        x = warm_start[0] / d
        y = warm_start[1] / e
    z = (As @ x[:, :, None]).squeeze(2).clamp(ls, us)

    for _ in range(iterations):
        x_t = (M_x @ x[:, :, None] + M_v @ (rho_vec * z - y)[:, :, None]).squeeze(2) + c
        z_t = (As @ x_t[:, :, None]).squeeze(2)
        x = alpha * x_t + (1.0 - alpha) * x
        z_relaxed = alpha * z_t + (1.0 - alpha) * z
        z_new = (z_relaxed + y / rho_vec).clamp(ls, us)
        y = y + rho_vec * (z_relaxed - z_new)
        z = z_new

    x_out, y_out, z_out = x * d, y * e, z / e
    Ax = (A @ x_out[:, :, None]).squeeze(2)
    primal = (Ax - z_out).abs().amax(dim=1)
    dual = ((P @ x_out[:, :, None]).squeeze(2) + q
            + (A.transpose(1, 2) @ y_out[:, :, None]).squeeze(2)).abs().amax(dim=1)
    return QPSolution(x_out, y_out, z_out, primal, dual)
