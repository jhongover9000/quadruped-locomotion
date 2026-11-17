"""Single-rigid-body model and the MPC's quadratic programme (Di Carlo et al. 2018).

The whole robot is one rigid body of mass m and inertia I (the robot's composite
inertia in its default pose). With foot forces f_i acting at r_i from the centre of
mass, and the rotation approximated by yaw alone:

    m p_dd = sum_i f_i + m g
    d/dt (I_w w) = sum_i r_i x f_i,     I_w = Rz(yaw) I Rz(yaw)'
    d/dt [roll pitch yaw] = Rz(yaw)' w

State x = [roll, pitch, yaw, p (3), w (3), v (3), 1] (13; the constant carries
gravity), input u = the four feet's ground reaction forces in the world frame (12).
Discretized with forward Euler at the MPC step dt and condensed over the horizon,
the unknowns are the forces at every step, and the cost

    sum_k (x_k - x_ref_k)' Q (x_k - x_ref_k) + alpha |u_k|^2

becomes 0.5 U'PU + q'U. Constraints: each foot's force lies in the friction pyramid
|f_x|, |f_y| <= mu f_z, with f_min <= f_z <= f_max for a foot in stance and f = 0 for
a foot in swing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

GRAVITY = 9.81
STATE_DIM = 13
NUM_FEET = 4
FORCE_DIM = 3 * NUM_FEET
ROWS_PER_FOOT = 5


@dataclass
class RigidBody:
    mass: float                 # kg, the whole robot
    inertia: list               # 3x3, kg m^2, about the centre of mass, base frame
    com_offset: list            # 3, m, centre of mass relative to the base origin, base frame

    @classmethod
    def from_mujoco(cls, model, base_body: str, joint_names: list[str],
                    joint_pos: list[float]) -> RigidBody:
        """Composite mass and inertia of everything below ``base_body``, with the joints
        at ``joint_pos`` and the base at the origin, level."""
        import mujoco

        from quadruped_rl.algorithms.mpc_baseline.kinematics import find_id

        base = find_id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
        data = mujoco.MjData(model)
        for name, q in zip(joint_names, joint_pos):
            data.qpos[model.jnt_qposadr[find_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = q
        mujoco.mj_kinematics(model, data)
        bodies = []
        for b in range(model.nbody):
            a = b
            while a not in (0, base):
                a = int(model.body_parentid[a])
            if a == base:
                bodies.append(b)
        # Express everything relative to the base frame (the base may sit anywhere at qpos 0).
        R_base, p_base = data.xmat[base].reshape(3, 3), data.xpos[base]
        masses = model.body_mass[bodies]
        pos = (data.xipos[bodies] - p_base) @ R_base
        mass = float(masses.sum())
        com = (masses[:, None] * pos).sum(0) / mass
        inertia = np.zeros((3, 3))
        for b, m_b, p_b in zip(bodies, masses, pos):
            R = R_base.T @ data.ximat[b].reshape(3, 3)
            d = p_b - com
            inertia += R @ np.diag(model.body_inertia[b]) @ R.T + m_b * (d @ d * np.eye(3) - np.outer(d, d))
        return cls(mass, inertia.tolist(), com.tolist())

    def to_dict(self) -> dict:
        return asdict(self)


def yaw_matrix(yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack([torch.stack([c, -s, zero], -1), torch.stack([s, c, zero], -1),
                        torch.stack([zero, zero, one], -1)], -2)


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack([torch.stack([zero, -z, y], -1), torch.stack([z, zero, -x], -1),
                        torch.stack([-y, x, zero], -1)], -2)


def discrete_dynamics(yaw: torch.Tensor, feet_r: torch.Tensor, mass: float,
                      inertia: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """x_{k+1} = A x_k + B u_k for every step of the horizon.

    yaw [B, H]; feet_r [B, H, 4, 3] (foot minus centre of mass, world frame);
    inertia [3, 3] (body frame). Returns A [B, H, 13, 13] and B [B, H, 13, 12].
    """
    batch, horizon = yaw.shape
    dtype, device = feet_r.dtype, feet_r.device
    Rz = yaw_matrix(yaw)                                           # [B, H, 3, 3]
    eye3 = torch.eye(3, dtype=dtype, device=device)
    A = torch.eye(STATE_DIM, dtype=dtype, device=device).repeat(batch, horizon, 1, 1)
    A[..., 0:3, 6:9] = Rz.transpose(-1, -2) * dt
    A[..., 3:6, 9:12] = eye3 * dt
    A[..., 11, 12] = -GRAVITY * dt
    I_world_inv = Rz @ torch.linalg.inv(inertia) @ Rz.transpose(-1, -2)
    Bm = torch.zeros(batch, horizon, STATE_DIM, FORCE_DIM, dtype=dtype, device=device)
    for i in range(NUM_FEET):
        Bm[..., 6:9, 3 * i:3 * i + 3] = I_world_inv @ skew(feet_r[:, :, i]) * dt
        Bm[..., 9:12, 3 * i:3 * i + 3] = eye3 * (dt / mass)
    return A, Bm


def condensed_cost(x0: torch.Tensor, x_ref: torch.Tensor, A: torch.Tensor, Bm: torch.Tensor,
                   Q: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """P, q of 0.5 U'PU + q'U for the horizon. x0 [B, 13], x_ref [B, H, 13] (states
    1..H), A/Bm from ``discrete_dynamics``, Q [13] diagonal weights."""
    batch, horizon = x_ref.shape[:2]
    n = horizon * FORCE_DIM
    # Rows of the stacked prediction X = Sx x0 + Su U, built step by step.
    Sx_rows, Su_rows = [], []
    x_free, Su_k = x0[:, :, None], None
    for k in range(horizon):
        x_free = A[:, k] @ x_free
        block = torch.zeros(batch, STATE_DIM, n, dtype=x0.dtype, device=x0.device)
        if Su_k is not None:
            block[:, :, :k * FORCE_DIM] = A[:, k] @ Su_k[:, :, :k * FORCE_DIM]
        block[:, :, k * FORCE_DIM:(k + 1) * FORCE_DIM] = Bm[:, k]
        Su_k = block
        Sx_rows.append(x_free)
        Su_rows.append(block)
    Su = torch.cat(Su_rows, dim=1)                                 # [B, 13H, n]
    err0 = torch.cat(Sx_rows, dim=1).squeeze(2) - x_ref.reshape(batch, -1)
    w = Q.repeat(horizon)
    P = 2.0 * (Su.transpose(1, 2) @ (w[:, None] * Su))
    P = P + 2.0 * alpha * torch.eye(n, dtype=x0.dtype, device=x0.device)
    q = 2.0 * (Su.transpose(1, 2) @ (w * err0)[:, :, None]).squeeze(2)
    return P, q


def friction_constraints(contact: torch.Tensor, mu: float, f_min: float, f_max: float,
                         dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constraint matrix [5*4*H, 12*H] (the same for every robot) and bounds [B, 5*4*H]
    from the contact plan ``contact`` [B, H, 4] (True: stance). A swing foot's rows are
    left unbounded: ``remove_swing_feet`` takes its force out of the problem instead,
    which converges faster than equality constraints."""
    batch, horizon, _ = contact.shape
    device = contact.device
    foot = torch.tensor([[1.0, 0.0, -mu], [1.0, 0.0, mu], [0.0, 1.0, -mu], [0.0, 1.0, mu],
                         [0.0, 0.0, 1.0]], dtype=dtype, device=device)
    C = torch.block_diag(*[foot] * (NUM_FEET * horizon))
    inf = float("inf")
    lo_one = torch.tensor([-inf, 0.0, -inf, 0.0, f_min], dtype=dtype, device=device)
    hi_one = torch.tensor([0.0, inf, 0.0, inf, f_max], dtype=dtype, device=device)
    stance = contact.reshape(batch, -1, 1)
    lower = torch.where(stance, lo_one, -inf).reshape(batch, -1)
    upper = torch.where(stance, hi_one, inf).reshape(batch, -1)
    return C, lower, upper


def remove_swing_feet(P: torch.Tensor, q: torch.Tensor,
                      contact: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decouple the force unknowns of swing feet and give them the solution 0."""
    batch = q.shape[0]
    keep = contact.reshape(batch, -1, 1).expand(-1, -1, 3).reshape(batch, -1).to(P.dtype)
    eye = torch.eye(q.shape[1], dtype=P.dtype, device=P.device)
    P = P * keep[:, :, None] * keep[:, None, :] + eye * (1.0 - keep)[:, :, None]
    return P, q * keep
