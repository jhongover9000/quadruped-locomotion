"""The convex MPC controller, independent of any simulator.

Each control step it maps the robot's state (``RobotState``: base pose and velocity,
joint positions and velocities, velocity command) to a joint-level command
(``JointCommand``): a desired joint position and velocity plus a feed-forward torque
per joint, the interface of most quadruped motor drivers. An adapter reads the state
from a simulator (controller.py: mjlab) and turns the command into whatever the
simulator accepts (``position_target``: a PD position target).

Pieces, run every control step:

* gait: a trot clock gives each leg stance or swing, now and over the MPC horizon. With
  a zero command the robot stands on all four feet; it stops at the next instant both
  diagonal pairs touch down and starts again with the first non-zero command.
* stance legs: the QP of centroidal_model.py gives the ground reaction forces over the
  horizon; the first step's forces become joint torques through the foot Jacobians.
* swing legs: footholds from the Raibert heuristic, a smooth swing curve from lift-off
  to foothold, and a joint position / velocity target from the foot Jacobian.

Foot positions and Jacobians come from the joint angles through kinematics.py, as they
would on hardware; the contact schedule is the controller's own plan (no contact sensing).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from quadruped_rl.algorithms.mpc_baseline import centroidal_model as cm
from quadruped_rl.algorithms.mpc_baseline.kinematics import LegKinematics
from quadruped_rl.algorithms.mpc_baseline.qp import solve_qp

# Trot: diagonal pairs move together, half a period apart. Leg names as in robots/config.py.
TROT_OFFSETS = {"FR": 0.0, "FL": 0.5, "RR": 0.5, "RL": 0.0}
MOVING_THRESHOLD = 0.05      # m/s or rad/s: smaller commands mean "stand"


@dataclass
class RobotState:
    base_pos: torch.Tensor       # [B, 3] world
    base_quat: torch.Tensor      # [B, 4] world, (w, x, y, z)
    base_lin_vel: torch.Tensor   # [B, 3] world, of the base origin
    base_ang_vel: torch.Tensor   # [B, 3] world
    joint_pos: torch.Tensor      # [B, 12] (as the encoders read it)
    joint_vel: torch.Tensor      # [B, 12]
    command: torch.Tensor        # [B, 3] vx, vy in the base's heading frame (m/s), wz (rad/s)


@dataclass
class JointCommand:
    q_des: torch.Tensor          # [B, 12]
    qd_des: torch.Tensor         # [B, 12]
    tau_ff: torch.Tensor         # [B, 12]


def position_target(cmd: JointCommand, q: torch.Tensor, qd: torch.Tensor, kp: torch.Tensor,
                    kd: torch.Tensor, torque_limit: torch.Tensor) -> torch.Tensor:
    """The PD position target that turns an actuator applying tau = kp (q_t - q) - kd qd
    into tau = tau_ff + kp (q_des - q) + kd (qd_des - qd), for every q and qd.

    The target is clipped so the torque at the moment it is applied (state ``q``, ``qd``)
    stays within ``torque_limit``. A pure torque command is q_des = q, qd_des = qd.
    """
    q_t = cmd.q_des + (cmd.tau_ff + kd * cmd.qd_des) / kp
    return torch.clamp(q_t, q + (kd * qd - torque_limit) / kp, q + (kd * qd + torque_limit) / kp)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def matrix_to_rpy(R: torch.Tensor) -> torch.Tensor:
    """Z-Y-X Euler angles [..., 3] (roll, pitch, yaw)."""
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    pitch = -torch.asin(R[..., 2, 0].clamp(-1.0, 1.0))
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return torch.stack([roll, pitch, yaw], -1)


def _matvec(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (M @ v[..., None]).squeeze(-1)


class MPCController:
    """Batched over B robots. ``settings`` is configs/algorithms/mpc.yaml; ``robot`` a
    RobotConfig (default pose, nominal height, PD gains, torque limits)."""

    def __init__(self, kinematics: LegKinematics, body: cm.RigidBody, settings: dict, robot,
                 num_envs: int, control_dt: float, device: str = "cpu",
                 drift_compensation: bool = True):
        from quadruped_rl.robots.config import LEGS

        self.kin, self.body, self.settings = kinematics, body, settings
        self.num_envs, self.control_dt, self.device = num_envs, float(control_dt), device
        self.drift_compensation = drift_compensation
        dtype = self.dtype = kinematics.dtype

        def t(v):
            return torch.as_tensor(v, dtype=dtype, device=device)

        self.horizon = int(settings["horizon_steps"])
        self.dt = float(settings["dt"])
        gait = settings["gait"]
        if gait.get("type", "trot") != "trot":
            raise ValueError(f"Only the trot gait is implemented, not '{gait['type']}'")
        self.period, self.duty = float(gait["period_s"]), float(gait["duty_factor"])
        self.leg_offset = t([TROT_OFFSETS[leg] for leg in LEGS])
        self.mu = float(settings["friction_coef"])
        w = settings["weights"]
        self.Q = t([*w["orientation"], *w["position"], *w["angular_velocity"],
                    *w["linear_velocity"], 0.0])
        self.alpha = float(w["force_regularization"])
        self.f_max = float(settings.get("max_foot_force_bw", 1.0)) * body.mass * cm.GRAVITY
        self.qp_iterations = int(settings.get("qp_iterations", 100))
        self.swing_height = float(settings.get("swing_height", 0.08))
        self.swing_kp = float(settings.get("swing_kp", 0.0))
        self.swing_kd = float(settings.get("swing_kd", 0.0))
        self.max_step = float(settings.get("max_foothold_offset", 0.15))
        self.max_pos_err = float(settings.get("max_position_error", 0.1))
        self.max_yaw_err = float(settings.get("max_yaw_error", 0.2))

        self.mass = body.mass
        self.inertia = t(body.inertia)
        self.com_offset = t(body.com_offset)
        self.height = float(robot.base_height_nominal)
        self.default_q = t(robot.default_joint_pos)
        self.kp, self.kd = t(robot.actuator.kp), t(robot.actuator.kd)
        self.torque_limit = t(robot.actuator.torque_limit)
        nominal, _ = self.kin.forward(self.default_q[None])
        self.nominal_foot = nominal[0]                                    # [4, 3] base frame

        n = num_envs
        self.phase = torch.zeros(n, dtype=dtype, device=device)
        self.gait_on = torch.zeros(n, dtype=torch.bool, device=device)
        self.prev_contact = torch.ones(n, 4, dtype=torch.bool, device=device)
        self.swing_start = torch.zeros(n, 4, 3, dtype=dtype, device=device)
        # Where the robot should be: the commanded velocity integrated from where it was.
        # Feeds back position as well as velocity, so an unmodelled steady push (the mass
        # and centre of mass differ from the model's) does not make the robot drift.
        self.target_xy = torch.zeros(n, 2, dtype=dtype, device=device)
        self.target_yaw = torch.zeros(n, dtype=dtype, device=device)
        self.has_target = torch.zeros(n, dtype=torch.bool, device=device)
        self.warm: tuple[torch.Tensor, torch.Tensor] | None = None
        self.last: dict[str, torch.Tensor] = {}

    # ---- state that persists between steps ------------------------------------------

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Forget the gait (and the solver's warm start) of robots that were reset."""
        ids = slice(None) if env_ids is None else env_ids
        self.phase[ids] = 0.0
        self.gait_on[ids] = False
        self.prev_contact[ids] = True
        self.has_target[ids] = False
        if self.warm is not None:
            self.warm[0][ids] = 0.0
            self.warm[1][ids] = 0.0

    def _update_gait(self, moving: torch.Tensor) -> None:
        step = self.control_dt / self.period
        new_phase = self.phase + step
        # Both pairs are on the ground at phase 0 and 0.5 (duty 0.5): the place to stop.
        crossed = torch.floor(new_phase * 2.0) != torch.floor(self.phase * 2.0)
        stop = self.gait_on & ~moving & crossed
        start = ~self.gait_on & moving
        self.phase = torch.where(self.gait_on, new_phase % 1.0, self.phase)
        self.gait_on = (self.gait_on & ~stop) | start
        self.phase = torch.where(stop | start, torch.zeros_like(self.phase), self.phase)

    def _schedule(self, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Stance flags and swing progress (0..1) of each leg at gait phase [...] -> [..., 4]."""
        leg_phase = (phase[..., None] + self.leg_offset) % 1.0
        stance = leg_phase < self.duty
        progress = ((leg_phase - self.duty) / (1.0 - self.duty)).clamp(0.0, 1.0)
        return stance, progress

    # ---- the control step -----------------------------------------------------------

    def step(self, s: RobotState) -> JointCommand:
        n, dtype, dev = self.num_envs, self.dtype, self.device
        R = quat_to_matrix(s.base_quat)
        rpy = matrix_to_rpy(R)
        yaw = rpy[:, 2]
        foot_b, J = self.kin.forward(s.joint_pos)                          # [n,4,3], [n,4,3,3]
        idx = self.kin.joint_index
        q_leg, qd_leg = s.joint_pos[:, idx], s.joint_vel[:, idx]           # [n, 4, 3]
        foot_vel_b = _matvec(J, qd_leg)
        foot_w = s.base_pos[:, None] + _matvec(R[:, None], foot_b)

        cmd = s.command
        moving = (cmd[:, :2].norm(dim=1) > MOVING_THRESHOLD) | (cmd[:, 2].abs() > MOVING_THRESHOLD)
        self._update_gait(moving)
        k_steps = torch.arange(self.horizon, dtype=dtype, device=dev)
        plan_phase = self.phase[:, None] + k_steps * (self.dt / self.period)
        contact_plan, _ = self._schedule(plan_phase)                        # [n, H, 4]
        contact_plan = contact_plan | ~self.gait_on[:, None, None]
        contact, progress = self._schedule(self.phase)
        contact = contact | ~self.gait_on[:, None]
        progress = torch.where(contact, torch.zeros_like(progress), progress)
        lift = self.prev_contact & ~contact
        self.swing_start = torch.where(lift[..., None], foot_w, self.swing_start)
        self.prev_contact = contact

        # Desired motion in the world frame.
        Rz = cm.yaw_matrix(yaw)
        v_cmd_h = torch.stack([cmd[:, 0], cmd[:, 1], torch.zeros_like(cmd[:, 0])], -1)
        v_des = _matvec(Rz, v_cmd_h)
        w_des = cmd[:, 2]
        com = s.base_pos + _matvec(R, self.com_offset)
        com_vel = s.base_lin_vel + torch.linalg.cross(s.base_ang_vel, com - s.base_pos)
        self._update_target(com[:, :2], yaw, v_des[:, :2], w_des)

        # Footholds (Raibert heuristic) and swing targets.
        t_swing = (1.0 - self.duty) * self.period
        t_stance = self.duty * self.period
        t_land = (1.0 - progress) * t_swing                                 # [n, 4]
        yaw_land = yaw[:, None] + w_des[:, None] * t_land
        hip_land = (s.base_pos[:, None] + v_des[:, None] * t_land[..., None]
                    + _matvec(cm.yaw_matrix(yaw_land), self.nominal_foot))
        spin = torch.stack([torch.zeros_like(w_des), torch.zeros_like(w_des), w_des], -1)
        offset = (v_des * (t_stance / 2) + (self.height / cm.GRAVITY) ** 0.5 * (com_vel - v_des)
                  + 0.5 * self.height / cm.GRAVITY * torch.linalg.cross(com_vel, spin))
        offset = offset[:, :2]
        offset = offset * (self.max_step / offset.norm(dim=1, keepdim=True).clamp_min(self.max_step))
        foothold = hip_land.clone()
        foothold[..., :2] = hip_land[..., :2] + offset[:, None]
        foothold[..., 2] = self.swing_start[..., 2]

        s_mid = (progress + 0.5 * self.control_dt / t_swing).clamp(0.0, 1.0)
        blend = 3 * s_mid**2 - 2 * s_mid**3
        blend_rate = 6 * s_mid * (1 - s_mid) / t_swing
        p_des = self.swing_start + (foothold - self.swing_start) * blend[..., None]
        v_des_foot = (foothold - self.swing_start) * blend_rate[..., None]
        p_des[..., 2] = (self.swing_start[..., 2] + self.swing_height * 16 * s_mid**2 * (1 - s_mid)**2)
        v_des_foot[..., 2] = self.swing_height * 32 * s_mid * (1 - s_mid) * (1 - 2 * s_mid) / t_swing
        Rt = R.transpose(1, 2)[:, None]
        rel = p_des - s.base_pos[:, None]
        p_des_b = _matvec(Rt, rel)
        v_des_b = _matvec(Rt, v_des_foot - s.base_lin_vel[:, None]
                          - torch.linalg.cross(s.base_ang_vel[:, None].expand_as(rel), rel))
        err = p_des_b - foot_b
        JJt = J @ J.transpose(-1, -2) + 1e-4 * torch.eye(3, dtype=dtype, device=dev)
        Jt = J.transpose(-1, -2)
        swing_q = q_leg + _matvec(Jt, torch.linalg.solve(JJt, err))
        swing_qd = _matvec(Jt, torch.linalg.solve(JJt, v_des_b))
        swing_tau = _matvec(Jt, self.swing_kp * err + self.swing_kd * (v_des_b - foot_vel_b))

        # Stance forces from the MPC.
        forces, sol = self._solve_mpc(rpy, com, s.base_ang_vel, com_vel, v_cmd_h, w_des,
                                      foot_w, foothold, contact, contact_plan)
        forces = forces * contact[..., None]
        stance_tau = -_matvec(Jt, _matvec(Rt, forces))
        lim = self.torque_limit[idx]
        stance_tau = torch.maximum(torch.minimum(stance_tau, lim), -lim)
        lead = 0.5 * self.control_dt if self.drift_compensation else 0.0
        stance_q = q_leg + qd_leg * lead

        c = contact[..., None]
        q_des = torch.where(c, stance_q, swing_q)
        qd_des = torch.where(c, qd_leg, swing_qd)
        tau_ff = torch.where(c, stance_tau, swing_tau)
        out = JointCommand(*(torch.empty(n, 12, dtype=dtype, device=dev) for _ in range(3)))
        for full, leg in ((out.q_des, q_des), (out.qd_des, qd_des), (out.tau_ff, tau_ff)):
            full[:, idx.reshape(-1)] = leg.reshape(n, -1)
        self.last = {"forces": forces, "contact": contact, "foothold": foothold, "p_des": p_des,
                     "qp_primal": sol.primal_residual, "qp_dual": sol.dual_residual}
        return out

    def _update_target(self, com_xy, yaw, v_xy, w_des) -> None:
        """Advance the target pose by the command; keep it within reach of the robot (a
        fall or a push must not leave a large error that the robot then rushes to close)."""
        self.target_xy = torch.where(self.has_target[:, None],
                                     self.target_xy + v_xy * self.control_dt, com_xy)
        self.target_yaw = torch.where(self.has_target, self.target_yaw + w_des * self.control_dt, yaw)
        self.has_target[:] = True
        err = self.target_xy - com_xy
        err = err * (self.max_pos_err / err.norm(dim=1, keepdim=True).clamp_min(self.max_pos_err))
        self.target_xy = com_xy + err
        yaw_err = torch.remainder(self.target_yaw - yaw + torch.pi, 2 * torch.pi) - torch.pi
        self.target_yaw = yaw + yaw_err.clamp(-self.max_yaw_err, self.max_yaw_err)

    def _solve_mpc(self, rpy, com, ang_vel, com_vel, v_cmd_h, w_des, foot_w, foothold,
                   contact, contact_plan):
        n, H, dtype, dev = self.num_envs, self.horizon, self.dtype, self.device
        k = torch.arange(H + 1, dtype=dtype, device=dev)
        # The target yaw, unwrapped next to the measured yaw (the model's yaw is continuous).
        yaw_err = torch.remainder(self.target_yaw - rpy[:, 2] + torch.pi, 2 * torch.pi) - torch.pi
        yaw_ref = (rpy[:, 2] + yaw_err)[:, None] + w_des[:, None] * k * self.dt  # [n, H+1]
        v_ref = _matvec(cm.yaw_matrix(yaw_ref), v_cmd_h[:, None])             # [n, H+1, 3]
        travel = torch.cumsum(
            torch.cat([torch.zeros_like(v_ref[:, :1]), v_ref[:, :-1]], 1) * self.dt, 1)
        pos_ref = travel.clone()
        pos_ref[..., :2] += self.target_xy[:, None]
        pos_ref[..., 2] = self.height + float(self.com_offset[2])
        x_ref = torch.zeros(n, H + 1, cm.STATE_DIM, dtype=dtype, device=dev)
        x_ref[..., 2] = yaw_ref
        x_ref[..., 3:6] = pos_ref
        x_ref[..., 8] = w_des[:, None]
        x_ref[..., 9:12] = v_ref
        x0 = torch.cat([rpy, com, ang_vel, com_vel, torch.ones_like(rpy[:, :1])], 1)

        # Where each foot pushes over the horizon: where it stands, or where it will land.
        foot_plan = torch.where(contact[..., None], foot_w, foothold)          # [n, 4, 3]
        com_k = com[:, None] + travel[:, :H]                                  # [n, H, 3]
        com_k[:, 1:, 2] = pos_ref[:, 1:H, 2]
        feet_r = foot_plan[:, None] - com_k[:, :, None]                       # [n, H, 4, 3]
        A, Bm = cm.discrete_dynamics(yaw_ref[:, :H], feet_r, self.mass, self.inertia, self.dt)
        P, q = cm.condensed_cost(x0, x_ref[:, 1:], A, Bm, self.Q, self.alpha)
        C, lower, upper = cm.friction_constraints(contact_plan, self.mu, 0.0, self.f_max, dtype)
        P, q = cm.remove_swing_feet(P, q, contact_plan)
        sol = solve_qp(P, q, C, lower, upper, iterations=self.qp_iterations, warm_start=self.warm)
        self.warm = (sol.x.clone(), sol.y.clone())
        return sol.x[:, :cm.FORCE_DIM].reshape(n, 4, 3), sol
