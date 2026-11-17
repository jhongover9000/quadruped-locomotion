"""Convex MPC baseline: QP solver, rigid-body model, kinematics and the torque-to-
position-target inversion, on the CPU. The closed loop in simulation runs with
`pytest -m sim` (test_mpc_stands_and_trots)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

from quadruped_rl.algorithms.mpc_baseline import centroidal_model as cm
from quadruped_rl.algorithms.mpc_baseline.core import (
    JointCommand,
    MPCController,
    RobotState,
    position_target,
)
from quadruped_rl.algorithms.mpc_baseline.kinematics import LegKinematics
from quadruped_rl.algorithms.mpc_baseline.qp import solve_qp
from quadruped_rl.robots import load_robot
from quadruped_rl.utils.config import load_group

INF = float("inf")


# ---- QP solver ----------------------------------------------------------------------

def test_qp_solver_finds_known_solutions_with_active_constraints():
    """min |x - (2, 2)|^2 (P = 2I, q = -4) under
    x1 + x2 <= 2, x1 >= 0, x2 <= 0.5 and, in the third problem only, x1 = x2.
    Solutions by hand from the KKT conditions."""
    A = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    P = 2.0 * torch.eye(2, dtype=torch.float64).repeat(3, 1, 1)
    q = torch.tensor([[-4.0, -4.0], [-1.0, -0.6], [-4.0, -4.0]], dtype=torch.float64)
    lower = torch.tensor([[-INF, 0.0, -INF, -INF]] * 2 + [[-INF, 0.0, -INF, 0.0]],
                         dtype=torch.float64)
    upper = torch.tensor([[2.0, INF, 0.5, INF]] * 2 + [[2.0, INF, 0.5, 0.0]], dtype=torch.float64)
    sol = solve_qp(P, q, A, lower, upper, iterations=400)
    expected = torch.tensor([[1.5, 0.5],      # x1 + x2 <= 2 and x2 <= 0.5 both active
                             [0.5, 0.3],      # interior: the unconstrained minimum
                             [0.5, 0.5]],     # x2 <= 0.5 and the equality active
                            dtype=torch.float64)
    assert torch.allclose(sol.x, expected, atol=1e-4)
    # Multipliers: P x + q + A'y = 0 with y >= 0 on active upper bounds.
    assert torch.allclose(sol.y[0], torch.tensor([1.0, 0.0, 2.0, 0.0], dtype=torch.float64),
                          atol=1e-3)
    assert (sol.primal_residual < 1e-5).all() and (sol.dual_residual < 1e-3).all()


def test_qp_solver_projects_a_force_onto_the_friction_pyramid():
    """The closest force to (10, 0, 10) N with |fx|, |fy| <= 0.6 fz lies on the face
    fx = 0.6 fz: (10, 0, 10) - (4 / 1.36) (1, 0, -0.6)."""
    C, lower, upper = cm.friction_constraints(torch.ones(1, 1, 4, dtype=torch.bool), 0.6, 0.0,
                                              100.0, dtype=torch.float32)
    target = torch.tensor([10.0, 0.0, 10.0]).repeat(4)
    P = 2.0 * torch.eye(12)[None]
    sol = solve_qp(P, -2.0 * target[None], C, lower, upper, iterations=300)
    expected = torch.tensor([10.0 - 4.0 / 1.36, 0.0, 10.0 + 2.4 / 1.36])
    assert torch.allclose(sol.x[0].reshape(4, 3), expected.expand(4, 3), atol=2e-2)


# ---- robot model --------------------------------------------------------------------

@pytest.fixture(scope="module")
def go1():
    robot = load_robot("go1")
    model = mujoco.MjModel.from_xml_path(str(robot.model_file))
    return robot, model


def test_kinematics_match_mujoco(go1):
    robot, model = go1
    kin = LegKinematics.from_mujoco(model, robot.base_body, robot.foot_sites, robot.joint_names,
                                    dtype=torch.float64)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    q = np.array(robot.default_joint_pos) + rng.uniform(-0.4, 0.4, size=(3, 12))
    foot, jac = kin.forward(torch.tensor(q))
    base = model.body(robot.base_body).id
    jacp = np.zeros((3, model.nv))
    for b in range(3):
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0                       # level base at the origin
        data.qpos[7:] = q[b]                     # Go1: joints follow the free joint in order
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        for leg, site in enumerate(robot.foot_sites):
            sid = model.site(site).id
            rel = data.site_xpos[sid] - data.xpos[base]
            assert np.allclose(foot[b, leg].numpy(), rel, atol=1e-9)
            mujoco.mj_jacSite(model, data, jacp, None, sid)
            cols = [6 + 3 * leg + j for j in range(3)]
            assert np.allclose(jac[b, leg].numpy(), jacp[:, cols], atol=1e-9)


def _standing_controller(go1, num_envs=2):
    robot, model = go1
    kin = LegKinematics.from_mujoco(model, robot.base_body, robot.foot_sites, robot.joint_names)
    body = cm.RigidBody.from_mujoco(model, robot.base_body, robot.joint_names,
                                    robot.default_joint_pos)
    settings = load_group("algorithms", "mpc")
    ctrl = MPCController(kin, body, settings, robot, num_envs, control_dt=0.02)
    q = torch.tensor(robot.default_joint_pos).repeat(num_envs, 1)
    zeros = torch.zeros(num_envs, 3)
    state = RobotState(base_pos=torch.tensor([0.0, 0.0, robot.base_height_nominal]).repeat(num_envs, 1),
                       base_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(num_envs, 1),
                       base_lin_vel=zeros, base_ang_vel=zeros, joint_pos=q,
                       joint_vel=torch.zeros_like(q), command=zeros)
    return ctrl, body, state


def test_rigid_body_matches_the_model(go1):
    robot, model = go1
    body = cm.RigidBody.from_mujoco(model, robot.base_body, robot.joint_names, robot.default_joint_pos)
    assert body.mass == pytest.approx(float(model.body_mass[1:].sum()))
    inertia = np.array(body.inertia)
    assert np.allclose(inertia, inertia.T) and (np.linalg.eigvalsh(inertia) > 0).all()
    assert inertia[0, 0] < inertia[1, 1] < inertia[2, 2]      # a long, flat body


def test_free_fall_prediction():
    A, _ = cm.discrete_dynamics(torch.zeros(1, 3), torch.zeros(1, 3, 4, 3), 10.0,
                                torch.eye(3), 0.1)
    x = torch.zeros(1, 13)
    x[0, 12] = 1.0
    for k in range(3):
        x = (A[:, k] @ x[..., None]).squeeze(-1)
    assert x[0, 11].item() == pytest.approx(-3 * 9.81 * 0.1)


def test_standing_robot_gets_foot_forces_summing_to_its_weight(go1):
    ctrl, body, state = _standing_controller(go1)
    cmd = ctrl.step(state)
    forces = ctrl.last["forces"]                                  # [B, 4, 3], world frame
    weight = body.mass * cm.GRAVITY
    assert ctrl.last["contact"].all()
    assert torch.allclose(forces.sum(1)[:, 2], torch.full((2,), weight), rtol=0.02)
    assert forces.sum(1)[:, :2].abs().max() < 0.02 * weight
    assert (forces[..., 2] > 0).all()
    # Stance torques: the legs push down (calf torques negative-signed on the Go1), within limits.
    lim = torch.tensor(go1[0].actuator.torque_limit)
    assert (cmd.tau_ff.abs() <= lim).all() and cmd.tau_ff.abs().max() > 1.0


def test_standing_robot_pushed_aside_is_pulled_back(go1):
    """The target position holds where the robot stood: a robot that drifted 5 cm forward
    (an unmodelled steady push) gets a net backward force, not just velocity damping."""
    ctrl, body, state = _standing_controller(go1)
    ctrl.step(state)                                              # sets the target pose
    state.base_pos = state.base_pos + torch.tensor([[0.05, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ctrl.step(state)
    fx = ctrl.last["forces"].sum(1)[:, 0]
    assert fx[0] < -0.01 * body.mass * cm.GRAVITY
    assert fx[1].abs() < 0.02 * body.mass * cm.GRAVITY


def test_trot_command_starts_the_gait(go1):
    ctrl, _, state = _standing_controller(go1)
    state.command = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ctrl.step(state)
    contact = ctrl.last["contact"]
    assert contact[0].tolist() == [True, False, False, True]      # FR+RL stance, FL+RR swing
    assert contact[1].all()
    assert (ctrl.last["forces"][0, [1, 2]] == 0).all()


# ---- torque to position target ------------------------------------------------------

def test_position_target_realizes_the_joint_command():
    torch.manual_seed(0)
    n = 12
    kp, kd = torch.full((n,), 20.0), torch.full((n,), 1.0)
    limit = torch.full((n,), 1e6)
    cmd = JointCommand(torch.randn(4, n), torch.randn(4, n), torch.randn(4, n))
    q, qd = torch.randn(4, n), torch.randn(4, n)
    q_t = position_target(cmd, q, qd, kp, kd, limit)
    for q_now, qd_now in ((q, qd), (q + 0.1, qd - 0.5)):          # holds for any later state
        applied = kp * (q_t - q_now) - kd * qd_now
        wanted = cmd.tau_ff + kp * (cmd.q_des - q_now) + kd * (cmd.qd_des - qd_now)
        assert torch.allclose(applied, wanted, atol=1e-4)
    # A pure torque: q_des = q, qd_des = qd gives exactly tau_ff now.
    pure = JointCommand(q, qd, cmd.tau_ff)
    assert torch.allclose(kp * (position_target(pure, q, qd, kp, kd, limit) - q) - kd * qd,
                          cmd.tau_ff, atol=1e-4)
    # Clipped to the torque limit.
    big = JointCommand(q, qd, torch.full((4, n), 50.0))
    tau = kp * (position_target(big, q, qd, kp, kd, torch.full((n,), 5.0)) - q) - kd * qd
    assert torch.allclose(tau, torch.full((4, n), 5.0), atol=1e-4)


# ---- the run's checkpoint -----------------------------------------------------------

def test_train_writes_a_checkpoint_the_scripts_find(tmp_path):
    from quadruped_rl.algorithms import get_algorithm
    from quadruped_rl.algorithms.checkpoints import (
        checkpoint_format,
        list_checkpoints,
        load_checkpoint,
    )

    agent = get_algorithm("mpc")(robot=load_robot("go1"), task_cfg=load_group("tasks", "flat_velocity"),
                                 algo_cfg=load_group("algorithms", "mpc"), run_dir=tmp_path)
    agent.train()
    found = list_checkpoints(tmp_path)
    assert [p.name for p in found] == ["model_0.pt"]
    ckpt = load_checkpoint(found[0])
    assert checkpoint_format(ckpt) == "mpc"
    assert ckpt["rigid_body"]["mass"] > 10.0


# ---- closed loop in simulation ------------------------------------------------------

@pytest.mark.sim
def test_mpc_stands_and_trots(tmp_path):
    from quadruped_rl.algorithms import get_algorithm
    from quadruped_rl.envs.commands import COMMAND_NAME

    robot, task = load_robot("go1"), load_group("tasks", "flat_velocity")
    agent = get_algorithm("mpc")(robot=robot, task_cfg=task, algo_cfg=load_group("algorithms", "mpc"),
                                 run_dir=tmp_path)
    agent.train()
    env = agent.make_env(4, play=True)
    viewer_env, policy = agent.make_policy(env, tmp_path / "checkpoints" / "model_0.pt")
    term = env.command_manager.get_term(COMMAND_NAME)
    env.reset()
    falls, vx = 0, []
    with torch.inference_mode():
        for t in range(150):                     # 1 s standing, then 2 s at 0.5 m/s
            term.vel_command_b[:] = torch.tensor([0.0 if t < 50 else 0.5, 0.0, 0.0])
            term.is_standing_env[:] = False
            term.time_left[:] = 1e9
            viewer_env.step(policy(viewer_env.get_observations()))
            falls += int(env.reset_terminated.sum())
            if t >= 100:
                vx.append(env.scene["robot"].data.root_link_lin_vel_b[:, 0].mean().item())
    env.close()
    assert falls == 0
    assert np.mean(vx) == pytest.approx(0.5, abs=0.15)
