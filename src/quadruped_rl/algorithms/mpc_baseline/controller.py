"""Convex MPC behind the Agent interface, so training, evaluation, the viewer and the
recorder run it like any learned policy.

There is nothing to learn: ``train()`` builds the controller from the robot's MuJoCo
model (leg kinematics, mass, inertia) and saves it with its settings as the run's
checkpoint, checkpoints/model_0.pt. ``make_policy`` rebuilds the controller from that
file, so a run plays back the controller it recorded even if the YAML changes later.

The controller itself (core.py) knows nothing about mjlab. ``MjlabPolicy`` is the
adapter: it reads the simulator state the controller needs (which is more than the
policy observation: base pose and velocity are ground truth, no state estimator) and
turns the controller's joint command into the environment's position-target action.
"""

from __future__ import annotations

import re
from pathlib import Path

from quadruped_rl.algorithms.base import Agent

CHECKPOINT = "checkpoints/model_0.pt"


def build_controller_data(robot, algo_cfg: dict) -> dict:
    """Everything the controller needs, as plain numbers: the checkpoint's contents."""
    import mujoco

    from quadruped_rl.algorithms.mpc_baseline.centroidal_model import RigidBody
    from quadruped_rl.algorithms.mpc_baseline.kinematics import LegKinematics

    model = mujoco.MjModel.from_xml_path(str(robot.model_file))
    kin = LegKinematics.from_mujoco(model, robot.base_body, robot.foot_sites, robot.joint_names)
    body = RigidBody.from_mujoco(model, robot.base_body, robot.joint_names, robot.default_joint_pos)
    return {"mpc": algo_cfg, "robot": robot.name, "kinematics": kin.params,
            "rigid_body": body.to_dict()}


class ConvexMPC(Agent):
    task_kinds = ("velocity",)
    checkpoint_patterns = ("checkpoints/model_*.pt",)

    def train(self, num_envs: int | None = None, max_iterations: int | None = None,
              resume: Path | None = None) -> None:
        import torch

        data = build_controller_data(self.robot, self.algo_cfg)
        path = self.run_dir / CHECKPOINT
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
        body = data["rigid_body"]
        print(f"Convex MPC for {self.robot.name}: mass {body['mass']:.2f} kg, inertia diagonal "
              f"{[round(body['inertia'][i][i], 4) for i in range(3)]} kg m^2. Nothing to train; "
              f"controller saved to {path}")

    def make_policy(self, env, checkpoint: Path):
        from quadruped_rl.algorithms.checkpoints import load_checkpoint, require_format
        from quadruped_rl.algorithms.mpc_baseline.centroidal_model import RigidBody
        from quadruped_rl.algorithms.mpc_baseline.core import MPCController
        from quadruped_rl.algorithms.mpc_baseline.kinematics import LegKinematics

        ckpt = load_checkpoint(checkpoint)
        require_format(ckpt, "mpc", Path(checkpoint))
        controller = MPCController(LegKinematics(ckpt["kinematics"], device=str(env.device)),
                                   RigidBody(**ckpt["rigid_body"]), ckpt["mpc"], self.robot,
                                   env.num_envs, float(env.step_dt), device=str(env.device))
        return env, MjlabPolicy(env, controller, self.robot, self.task_cfg)


class MjlabPolicy:
    """policy(obs) -> actions for an mjlab velocity environment (the observation is unused)."""

    def __init__(self, env, controller, robot, task_cfg: dict):
        import torch

        from quadruped_rl.envs.builder import make_action
        from quadruped_rl.envs.commands import COMMAND_NAME

        self.env, self.controller = env, controller
        self.entity = env.scene["robot"]
        if list(self.entity.joint_names) != list(robot.joint_names):
            raise ValueError("joint order of the environment differs from the robot config")
        self.command = env.command_manager.get_term(COMMAND_NAME)
        scale = make_action(robot, task_cfg["action"]).scale
        dev = env.device

        def t(v):
            return torch.as_tensor(v, dtype=torch.float32, device=dev)

        self.scale = t([scale[re.escape(n)] for n in robot.joint_names])
        self.offset = t(robot.default_joint_pos)
        act = robot.actuator
        self.kp, self.kd, self.torque_limit = t(act.kp), t(act.kd), t(act.torque_limit)

    def state(self):
        from quadruped_rl.algorithms.mpc_baseline.core import RobotState

        d = self.entity.data
        return RobotState(base_pos=d.root_link_pos_w, base_quat=d.root_link_quat_w,
                          base_lin_vel=d.root_link_lin_vel_w, base_ang_vel=d.root_link_ang_vel_w,
                          joint_pos=d.joint_pos_biased, joint_vel=d.joint_vel,
                          command=self.command.command)

    def __call__(self, obs):
        import torch

        from quadruped_rl.algorithms.mpc_baseline.core import position_target

        with torch.no_grad():
            reset = (self.env.episode_length_buf == 0).nonzero().flatten()
            if len(reset):
                self.controller.reset(reset)
            s = self.state()
            cmd = self.controller.step(s)
            q_target = position_target(cmd, s.joint_pos, s.joint_vel, self.kp, self.kd,
                                       self.torque_limit)
            return (q_target - self.offset) / self.scale
