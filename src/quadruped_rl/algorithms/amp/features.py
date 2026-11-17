"""What the AMP discriminator sees of one robot state, from a clip frame or the simulator.

Both sides go through ``amp_features``, so clip and simulator features cannot drift
apart. The features leave out the absolute position and heading (a style is the same
wherever the robot walks and whichever way it faces):

    joint_pos       q                       [J]  rad
    joint_vel       qd                      [J]  rad/s
    base_lin_vel    v_b = R^T v_w           [3]  trunk origin velocity, base frame
    base_ang_vel    w_b = R^T w_w           [3]  base frame
    gravity         g_b = R^T [0, 0, -1]    [3]  tilt
    base_height     z                       [1]  m, above the environment's origin

so 2 J + 10 numbers (34 for a 12-joint robot). Clips are recorded on flat ground; on
generated rough terrain the height is measured from the environment's origin, which is
only approximately the ground under the robot.
"""

from __future__ import annotations

import numpy as np
import torch


def feature_dim(num_joints: int) -> int:
    return 2 * num_joints + 10


def quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """R(q)^T v for unit quaternions q = (w, x, y, z) [..., 4] and vectors [..., 3]."""
    w, xyz = quat[..., :1], quat[..., 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec - w * t + torch.cross(xyz, t, dim=-1)


def amp_features(joint_pos: torch.Tensor, joint_vel: torch.Tensor, base_quat_w: torch.Tensor,
                 base_lin_vel_w: torch.Tensor, base_ang_vel_w: torch.Tensor,
                 base_height: torch.Tensor) -> torch.Tensor:
    """[..., J], [..., J], [..., 4] (wxyz), [..., 3], [..., 3], [...] -> [..., 2 J + 10]."""
    gravity = torch.tensor([0.0, 0.0, -1.0], dtype=base_quat_w.dtype,
                           device=base_quat_w.device).expand_as(base_lin_vel_w)
    return torch.cat([
        joint_pos,
        joint_vel,
        quat_rotate_inverse(base_quat_w, base_lin_vel_w),
        quat_rotate_inverse(base_quat_w, base_ang_vel_w),
        quat_rotate_inverse(base_quat_w, gravity),
        base_height.unsqueeze(-1),
    ], dim=-1)


def clip_features(clip, root_body: str) -> torch.Tensor:
    """Features of every frame of a clip (motions/format.py), [T, 2 J + 10], float32."""
    i = clip.body_names.index(root_body)

    def t(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, dtype=np.float32))

    return amp_features(t(clip.joint_pos), t(clip.joint_vel), t(clip.body_quat_w[:, i]),
                        t(clip.body_lin_vel_w[:, i]), t(clip.body_ang_vel_w[:, i]),
                        t(clip.body_pos_w[:, i, 2]))


def env_features(env, entity: str = "robot") -> torch.Tensor:
    """Features of every environment's current state, [N, 2 J + 10]."""
    data = env.scene[entity].data
    height = data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return amp_features(data.joint_pos, data.joint_vel, data.root_link_quat_w,
                        data.root_link_lin_vel_w, data.root_link_ang_vel_w, height)
