"""Leg kinematics read from a MuJoCo model: foot positions and foot Jacobians in the
base frame, batched in torch.

``LegKinematics.from_mujoco`` walks each foot site's body chain up to the base body
and folds it into three hinge joints (fixed bodies in between are merged into the
neighbouring transforms). The result is a handful of numbers per leg (``params``), so
a controller can be saved, and rebuilt without the model or the simulator.
"""

from __future__ import annotations

import numpy as np
import torch

LEG_JOINTS = 3


def find_id(model, objtype, name: str) -> int:
    """Id of ``name`` in ``model``; also matches names an mjlab scene prefixes ("robot/x")."""
    import mujoco

    i = mujoco.mj_name2id(model, objtype, name)
    if i >= 0:
        return i
    count = {mujoco.mjtObj.mjOBJ_BODY: model.nbody, mujoco.mjtObj.mjOBJ_SITE: model.nsite,
             mujoco.mjtObj.mjOBJ_JOINT: model.njnt}[objtype]
    for j in range(count):
        if (mujoco.mj_id2name(model, objtype, j) or "").endswith("/" + name):
            return j
    raise KeyError(f"'{name}' not found in the MuJoCo model")


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues: rotation by ``angle`` [...] about the unit ``axis`` [3] -> [..., 3, 3]."""
    kx, ky, kz = axis
    zero = torch.zeros((), dtype=axis.dtype, device=axis.device)
    K = torch.stack([torch.stack([zero, -kz, ky]), torch.stack([kz, zero, -kx]),
                     torch.stack([-ky, kx, zero])])
    s, c = torch.sin(angle)[..., None, None], torch.cos(angle)[..., None, None]
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return eye + s * K + (1 - c) * (K @ K)


class LegKinematics:
    """Four legs of three hinge joints each.

    params (all in the parent body's frame): ``offset_pos`` [4, 3, 3] and ``offset_rot``
    [4, 3, 3, 3] place each joint's body in the previous one (the first in the base),
    ``axis`` and ``anchor`` [4, 3, 3] are the hinge axis and point in that body,
    ``foot`` [4, 3] is the foot site in the last body, and ``joint_index`` [4, 3] says
    which entries of the joint vector belong to each leg.
    """

    def __init__(self, params: dict, device: str = "cpu", dtype=torch.float32):
        self.params = params
        self.device, self.dtype = device, dtype

        def t(key):
            return torch.as_tensor(np.asarray(params[key]), dtype=dtype, device=device)

        self.offset_pos, self.offset_rot = t("offset_pos"), t("offset_rot")
        self.axis, self.anchor, self.foot = t("axis"), t("anchor"), t("foot")
        self.joint_index = torch.as_tensor(np.asarray(params["joint_index"]), dtype=torch.long,
                                           device=device)

    @classmethod
    def from_mujoco(cls, model, base_body: str, foot_sites: list[str], joint_names: list[str],
                    **kwargs) -> LegKinematics:
        import mujoco

        base = find_id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
        joint_ids = [find_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        keys = ("offset_pos", "offset_rot", "axis", "anchor", "foot", "joint_index")
        params = {k: [] for k in keys}
        for site in foot_sites:
            sid = find_id(model, mujoco.mjtObj.mjOBJ_SITE, site)
            chain, b = [], int(model.site_bodyid[sid])
            while b != base:
                if b == 0:
                    raise ValueError(f"foot site '{site}' is not below the base body '{base_body}'")
                chain.append(b)
                b = int(model.body_parentid[b])
            leg = {k: [] for k in keys}
            pos, rot = np.zeros(3), np.eye(3)           # fixed transform since the last joint
            for b in reversed(chain):
                pos, rot = pos + rot @ model.body_pos[b], rot @ _quat_to_mat(model.body_quat[b])
                jnts = range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b])
                for j in jnts:
                    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                        raise ValueError(f"leg of '{site}': only hinge joints are supported")
                    if j not in joint_ids:
                        raise ValueError(f"joint '{model.joint(j).name}' is not in joint_names")
                if len(jnts) > 1:
                    raise ValueError(f"leg of '{site}': one hinge per body is supported")
                if len(jnts) == 1:
                    j = jnts[0]
                    leg["offset_pos"].append(pos)
                    leg["offset_rot"].append(rot)
                    leg["axis"].append(model.jnt_axis[j] / np.linalg.norm(model.jnt_axis[j]))
                    leg["anchor"].append(model.jnt_pos[j].copy())
                    leg["joint_index"].append(joint_ids.index(j))
                    pos, rot = np.zeros(3), np.eye(3)
            if len(leg["axis"]) != LEG_JOINTS:
                raise ValueError(f"leg of '{site}' has {len(leg['axis'])} joints, expected {LEG_JOINTS}")
            for k in keys[:4] + ("joint_index",):
                params[k].append(np.array(leg[k]))
            params["foot"].append(pos + rot @ model.site_pos[sid])
        params = {k: np.array(v).tolist() for k, v in params.items()}
        return cls(params, **kwargs)

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint positions [B, 12] -> foot positions [B, 4, 3] and Jacobians [B, 4, 3, 3]
        (d foot / d leg joints), both in the base frame."""
        batch = q.shape[0]
        q_leg = q[:, self.joint_index]                                     # [B, 4, 3]
        rot = torch.eye(3, dtype=q.dtype, device=q.device).expand(batch, 4, 3, 3)
        pos = torch.zeros(batch, 4, 3, dtype=q.dtype, device=q.device)
        anchors, axes = [], []
        for j in range(LEG_JOINTS):
            pos = pos + (rot @ self.offset_pos[:, j, :, None]).squeeze(-1)
            rot = rot @ self.offset_rot[:, j]
            anchor = pos + (rot @ self.anchor[:, j, :, None]).squeeze(-1)
            axes.append((rot @ self.axis[:, j, :, None]).squeeze(-1))
            anchors.append(anchor)
            spin = torch.stack([axis_angle_matrix(self.axis[leg, j], q_leg[:, leg, j])
                                for leg in range(4)], dim=1)               # [B, 4, 3, 3]
            rot = rot @ spin
            pos = anchor - (rot @ self.anchor[:, j, :, None]).squeeze(-1)
        foot = pos + (rot @ self.foot[:, :, None]).squeeze(-1)
        jac = torch.stack([torch.linalg.cross(a, foot - p) for a, p in zip(axes, anchors)], dim=-1)
        return foot, jac
