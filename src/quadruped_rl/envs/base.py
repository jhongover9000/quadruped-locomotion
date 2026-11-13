"""QuadrupedEnv: the batched environment every algorithm talks to.

A thin layer over mjlab's ManagerBasedRlEnv (thousands of robots stepped in parallel on
the GPU). Build one with ``envs.builder.make_env``.

    obs, info = env.reset()              # obs = {"actor": [N, d_a], "critic": [N, d_c]}
    obs, reward, terminated, truncated, info = env.step(actions)   # actions [N, 12]

``terminated`` marks failures (falls); ``truncated`` marks time-outs, where the
critic should bootstrap because the robot didn't fail. Environments reset themselves
when either is set.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv


class QuadrupedEnv(ManagerBasedRlEnv):
    def __init__(self, cfg, device: str, clip_reward_at_zero: bool = False, render_mode=None):
        super().__init__(cfg, device=device, render_mode=render_mode)
        self.clip_reward_at_zero = clip_reward_at_zero

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)
        if self.clip_reward_at_zero:
            reward = torch.clamp(reward, min=0.0)
        return obs, reward, terminated, truncated, extras
