"""AMP discriminator D(s, s') scoring whether a window of states looks like the reference
clips (Peng et al., 2021). Inputs are normalized with running statistics.

Least-squares GAN loss, with targets +1 for clip windows and -1 for policy windows, and a
gradient penalty on clip windows that keeps D smooth around the reference data:

    L_D = E_clip[(D - 1)^2] + E_policy[(D + 1)^2] + (w_gp / 2) E_clip[ |grad_x D|^2 ]

(x is the normalized input). Style reward for a policy window:

    r_style = max(0, 1 - 0.25 (D - 1)^2)        1 at D = 1 (looks like a clip), 0 at D <= -1
"""

from __future__ import annotations

import torch
from torch import nn

from quadruped_rl.algorithms.ppo.networks import EmpiricalNormalization, mlp


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], activation: str = "relu"):
        super().__init__()
        self.norm = EmpiricalNormalization(input_dim)
        self.net = mlp(input_dim, hidden, 1, activation)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Raw windows [B, input_dim] -> scores [B]."""
        return self.net(self.norm(windows)).squeeze(-1)

    @torch.no_grad()
    def style_reward(self, windows: torch.Tensor) -> torch.Tensor:
        d = self(windows)
        return torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)

    def loss(self, clip: torch.Tensor, policy: torch.Tensor,
             grad_penalty: float) -> tuple[torch.Tensor, dict[str, float]]:
        """L_D on a batch of clip and policy windows; also the terms, for logging."""
        x = self.norm(clip).requires_grad_(True)
        d_clip = self.net(x).squeeze(-1)
        d_policy = self(policy)
        grad = torch.autograd.grad(d_clip.sum(), x, create_graph=True)[0]
        penalty = grad.square().sum(-1).mean()
        lsgan = (d_clip - 1.0).square().mean() + (d_policy + 1.0).square().mean()
        loss = lsgan + 0.5 * grad_penalty * penalty
        return loss, {"lsgan": lsgan.item(), "grad_penalty": penalty.item(),
                      "d_clip": d_clip.mean().item(), "d_policy": d_policy.mean().item()}
