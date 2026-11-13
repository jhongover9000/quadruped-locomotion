"""Actor-critic networks for PPO.

Actor: a Gaussian policy. An MLP outputs the mean action mu(o); a learned,
state-independent standard deviation sigma sets how much the policy explores.

    a ~ N(mu(o), diag(sigma^2))

Critic: an MLP estimating the value V(s) (expected discounted return from state s). It
may see privileged observations the actor can't (asymmetric actor-critic).

Both normalize their inputs with running statistics, (x - mean) / (std + eps), so
joint velocities in rad/s and gravity components in [-1, 1] arrive on similar scales.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}


class EmpiricalNormalization(nn.Module):
    """Running mean and variance over every observation seen during rollouts."""

    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (torch.sqrt(self.var) + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Merge a batch [B, dim] into the running statistics (parallel Welford)."""
        batch_count = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total


def mlp(in_dim: int, hidden: list[int], out_dim: int, activation: str) -> nn.Sequential:
    layers, dim = [], in_dim
    for h in hidden:
        layers += [nn.Linear(dim, h), ACTIVATIONS[activation]()]
        dim = h
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, actor_dim: int, critic_dim: int, action_dim: int, actor_hidden: list[int],
                 critic_hidden: list[int], activation: str = "elu", init_noise_std: float = 1.0):
        super().__init__()
        self.actor = mlp(actor_dim, actor_hidden, action_dim, activation)
        self.critic = mlp(critic_dim, critic_hidden, 1, activation)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_noise_std)).log())
        self.actor_norm = EmpiricalNormalization(actor_dim)
        self.critic_norm = EmpiricalNormalization(critic_dim)

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.exp()

    def distribution(self, actor_obs: torch.Tensor) -> Normal:
        mean = self.actor(self.actor_norm(actor_obs))
        return Normal(mean, self.std.expand_as(mean))

    def act(self, actor_obs: torch.Tensor):
        """Sample an exploratory action; also return what PPO needs to store."""
        dist = self.distribution(actor_obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.mean, dist.stddev

    def act_inference(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (the mean) for evaluation and deployment."""
        return self.actor(self.actor_norm(actor_obs))

    def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.critic_norm(critic_obs)).squeeze(-1)

    def update_normalization(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> None:
        self.actor_norm.update(actor_obs)
        self.critic_norm.update(critic_obs)
