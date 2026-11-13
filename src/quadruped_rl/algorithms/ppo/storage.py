"""Rollout buffer and Generalized Advantage Estimation (GAE).

Each iteration every environment runs T steps; the buffer holds those T x N
transitions. Advantages measure how much better an action did than the critic
expected:

    delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)       one-step surprise
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}       smoothed over time
    R_t     = A_t + V(s_t)                                            critic target

lambda trades bias (lambda = 0: trust the critic) against variance (lambda = 1:
trust the actual returns).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Batch:
    actor_obs: torch.Tensor
    critic_obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    old_mean: torch.Tensor
    old_std: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutStorage:
    def __init__(self, num_envs: int, num_steps: int, actor_dim: int, critic_dim: int,
                 action_dim: int, device: str):
        def buf(*shape):
            return torch.zeros(num_steps, num_envs, *shape, device=device)

        self.num_envs, self.num_steps = num_envs, num_steps
        self.actor_obs = buf(actor_dim)
        self.critic_obs = buf(critic_dim)
        self.actions = buf(action_dim)
        self.action_mean = buf(action_dim)
        self.action_std = buf(action_dim)
        self.log_probs = buf()
        self.values = buf()
        self.rewards = buf()
        self.dones = buf()
        self.advantages = buf()
        self.returns = buf()
        self.step = 0

    def add(self, actor_obs, critic_obs, actions, log_probs, values, rewards, dones, mean, std):
        t = self.step
        self.actor_obs[t].copy_(actor_obs)
        self.critic_obs[t].copy_(critic_obs)
        self.actions[t].copy_(actions)
        self.log_probs[t].copy_(log_probs)
        self.values[t].copy_(values)
        self.rewards[t].copy_(rewards)
        self.dones[t].copy_(dones)
        self.action_mean[t].copy_(mean)
        self.action_std[t].copy_(std)
        self.step += 1

    def clear(self) -> None:
        self.step = 0

    @torch.no_grad()
    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        advantage = torch.zeros_like(last_values)
        for t in reversed(range(self.num_steps)):
            next_values = last_values if t == self.num_steps - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * not_done * next_values - self.values[t]
            advantage = delta + gamma * lam * not_done * advantage
            self.returns[t] = advantage + self.values[t]
        self.advantages = self.returns - self.values
        # Normalize over the whole buffer: keeps the policy step size independent of reward scale.
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batches(self, num_minibatches: int, num_epochs: int):
        """Shuffle all T x N transitions each epoch and yield equal-sized minibatches."""
        total = self.num_steps * self.num_envs
        size = total // num_minibatches

        def flat(x):
            return x.flatten(0, 1)

        data = [flat(x) for x in (self.actor_obs, self.critic_obs, self.actions, self.log_probs,
                                  self.values, self.action_mean, self.action_std, self.advantages,
                                  self.returns)]
        for _ in range(num_epochs):
            perm = torch.randperm(total, device=self.actions.device)
            for i in range(num_minibatches):
                idx = perm[i * size:(i + 1) * size]
                yield Batch(*(x[idx] for x in data))
