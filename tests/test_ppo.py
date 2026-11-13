"""PPO learns a toy task: a point mass must move to a random goal.

Runs in seconds on CPU with no simulator, so it isolates the PPO math (GAE, clipping,
normalization, adaptive learning rate) from the environment.
"""

import pytest

torch = pytest.importorskip("torch")

from quadruped_rl.algorithms.ppo.agent import PPOAgent  # noqa: E402


class ToyReachEnv:
    """2-D point mass. Observation: goal - position, velocity. Action: force."""

    def __init__(self, num_envs=256, max_steps=50):
        self.num_envs, self.num_actions, self.max_episode_length = num_envs, 2, max_steps
        self.dt = 0.1
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.pos = torch.zeros(num_envs, 2)
        self.vel = torch.zeros(num_envs, 2)
        self.goal = torch.zeros(num_envs, 2)

    def _obs(self):
        o = torch.cat([self.goal - self.pos, self.vel], dim=1)
        return {"actor": o, "critic": o}

    def _reset(self, ids):
        self.pos[ids] = 0.0
        self.vel[ids] = 0.0
        self.goal[ids] = torch.rand(len(ids), 2) * 2 - 1
        self.episode_length_buf[ids] = 0

    def reset(self):
        self._reset(torch.arange(self.num_envs))
        return self._obs(), {}

    def step(self, action):
        force = action.clamp(-1, 1)
        self.vel = 0.9 * self.vel + force * self.dt
        self.pos = self.pos + self.vel * self.dt
        dist = torch.norm(self.goal - self.pos, dim=1)
        reward = torch.exp(-4 * dist**2) - 0.01 * torch.sum(force**2, dim=1)
        self.episode_length_buf += 1
        truncated = self.episode_length_buf >= self.max_episode_length
        terminated = torch.zeros_like(truncated)
        ids = truncated.nonzero().squeeze(-1)
        if len(ids):
            self._reset(ids)
        return self._obs(), reward, terminated, truncated, {"log": {}}


ALGO = {"steps_per_env": 50, "learning_rate": 1e-3, "adaptive_lr_target_kl": 0.01,
        "num_epochs": 5, "num_minibatches": 4, "clip_ratio": 0.2, "gamma": 0.99,
        "gae_lambda": 0.95, "entropy_coef": 0.0, "value_loss_coef": 1.0, "max_grad_norm": 1.0,
        "init_noise_std": 0.5, "actor_hidden": [64, 64], "critic_hidden": [64, 64],
        "activation": "elu", "save_interval": 1000}


def test_ppo_learns_toy_reach(tmp_path):
    torch.manual_seed(0)
    env = ToyReachEnv()
    agent = PPOAgent(robot=None, task_cfg={}, algo_cfg=ALGO, run_dir=tmp_path)
    agent.learn(env, max_iterations=60)

    import csv
    rows = list(csv.DictReader(open(tmp_path / "progress.csv")))
    returns = [float(r["mean_return"]) for r in rows if r["mean_return"]]
    early, late = sum(returns[:5]) / 5, sum(returns[-5:]) / 5
    assert late > early + 5.0, f"return did not improve enough: {early:.2f} -> {late:.2f}"
    assert (tmp_path / "checkpoints" / "model_60.pt").exists()
