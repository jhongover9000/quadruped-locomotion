"""Common interface every method implements, so training, evaluation, and the viewer
treat them identically."""

from __future__ import annotations

from pathlib import Path


class Agent:
    def __init__(self, robot, task_cfg: dict, algo_cfg: dict, run_dir: Path,
                 device: str = "cpu", seed: int = 0):
        self.robot = robot
        self.task_cfg = task_cfg
        self.algo_cfg = algo_cfg
        self.run_dir = Path(run_dir)
        self.device = device
        self.seed = seed

    def train(self, num_envs: int | None = None, max_iterations: int | None = None,
              resume: Path | None = None) -> None:
        """Run training, writing checkpoints and logs into ``run_dir``."""
        raise NotImplementedError(f"{type(self).__name__}.train() is not implemented yet")

    def make_policy(self, env, checkpoint: Path):
        """Load ``checkpoint`` for ``env``. Returns (env_for_viewer, policy), where
        policy maps env.get_observations() to deterministic actions."""
        raise NotImplementedError(f"{type(self).__name__}.make_policy() is not implemented yet")
