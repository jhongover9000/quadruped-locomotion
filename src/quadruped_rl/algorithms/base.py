"""Common interface every method implements, so training and evaluation scripts
treat them identically."""

from __future__ import annotations

from pathlib import Path


class Agent:
    def __init__(self, robot, task_cfg: dict, algo_cfg: dict, run_dir: Path):
        self.robot = robot
        self.task_cfg = task_cfg
        self.algo_cfg = algo_cfg
        self.run_dir = Path(run_dir)

    def train(self) -> None:
        """Run training, writing checkpoints and logs into ``run_dir``."""
        raise NotImplementedError(f"{type(self).__name__}.train() is not implemented yet")

    def act(self, obs):
        """Deterministic action for evaluation and deployment."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError
