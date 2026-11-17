"""Common interface every method implements, so training, evaluation, the viewer and
the recorder treat them identically.

What the rest of the repository relies on (tests/test_compatibility.py checks it):

* ``implemented``: False for methods that are still stubs; scripts/train.py refuses
  them before creating a run directory.
* ``task_kinds``: which kinds of task the method can train on ("velocity": follow a
  commanded velocity; "tracking": follow reference motion clips). A task YAML names its
  kind with ``kind:``.
* ``train()`` writes TensorBoard events and checkpoints somewhere under ``run_dir``;
  ``checkpoint_patterns`` says where the checkpoints are, so the latest can be found.
* ``make_env()`` builds the environment the method trains or plays in.
* ``make_policy(env, checkpoint)`` returns ``(viewer_env, policy)`` with
  ``policy(viewer_env.get_observations()) -> actions``, deterministic.
"""

from __future__ import annotations

from pathlib import Path


class Agent:
    implemented: bool = True
    task_kinds: tuple[str, ...] = ("velocity",)
    # Globs relative to the run directory; the number in the file name orders them.
    checkpoint_patterns: tuple[str, ...] = ("checkpoints/model_*.pt", "model_*.pt")

    def __init__(self, robot, task_cfg: dict, algo_cfg: dict, run_dir: Path,
                 device: str = "cpu", seed: int = 0):
        self.robot = robot
        self.task_cfg = task_cfg
        self.algo_cfg = algo_cfg
        self.run_dir = Path(run_dir)
        self.device = device
        self.seed = seed

    @classmethod
    def resume_checkpoint(cls, run_dir: Path) -> tuple[Path, int] | None:
        """The checkpoint ``--continue-run`` resumes from and its iteration (None: none yet)."""
        from quadruped_rl.algorithms.checkpoints import checkpoint_number, list_checkpoints

        ckpts = list_checkpoints(run_dir, cls.checkpoint_patterns)
        return (ckpts[-1], checkpoint_number(ckpts[-1])) if ckpts else None

    def make_env(self, num_envs: int, play: bool = False, seed: int | None = None):
        """The environment for this method's task (velocity tasks: envs/builder.py)."""
        from quadruped_rl.envs.builder import make_env
        from quadruped_rl.utils import watchdog

        return watchdog.watch_env(make_env(self.robot, self.task_cfg, num_envs, self.device,
                                           play=play, seed=self.seed if seed is None else seed))

    def train(self, num_envs: int | None = None, max_iterations: int | None = None,
              resume: Path | None = None) -> None:
        """Run training, writing checkpoints and logs into ``run_dir``."""
        raise NotImplementedError(f"{type(self).__name__}.train() is not implemented yet")

    def make_policy(self, env, checkpoint: Path):
        """Load ``checkpoint`` for ``env``. Returns (env_for_viewer, policy), where
        policy maps env.get_observations() to deterministic actions."""
        raise NotImplementedError(f"{type(self).__name__}.make_policy() is not implemented yet")


class NotImplementedAgent(Agent):
    """Base for methods that are planned but not built: every entry point says so."""

    implemented = False
    phase = "a later phase"

    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError(f"{type(self).__name__} is not implemented yet ({self.phase})")

    def make_policy(self, env, checkpoint: Path):
        raise NotImplementedError(f"{type(self).__name__} is not implemented yet ({self.phase})")


def task_kind(task_cfg: dict) -> str:
    """A task's kind; configs saved before kinds existed were all velocity tasks."""
    return task_cfg.get("kind", "velocity")
