"""DeepMimic agent: PPO optimizing the imitation reward of a tracking task (reference state
initialization, early termination), i.e. the model-free counterpart of SuperMimic on the
same clips.

Not built yet. The natural route is PPO on the SuperMimic backend's tracking environment
(algorithms/supermimic/), so both methods see identical clips, rewards and terminations.

TODO(phase 4)."""

from quadruped_rl.algorithms.base import NotImplementedAgent


class DeepMimicAgent(NotImplementedAgent):
    task_kinds = ("tracking",)
    phase = "phase 4"
