"""DeepMimic agent. The optimizer is PPO; the difference is the task
(tasks/motion_tracking.yaml): imitation reward, reference state initialization, and
early termination. Kept as its own class so it can diverge later.

TODO(phase 4)."""

from quadruped_rl.algorithms.ppo.agent import PPOAgent


class DeepMimicAgent(PPOAgent):
    pass
