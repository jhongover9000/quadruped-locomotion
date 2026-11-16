"""AMP agent: PPO whose reward mixes the task reward with the discriminator's style
reward, r = w_task r_task + w_style r_style. Trains the discriminator alongside the policy.

TODO(phase 4)."""

from quadruped_rl.algorithms.base import NotImplementedAgent


class AMPAgent(NotImplementedAgent):
    task_kinds = ("velocity",)
    phase = "phase 4"
