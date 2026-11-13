"""SuperMimic-style agent (Liao et al., CoRL 2026). Training loop:
  1. collect transitions in the simulator with the current policy
  2. update the world model on the replay buffer
  3. freeze it; update the policy by BPTT through short imagined rollouts
  4. repeat

TODO(phase 4)."""

from quadruped_rl.algorithms.base import Agent


class SuperMimicAgent(Agent):
    pass
