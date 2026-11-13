"""Short imagined rollouts in the world model and backpropagation through time:
the policy loss is the negative tracking return over H imagined steps plus a terminal
critic value, differentiated end to end through f_phi.

TODO(phase 4)."""
