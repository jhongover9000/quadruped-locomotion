"""Rollout buffer and GAE advantage estimation.

    delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
    A_t     = sum_l (gamma lambda)^l delta_{t+l}

TODO(phase 1)."""
