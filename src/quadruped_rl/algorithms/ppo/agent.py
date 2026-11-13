"""PPO agent: clipped surrogate objective with GAE advantages.

    L(theta) = E[ min( rho_t A_t, clip(rho_t, 1 - eps, 1 + eps) A_t ) ]
    rho_t = pi_theta(a_t | o_t) / pi_theta_old(a_t | o_t)

Asymmetric actor-critic: the actor gets the 'policy' observations, the critic the
privileged 'critic' observations.

TODO(phase 1)."""

from quadruped_rl.algorithms.base import Agent


class PPOAgent(Agent):
    pass
