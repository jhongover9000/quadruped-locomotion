"""Learned dynamics model s_{t+1} = f_phi(s_t, a_t), fit to transitions logged
from the (non-differentiable) MuJoCo simulator. Being a neural network, it is
differentiable, which gives the policy analytic gradients.

TODO(phase 4)."""
