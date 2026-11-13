"""Simplified single-rigid-body dynamics linearized about the current yaw:

    m p_dd = sum_i f_i + m g
    d/dt (I w) = sum_i r_i x f_i

with friction-cone constraints on the stance-foot forces f_i. Solved as a QP each step.

TODO(phase 5)."""
