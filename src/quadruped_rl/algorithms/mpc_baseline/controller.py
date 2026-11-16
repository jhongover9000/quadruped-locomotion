"""Convex MPC controller exposing the Agent interface, so the evaluator can
run it like any learned policy. train() will be a no-op.

TODO(phase 5)."""

from quadruped_rl.algorithms.base import NotImplementedAgent


class ConvexMPC(NotImplementedAgent):
    task_kinds = ("velocity",)
    phase = "phase 5"
