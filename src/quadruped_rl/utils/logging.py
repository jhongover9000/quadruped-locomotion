"""Run logging: TensorBoard by default, Weights & Biases optionally.

Every algorithm logs the same scalar names (reward terms, episode length, terrain level,
learning rate, losses) so runs can be compared side by side.

TODO(phase 1): thin wrapper around torch.utils.tensorboard.SummaryWriter with an
optional wandb mirror (enabled by the ``logging`` extra and a config flag).
"""
