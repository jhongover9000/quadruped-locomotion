"""Checkpoint files: which trainer wrote one, and where a run keeps them.

Each trainer has its own file layout; loading one trainer's file with another used to
fail with a bare KeyError. ``require_format`` turns that into a message naming both.
"""

from __future__ import annotations

import re
from pathlib import Path

# Keys that identify each trainer's checkpoint dict.
_SIGNATURES = {
    "ppo": ("model", "optimizer", "learning_rate"),
    "ppo_rslrl": ("actor_state_dict",),
    "supermimic": ("wm", "policy", "critic"),
}
ALL_PATTERNS = ("checkpoints/model_*.pt", "model_*.pt", "ckpt_[0-9]*.pt")


def load_checkpoint(path: Path, device: str = "cpu") -> dict:
    """torch.load that also reads checkpoints carrying plain Python config objects."""
    import pickle

    import torch

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        # The SuperMimic trainer stores its config (with pathlib paths) in the file.
        return torch.load(path, map_location=device, weights_only=False)


def checkpoint_format(ckpt: dict) -> str:
    for name, keys in _SIGNATURES.items():
        if all(k in ckpt for k in keys):
            return name
    return "unknown"


def require_format(ckpt: dict, expected: str, path: Path) -> None:
    found = checkpoint_format(ckpt)
    if found != expected:
        hint = (f"it was written by '{found}'; load it with --algo {found}" if found != "unknown"
                else f"its keys are {sorted(ckpt)[:6]}")
        raise ValueError(f"{path} is not a '{expected}' checkpoint: {hint}. Checkpoints do not "
                         "transfer between trainers (different networks and normalization).")


def checkpoint_number(path: Path) -> int:
    return int(re.search(r"(\d+)\.pt$", path.name).group(1))


def list_checkpoints(run_dir: Path, patterns: tuple[str, ...] = ALL_PATTERNS) -> list[Path]:
    """A run's numbered checkpoints, oldest first."""
    found = {p for pattern in patterns for p in Path(run_dir).glob(pattern)}
    return sorted(found, key=checkpoint_number)
