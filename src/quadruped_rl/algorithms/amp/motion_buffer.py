"""The two sources of discriminator samples: windows of consecutive reference-clip frames
(``MotionBuffer``) and windows of consecutive policy states (``ReplayBuffer``).

A window is ``num_frames`` consecutive states' features (features.py) concatenated,
[num_frames * D]; with 2 frames it is the transition (s, s'). Clips are at the control
rate, so a clip window and a policy window span the same time.
"""

from __future__ import annotations

import torch


class MotionBuffer:
    """Every frame of the style clips as features, sampled as windows that stay inside one
    clip. A clip is drawn with probability proportional to duration^alpha (alpha = 1:
    uniform over frames; 0: uniform over clips), then a window uniformly within it."""

    def __init__(self, clip_features: list[torch.Tensor], num_frames: int, device: str = "cpu",
                 clip_weight_alpha: float = 0.5):
        self.num_frames = int(num_frames)
        kept = [f for f in clip_features if f.shape[0] >= self.num_frames]
        if not kept:
            raise ValueError(f"No clip has the {self.num_frames} frames a window needs")
        starts, weights, offset = [], [], 0
        for f in kept:
            n = f.shape[0] - self.num_frames + 1
            starts.append(torch.arange(n) + offset)
            weights.append(torch.full((n,), float(f.shape[0]) ** clip_weight_alpha / n))
            offset += f.shape[0]
        self.features = torch.cat(kept).to(device)                 # [frames, D]
        self.starts = torch.cat(starts).to(device)
        self.weights = torch.cat(weights).double().to(device)
        self.weights /= self.weights.sum()
        self.num_clips = len(kept)

    @property
    def dim(self) -> int:
        return self.num_frames * self.features.shape[1]

    def sample(self, n: int) -> torch.Tensor:
        """``n`` windows, [n, num_frames * D]."""
        idx = self.starts[torch.multinomial(self.weights, n, replacement=True)]
        return torch.cat([self.features[idx + k] for k in range(self.num_frames)], dim=-1)


class ReplayBuffer:
    """The most recent ``capacity`` policy windows (first in, first out), sampled uniformly."""

    def __init__(self, capacity: int, dim: int, device: str = "cpu"):
        self.data = torch.zeros(int(capacity), dim, device=device)
        self.capacity, self.size, self.next = int(capacity), 0, 0

    def __len__(self) -> int:
        return self.size

    def insert(self, windows: torch.Tensor) -> None:
        windows = windows[-self.capacity:]
        n = windows.shape[0]
        idx = (torch.arange(n, device=self.data.device) + self.next) % self.capacity
        self.data[idx] = windows
        self.next = (self.next + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, n: int) -> torch.Tensor:
        if self.size == 0:
            raise RuntimeError("The replay buffer is empty")
        return self.data[torch.randint(self.size, (n,), device=self.data.device)]
