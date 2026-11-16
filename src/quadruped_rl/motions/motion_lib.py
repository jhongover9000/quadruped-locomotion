"""A clip library on disk: the manifest, its groups, and summaries.

    lib = MotionLibrary("…/data/npz/go1")
    lib.groups()                      # {"train": 275, "hard": 179, "holdout": 66}
    lib.names("holdout")              # clip names to give scripts/record.py --clips
    clip = lib.load("ai4a_D1_004_KAN01_001_s0")

The library is built by the SuperMimic reference repository's data scripts (retargeting,
checks, feasibility screening, group assignment; docs/supermimic.md) and is read during
training by its GPU motion library. This class is the numpy-only view used for
inspection, tests and choosing clips; scripts/motions.py prints it.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from quadruped_rl.motions.format import Clip, load_clip

KEEP = ("1", "True", "true")


class MotionLibrary:
    def __init__(self, directory: str | Path, manifest: str | Path | None = None):
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"No motion library at {self.directory}")
        self.manifest = Path(manifest) if manifest else self.directory / "manifest.csv"
        self.rows: list[dict] = []
        if self.manifest.exists():
            with open(self.manifest, newline="", encoding="utf-8") as f:
                self.rows = list(csv.DictReader(f))
        else:                                   # no manifest: every clip, one group
            self.rows = [{"name": p.stem, "keep": "1", "group": "all"}
                         for p in sorted(self.directory.glob("*.npz"))]

    def kept(self) -> list[dict]:
        return [r for r in self.rows if r.get("keep", "1") in KEEP]

    def groups(self) -> dict[str, int]:
        return dict(Counter(r.get("group") or "ungrouped" for r in self.kept()))

    def names(self, groups: str | None = None, exclude: str | None = "_mirror$") -> list[str]:
        """Kept clips, optionally of some groups ("train,hard"), without mirrored copies."""
        wanted = set(groups.split(",")) if groups else None
        rx = re.compile(exclude) if exclude else None
        return [r["name"] for r in self.kept()
                if (wanted is None or r.get("group") in wanted)
                and not (rx and rx.search(r["name"]))
                and (self.directory / f"{r['name']}.npz").exists()]

    def load(self, name: str) -> Clip:
        return load_clip(self.directory / f"{name}.npz")

    def summary(self, root_body: str, groups: str | None = None, limit: int | None = None) -> list[dict]:
        """Per clip: frames, seconds, mean horizontal speed (loads the files: slow for all)."""
        out = []
        for name in self.names(groups)[:limit]:
            c = self.load(name)
            out.append({"name": name, "frames": c.frames, "seconds": round(c.duration_s, 2),
                        "mean_speed_mps": round(c.mean_speed(root_body), 3)})
        return out
