"""YAML config loading with single inheritance and command-line overrides.

A config may name a parent with ``inherit: <path relative to configs/>``. The child is
deep-merged on top of the parent, so ``rough_velocity.yaml`` lists only what differs
from ``flat_velocity.yaml``. Lists are replaced, not merged. A dict containing
``_replace: true`` replaces the parent's dict instead of merging into it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

from quadruped_rl.utils.paths import CONFIG_DIR


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with ``override`` merged recursively onto ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and value.get("_replace"):
            out[key] = {k: copy.deepcopy(v) for k, v in value.items() if k != "_replace"}
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML config, resolving ``inherit:`` chains."""
    path = Path(path)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    path = path.resolve()
    if path in _seen:
        chain = " -> ".join(str(p) for p in (*_seen, path))
        raise ValueError(f"Circular inherit: {chain}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent is None:
        return cfg
    return deep_merge(load_config(parent, (*_seen, path)), cfg)


def load_group(group: str, name: str) -> dict[str, Any]:
    """Load ``configs/<group>/<name>.yaml``, e.g. ``load_group("tasks", "flat_velocity")``."""
    if name.startswith("_"):
        raise ValueError(f"'{name}' is a template; copy it to a new file first.")
    return load_config(Path(group) / f"{name}.yaml")


def available(group: str) -> list[str]:
    """Names of the loadable configs in a group (templates excluded)."""
    return sorted(p.stem for p in (CONFIG_DIR / group).glob("*.yaml") if not p.stem.startswith("_"))


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML (numbers, lists, null)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        key, raw = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg
