"""YAML config loading with single inheritance and command-line overrides.

A config may name a parent with ``inherit: <path relative to configs/>``. The child is
deep-merged on top of the parent, so ``rough_velocity.yaml`` lists only what differs
from ``flat_velocity.yaml``. Lists are replaced, not merged. A dict containing
``_replace: true`` replaces the parent's dict instead of merging into it.

Numbers are read the YAML 1.2 way: ``1e-3`` and ``-1e-3`` are floats (PyYAML's YAML 1.1
rules would read them as strings because they have no decimal point).
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from quadruped_rl.utils.paths import CONFIG_DIR


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads exponent floats without a decimal point (1e-3, -2E4)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def parse_yaml(text: str) -> Any:
    """Parse YAML text with this module's number rules."""
    return yaml.load(text, Loader=_Loader)


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
        cfg = parse_yaml(f.read()) or {}
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


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    """Set ``cfg["a"]["b"]["c"] = value`` for ``key="a.b.c"``, creating dicts on the way."""
    node = cfg
    *parents, leaf = key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML (numbers, lists, null)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        key, raw = item.split("=", 1)
        set_dotted(cfg, key, parse_yaml(raw))
    return cfg


def apply_dotted(cfg: dict, values: dict | None) -> dict:
    """Like apply_overrides, for a mapping of already-parsed values {"a.b.c": value}."""
    cfg = copy.deepcopy(cfg)
    for key, value in (values or {}).items():
        set_dotted(cfg, str(key), copy.deepcopy(value))
    return cfg


def load_preset(name: str) -> dict[str, Any]:
    """Load ``configs/experiments/<name>.yaml``: an experiment preset for scripts/train.py.

    Keys (all optional): ``task:`` {dotted key: value} overrides, applied before --set (so
    --set wins); ``task_name:``, ``robot:``, ``algo:`` the task, robot and algorithm used
    when --task, --robot, --algo are not given on the command line.
    """
    cfg = load_group("experiments", name)
    unknown = set(cfg) - {"task", "task_name", "robot", "algo"}
    if unknown:
        raise ValueError(f"Preset '{name}': unknown keys {sorted(unknown)}")
    if not isinstance(cfg.get("task") or {}, dict):
        raise TypeError(f"Preset '{name}': 'task' must map dotted keys to values")
    return cfg
