"""Terrain: a flat plane, or a procedurally generated grid with a difficulty curriculum.

Curriculum grid: rows are difficulty levels, columns are sub-terrain types in the
configured proportions. Robots that walk past half a tile in an episode move to a
harder row; robots that cover less than half their commanded distance move down.
"""

from __future__ import annotations

from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ALL_TERRAIN_PRESETS
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg


def _tuples(value):
    """YAML gives lists; mjlab range parameters expect tuples."""
    return tuple(value) if isinstance(value, list) else value


def _sub_terrain(name: str, spec):
    if name not in ALL_TERRAIN_PRESETS:
        raise KeyError(f"Unknown sub-terrain '{name}'. Available: {sorted(ALL_TERRAIN_PRESETS)}")
    params = {"proportion": float(spec)} if isinstance(spec, (int, float)) else dict(spec)
    return ALL_TERRAIN_PRESETS[name](**{k: _tuples(v) for k, v in params.items()})


def make_terrain(terrain_cfg: dict, play: bool = False) -> TerrainEntityCfg:
    kind = terrain_cfg.get("type", "flat")
    if kind == "flat":
        return TerrainEntityCfg(terrain_type="plane")
    if kind != "curriculum":
        raise ValueError(f"Unknown terrain type '{kind}' (expected flat or curriculum)")

    generator = TerrainGeneratorCfg(
        size=tuple(terrain_cfg.get("tile_size", (8.0, 8.0))),
        border_width=20.0,
        num_rows=5 if play else int(terrain_cfg.get("levels", 10)),
        num_cols=5 if play else int(terrain_cfg.get("tiles_per_level", 20)),
        sub_terrains={n: _sub_terrain(n, s) for n, s in terrain_cfg["sub_terrains"].items()},
        curriculum=not play,
        add_lights=True,
    )
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=int(terrain_cfg.get("max_init_level", 5)),
    )


def uses_generator(terrain_cfg: dict) -> bool:
    return terrain_cfg.get("type", "flat") == "curriculum"
