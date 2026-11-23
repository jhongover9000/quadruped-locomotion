"""Terrain: a flat plane, or a procedurally generated grid with a difficulty curriculum.

Curriculum grid: rows are difficulty levels, columns are sub-terrain types in the
configured proportions. Robots that walk past half a tile in an episode move to a
harder row; robots that cover less than half their commanded distance move down.
With ``curriculum.promote_requires_timeout`` a robot also has to finish its episode
without a fall to move up (``terrain_levels_vel``).

Play mode (play.py, record.py) builds a 5 x 5 grid of randomly chosen sub-terrains with
difficulties drawn from ``terrain.play_difficulty_range``; ``spawn_tiles`` says which
tile each robot started on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ALL_TERRAIN_PRESETS
from mjlab.terrains.heightfield_terrains import HfPyramidSlopedTerrainCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg


def _tuples(value):
    """YAML gives lists; mjlab range parameters expect tuples."""
    return tuple(value) if isinstance(value, list) else value


def _sub_terrain(name: str, spec):
    if name not in ALL_TERRAIN_PRESETS:
        raise KeyError(f"Unknown sub-terrain '{name}'. Available: {sorted(ALL_TERRAIN_PRESETS)}")
    params = {"proportion": float(spec)} if isinstance(spec, (int, float)) else dict(spec)
    return ALL_TERRAIN_PRESETS[name](**{k: _tuples(v) for k, v in params.items()})


@dataclass(kw_only=True)
class _LoggedSubTerrain(SubTerrainCfg):
    """A sub-terrain that records (name, difficulty) in ``log`` for every tile it builds.

    Play mode only: mjlab's random grid does not keep which type and difficulty it drew
    for each tile. All sub-terrains of one grid share one ``log``; mjlab 1.6 builds the
    random grid in row-major order (TerrainGenerator._generate_random_terrains), so entry
    ``row * num_cols + col`` describes tile (row, col).
    """

    inner: SubTerrainCfg
    name: str
    log: list = field(default_factory=list)

    def function(self, difficulty, spec, rng):
        self.inner.size = self.size                 # the generator sets size on the wrapper
        self.log.append((self.name, float(difficulty)))
        return self.inner.function(difficulty, spec, rng)


def _logged(sub_terrains: dict[str, SubTerrainCfg]) -> dict[str, SubTerrainCfg]:
    log: list = []
    return {name: _LoggedSubTerrain(inner=cfg, name=name, log=log, proportion=cfg.proportion,
                                    flat_patch_sampling=cfg.flat_patch_sampling)
            for name, cfg in sub_terrains.items()}


def play_difficulty_range(terrain_cfg: dict) -> tuple[float, float]:
    lo, hi = (float(v) for v in terrain_cfg.get("play_difficulty_range", (0.0, 1.0)))
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(f"terrain.play_difficulty_range must satisfy 0 <= lo <= hi <= 1, got {[lo, hi]}")
    return lo, hi


def make_terrain(terrain_cfg: dict, play: bool = False) -> TerrainEntityCfg:
    kind = terrain_cfg.get("type", "flat")
    if kind == "flat":
        return TerrainEntityCfg(terrain_type="plane")
    if kind != "curriculum":
        raise ValueError(f"Unknown terrain type '{kind}' (expected flat or curriculum)")

    sub_terrains = {n: _sub_terrain(n, s) for n, s in terrain_cfg["sub_terrains"].items()}
    generator = TerrainGeneratorCfg(
        size=tuple(terrain_cfg.get("tile_size", (8.0, 8.0))),
        border_width=20.0,
        num_rows=5 if play else int(terrain_cfg.get("levels", 10)),
        num_cols=5 if play else int(terrain_cfg.get("tiles_per_level", 20)),
        sub_terrains=_logged(sub_terrains) if play else sub_terrains,
        curriculum=not play,
        difficulty_range=play_difficulty_range(terrain_cfg) if play else (0.0, 1.0),
        add_lights=True,
    )
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=int(terrain_cfg.get("max_init_level", 5)),
    )


def uses_generator(terrain_cfg: dict) -> bool:
    return terrain_cfg.get("type", "flat") == "curriculum"


# ---- curriculum -----------------------------------------------------------------------

def terrain_levels_vel(env, env_ids: torch.Tensor, command_name: str,
                       promote_requires_timeout: bool = False) -> dict[str, torch.Tensor]:
    """mjlab's terrain_levels_vel (tasks/velocity/mdp/curriculums.py), which promotes on
    distance alone: a robot that slides or tumbles more than half a tile before its
    episode ends in a fall still moves up. With ``promote_requires_timeout`` it moves up
    only if its episode also ended by a time-out (time_out or out_of_terrain_bounds) and
    not by a termination (base_contact, bad_orientation). Demotion is unchanged.

    Runs inside env._reset_idx, so the termination manager still holds the flags of the
    step that ended these episodes.
    """
    if not promote_requires_timeout:
        return velocity_mdp.terrain_levels_vel(env, env_ids, command_name)

    asset = env.scene["robot"]
    terrain = env.scene.terrain
    terrain_generator = terrain.cfg.terrain_generator
    command = env.command_manager.get_command(command_name)

    distance = torch.norm(asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
                          dim=1)
    timed_out = (env.termination_manager.time_outs[env_ids]
                 & ~env.termination_manager.terminated[env_ids])
    move_up = (distance > terrain_generator.size[0] / 2) & timed_out
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down = move_down & ~(distance > terrain_generator.size[0] / 2)
    if env.common_step_counter == 0:           # first reset: robots are still at their spawn pose
        move_up = torch.zeros_like(move_up)
        move_down = torch.zeros_like(move_down)
    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    result = {"mean": torch.mean(levels), "max": torch.max(levels)}
    names = list(terrain_generator.sub_terrains.keys())
    if terrain.terrain_origins.shape[1] == len(names):
        for i, name in enumerate(names):
            mask = terrain.terrain_types == i
            if mask.any():
                result[name] = torch.mean(levels[mask])
    return result


# ---- which tile each robot is on ------------------------------------------------------

def _slope_deg(cfg: SubTerrainCfg, difficulty: float) -> float | None:
    """Nominal slope of an hf_pyramid_slope(_inv) tile: mjlab interpolates slope_range
    (rise / run) linearly in difficulty (heightfield_terrains.py, HfPyramidSlopedTerrainCfg)."""
    if not isinstance(cfg, HfPyramidSlopedTerrainCfg):
        return None
    lo, hi = cfg.slope_range
    return math.degrees(math.atan(lo + difficulty * (hi - lo)))


def spawn_tiles(env) -> list[dict] | None:
    """For each robot: grid row and column of the tile it spawned on, the sub-terrain type,
    its difficulty, and for slope tiles the nominal slope in degrees (None otherwise).
    None for a flat plane. Read before any curriculum update moves robots to other tiles."""
    terrain = env.scene.terrain
    gen = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if gen is None or getattr(terrain, "terrain_origins", None) is None:
        return None
    rows, cols = terrain.terrain_origins.shape[:2]
    names = list(gen.sub_terrains)
    if gen.curriculum:                          # one column per type, difficulty along rows
        lo, hi = gen.difficulty_range
        layout = [[(names[c], lo + (hi - lo) * r / max(rows - 1, 1)) for c in range(cols)]
                  for r in range(rows)]
    else:
        log = getattr(next(iter(gen.sub_terrains.values())), "log", None)
        if log is None or len(log) != rows * cols:
            return None                         # grid not built through make_terrain(play=True)
        layout = [[log[r * cols + c] for c in range(cols)] for r in range(rows)]
    tiles = []
    for i, (r, c) in enumerate(zip(terrain.terrain_levels.tolist(), terrain.terrain_types.tolist())):
        name, difficulty = layout[r][c]
        cfg = getattr(gen.sub_terrains[name], "inner", gen.sub_terrains[name])
        slope = _slope_deg(cfg, difficulty)
        tiles.append({"robot": i, "row": int(r), "col": int(c), "type": name,
                      "difficulty": round(float(difficulty), 4),
                      "slope_deg": None if slope is None else round(slope, 2)})
    return tiles
