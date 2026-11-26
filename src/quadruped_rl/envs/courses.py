"""Evaluation courses: fixed terrain that every velocity-command policy runs unchanged.

Two courses, each described by a YAML file in ``configs/courses/``:

- **gauntlet**: a straight track along +x made of back-to-back sections (flat, ramp,
  stairs, curb, rough, blocks, wave), each with its own friction. One identical lane per
  robot, ``lane_width_m`` wide, side by side in y.
- **ladders**: one straight gauntlet track per challenge type, whose levels step through
  rising difficulty. The YAML gives each ladder a template of sections and a list of level
  values; ``ladder_course`` expands one ladder into an ordinary gauntlet course.
- **beacon**: a square field of rocky, hilly ground built from square tiles. Every tile is
  its own heightfield geom with its own friction; all tiles sample one global height grid
  and share their edge vertices, so the surface is continuous across tiles.

The geometry is plain numpy (``Box`` and ``HField`` primitives, the beacon height grid)
so it can be tested and previewed without a simulator. ``CourseSubTerrainCfg`` turns it
into MuJoCo geoms through mjlab's TerrainGenerator: the gauntlet is a 1 x N grid whose
tiles are the lanes (so mjlab puts robot i at the start of lane i), the beacon field a
1 x 1 grid holding all of its tiles.

Friction per region: MuJoCo takes contact parameters from the geom with the higher
``priority``. The robot's feet have priority 1 and its other geoms 0
(builder.make_robot_entity), so every course geom gets COURSE_PRIORITY = 2 and the
course's friction is the one in effect. With priority, condim, solref and solimp also
come from the course geom; they are copied from the feet's settings so a foot contact
differs from training only in its sliding friction.
"""

from __future__ import annotations

import ast
import copy
import functools
import math
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

COURSE_PRIORITY = 2              # above the robot's feet (1) and other geoms (0)
COURSE_CONDIM = 6                # as the feet (builder.make_robot_entity)
COURSE_SOLREF = (0.01, 1.0)      # as the robot's collision geoms
TORSIONAL_FRICTION = 5e-3        # as the feet
ROLLING_FRICTION = 5e-4          # as the feet
BASE_DEPTH_M = 0.5               # every geom reaches this far below the course's lowest point

SECTION_TYPES = ("flat", "ramp", "stairs", "curb", "rough", "blocks", "wave")
JUMP_TYPES = ("stairs", "curb", "blocks")   # vertical edges by design

# Directional gauntlet: the body yaw each section's ``mode`` asks for while the robot travels
# +x (left: -90 deg, so the robot's left side leads). A spin section turns on the spot by
# ``spin_deg`` and then goes on forward.
MODE_YAW_DEG = {"forward": 0.0, "backward": 180.0, "left": -90.0, "right": 90.0,
                "diag_fl": -45.0, "diag_fr": 45.0, "diag_bl": -135.0, "diag_br": 135.0,
                "spin": 0.0}


# ---- loading ------------------------------------------------------------------------

def load_course(name_or_path: str | Path) -> dict:
    """``configs/courses/<name>.yaml`` (or a path to a YAML file), checked."""
    from quadruped_rl.utils.config import load_config, load_group

    path = Path(name_or_path)
    course = load_config(path.resolve()) if path.suffix in (".yaml", ".yml") else \
        load_group("courses", str(name_or_path))
    kind = course.get("kind")
    if kind == "gauntlet":
        gauntlet_sections(course)                  # raises on a bad section
    elif kind == "ladders":
        for name in ladder_names(course):
            gauntlet_sections(ladder_course(course, name))
    elif kind == "beacon":
        _beacon_grid_shape(course)
    else:
        raise ValueError(f"Course '{name_or_path}': kind must be gauntlet, ladders or beacon, "
                         f"got {kind!r}")
    return course


# ---- primitives ---------------------------------------------------------------------

@dataclass
class Box:
    """Axis-aligned box from (x0, y0, z_bottom) to (x1, y1, z_top)."""

    x0: float
    x1: float
    y0: float
    y1: float
    z_top: float
    z_bottom: float
    friction: float
    section: int = -1


@dataclass
class HField:
    """Heightfield over [x0, x1] x [y0, y1]: ``heights[iy, ix]`` are absolute z values at
    evenly spaced nodes, row 0 at y0 and column 0 at x0 (MuJoCo's layout)."""

    x0: float
    x1: float
    y0: float
    y1: float
    heights: np.ndarray
    z_bottom: float
    friction: float
    section: int = -1


def _bilinear(h: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of grid ``h[iy, ix]`` at fractional indices u (x), v (y)."""
    ny, nx = h.shape
    i0 = np.clip(np.floor(u).astype(int), 0, nx - 2)
    j0 = np.clip(np.floor(v).astype(int), 0, ny - 2)
    fu, fv = u - i0, v - j0
    return ((1 - fu) * (1 - fv) * h[j0, i0] + fu * (1 - fv) * h[j0, i0 + 1]
            + (1 - fu) * fv * h[j0 + 1, i0] + fu * fv * h[j0 + 1, i0 + 1])


def primitive_top(p: Box | HField, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Top surface of one primitive at (x, y); NaN outside its footprint. Heightfields are
    interpolated bilinearly (MuJoCo triangulates the cells: same values on cell edges)."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    eps = 1e-9
    inside = (x >= p.x0 - eps) & (x <= p.x1 + eps) & (y >= p.y0 - eps) & (y <= p.y1 + eps)
    out = np.full(x.shape, np.nan)
    if not inside.any():
        return out
    if isinstance(p, Box):
        out[inside] = p.z_top
        return out
    ny, nx = p.heights.shape
    u = (x[inside] - p.x0) / (p.x1 - p.x0) * (nx - 1)
    v = (y[inside] - p.y0) / (p.y1 - p.y0) * (ny - 1)
    out[inside] = _bilinear(p.heights, np.clip(u, 0, nx - 1), np.clip(v, 0, ny - 1))
    return out


def surface_height(prims: list, x, y) -> np.ndarray:
    """Highest primitive top at (x, y), NaN where there is none."""
    tops = np.stack([primitive_top(p, x, y) for p in prims])
    with np.errstate(all="ignore"):
        out = np.nanmax(np.where(np.isnan(tops), -np.inf, tops), axis=0)
    return np.where(np.isinf(out), np.nan, out)


def primitive_bounds_z(prims: list) -> tuple[float, float]:
    lo = min(p.z_bottom for p in prims)
    hi = max(p.z_top if isinstance(p, Box) else float(p.heights.max()) for p in prims)
    return lo, hi


# ---- gauntlet -----------------------------------------------------------------------

@dataclass
class Section:
    index: int
    type: str
    label: str
    x0: float               # course coordinate of the section's start, m
    x1: float
    friction: float
    z_start: float          # nominal height where the section starts (after any jump at x0)
    z_end: float            # nominal height where it ends
    params: dict = field(default_factory=dict)

    @property
    def length(self) -> float:
        return self.x1 - self.x0

    @property
    def mode(self) -> str:
        return str(self.params.get("mode", "forward"))


_REQUIRED = {"flat": (), "ramp": ("angle_deg",), "stairs": ("n", "rise_m", "tread_m"),
             "curb": ("height_m",), "rough": ("amplitude_m",), "blocks": ("h_max_m", "block_m"),
             "wave": ("amplitude_m", "wavelength_m")}


def _direction(spec: dict, label: str) -> float:
    d = spec.get("direction", "up")
    if d not in ("up", "down"):
        raise ValueError(f"Section '{label}': direction must be up or down, got {d!r}")
    return 1.0 if d == "up" else -1.0


def gauntlet_sections(course: dict) -> list[Section]:
    """The sections in order, with their x range and nominal start and end heights.

    Jumps happen only at the start of a stairs or curb section: stairs rise (or drop) by
    ``rise_m`` at the start of each of their ``n`` treads, a curb by ``height_m`` at its
    start. A blocks section stands on the height it starts at; its blocks stick up 0 to
    ``h_max_m`` above it.
    """
    sections, x, z = [], 0.0, 0.0
    for i, spec in enumerate(course["sections"]):
        kind = spec.get("type")
        label = str(spec.get("label", f"section {i}"))
        if kind not in SECTION_TYPES:
            raise ValueError(f"Section {i} ('{label}'): unknown type {kind!r}; one of {SECTION_TYPES}")
        missing = [k for k in _REQUIRED[kind] if k not in spec]
        if missing:
            raise ValueError(f"Section {i} ('{label}', {kind}) needs {missing}")
        params = {k: v for k, v in spec.items() if k not in ("type", "label", "length_m", "friction")}
        if kind == "stairs":
            length = int(spec["n"]) * float(spec["tread_m"])
            if "length_m" in spec and abs(float(spec["length_m"]) - length) > 1e-6:
                raise ValueError(f"Section '{label}': stairs length_m must be n x tread_m = {length}")
        else:
            if "length_m" not in spec:
                raise ValueError(f"Section {i} ('{label}') needs length_m")
            length = float(spec["length_m"])
        if length <= 0:
            raise ValueError(f"Section '{label}': length must be positive")
        friction = float(spec.get("friction", course.get("default_friction", 1.0)))
        if friction <= 0:
            raise ValueError(f"Section '{label}': friction must be positive")
        mode = spec.get("mode", "forward")
        if mode not in MODE_YAW_DEG:
            raise ValueError(f"Section '{label}': mode must be one of {list(MODE_YAW_DEG)}, got {mode!r}")
        if mode == "spin" and "spin_deg" not in spec:
            raise ValueError(f"Section '{label}': a spin section needs spin_deg")
        if "speed_m_s" in spec and float(spec["speed_m_s"]) <= 0:
            raise ValueError(f"Section '{label}': speed_m_s must be positive")
        z0 = z
        if kind == "ramp":
            z1 = z0 + length * math.tan(math.radians(float(spec["angle_deg"])))
        elif kind == "stairs":
            z0 = z + _direction(spec, label) * float(spec["rise_m"])
            z1 = z + _direction(spec, label) * float(spec["rise_m"]) * int(spec["n"])
        elif kind == "curb":
            z0 = z1 = z + _direction(spec, label) * float(spec["height_m"])
        elif kind == "wave":
            cycles = length / float(spec["wavelength_m"])
            if abs(cycles - round(cycles)) > 1e-6:
                raise ValueError(f"Section '{label}': length_m must be a whole number of "
                                 f"wavelengths (got {cycles:.3f})")
            z1 = z0
        else:
            z1 = z0
        sections.append(Section(i, kind, label, x, x + length, friction, z0, z1, params))
        x, z = x + length, z1
    if not sections:
        raise ValueError("A gauntlet needs at least one section")
    return sections


def course_length(course: dict) -> float:
    return gauntlet_sections(course)[-1].x1


def is_directional(course: dict) -> bool:
    """A gauntlet whose sections set a ``mode`` or ``speed_m_s`` (gauntlet_directions)."""
    return any("mode" in s or "speed_m_s" in s for s in course.get("sections") or [])


_LEVEL = re.compile(r"^L(\d+) ")


def section_level(label: str) -> int | None:
    """k of a section labelled ``L<k> <part>`` (ladders, graded gauntlet), else None."""
    m = _LEVEL.match(str(label))
    return int(m.group(1)) if m else None


# ---- ladders ------------------------------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv}


def _arith(node) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _arith(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_arith(node.left), _arith(node.right))
    raise ValueError("only numbers, $v, + - * / and brackets are allowed")


def template_value(value, v: float):
    """A template field with the level value ``v`` filled in: a string that mentions ``$v``
    is arithmetic on it ("$v", "-$v", "$v * 2", "($v + 0.1) / 2"); anything else is kept."""
    if not isinstance(value, str) or "$v" not in value:
        return value
    try:
        return _arith(ast.parse(value.replace("$v", f"({float(v)!r})"), mode="eval").body)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Ladder template value {value!r}: {e}") from None


def ladder_names(course: dict) -> list[str]:
    ladders = course.get("ladders") or {}
    if not ladders:
        raise ValueError("A ladders course needs at least one ladder under `ladders`")
    return list(ladders)


def ladder_course(course: dict, name: str) -> dict:
    """One ladder of a ``kind: ladders`` course as a gauntlet course.

    The track is a ``start_m`` flat start, then for each level k (value v) the template's
    sections with ``$v`` filled in and labelled ``L<k> <label>``, each level followed by a
    ``rest_m`` flat rest (``L<k> rest``) unless every template section is itself flat (as
    in the friction ladder; a ladder can set ``rest_m`` to override), then a ``finish_m``
    flat finish. The other course keys (lane width, controller, ...) are kept, and
    ``course["ladder"]`` records the ladder's name, level values and unit.
    """
    ladders = course.get("ladders") or {}
    if name not in ladders:
        raise ValueError(f"No ladder '{name}'; the course has {list(ladders)}")
    lad = ladders[name]
    template = lad.get("template") or []
    levels = [float(v) for v in lad.get("levels") or []]
    if not template or not levels:
        raise ValueError(f"Ladder '{name}' needs a template and levels")
    all_flat = all(t.get("type") == "flat" for t in template)
    rest = float(lad.get("rest_m", 0.0 if all_flat else course.get("rest_m", 1.5)))
    sections = [{"type": "flat", "length_m": float(course.get("start_m", 3.0)), "label": "start"}]
    for k, v in enumerate(levels, 1):
        for t in template:
            sec = {key: template_value(val, v) for key, val in t.items() if key != "label"}
            sec["label"] = f"L{k} {t.get('label', t.get('type'))}"
            sections.append(sec)
        if rest > 0:
            sections.append({"type": "flat", "length_m": rest, "label": f"L{k} rest"})
    sections.append({"type": "flat", "length_m": float(course.get("finish_m", 3.0)),
                     "label": "finish"})
    out = {k: copy.deepcopy(v) for k, v in course.items()
           if k not in ("kind", "ladders", "start_m", "finish_m", "rest_m")}
    out.update(kind="gauntlet", sections=sections,
               ladder={"name": name, "levels": levels, "unit": str(lad.get("unit", ""))})
    return out


def nominal_height(sections: list[Section], x) -> np.ndarray:
    """Nominal centre-line height at course coordinate x (no rough noise, no blocks)."""
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    for s in sections:
        m = (x >= s.x0) & (x < s.x1) if s is not sections[-1] else (x >= s.x0) & (x <= s.x1)
        u = x[m] - s.x0
        if s.type == "ramp":
            out[m] = s.z_start + (s.z_end - s.z_start) * u / s.length
        elif s.type == "stairs":
            step = np.minimum(np.floor(u / float(s.params["tread_m"])), int(s.params["n"]) - 1)
            out[m] = s.z_start + (s.z_end - s.z_start) * step / max(int(s.params["n"]) - 1, 1) \
                if int(s.params["n"]) > 1 else s.z_start
        elif s.type == "wave":
            out[m] = s.z_start + float(s.params["amplitude_m"]) * np.sin(
                2 * np.pi * u / float(s.params["wavelength_m"]))
        else:
            out[m] = s.z_start
    return out


def _grid(a: float, b: float, step: float) -> np.ndarray:
    n = max(int(round((b - a) / step)), 1) + 1
    return np.linspace(a, b, n)


def lane_primitives(course: dict) -> list[Box | HField]:
    """One lane's geometry in lane coordinates: x along the course from 0, y across it
    from -width/2 to +width/2. Seeded, so every lane (and every build) is identical."""
    sections = gauntlet_sections(course)
    w = float(course.get("lane_width_m", 2.0))
    y0, y1 = -w / 2, w / 2
    res = float(course.get("hfield_resolution_m", 0.05))
    seed = int(course.get("seed", 0))
    lowest = min(min(s.z_start, s.z_end) for s in sections)
    for s in sections:
        if s.type == "wave":
            lowest = min(lowest, s.z_start - float(s.params["amplitude_m"]))
        if s.type == "rough":
            lowest = min(lowest, s.z_start - float(s.params["amplitude_m"]))
    bottom = lowest - BASE_DEPTH_M

    prims: list[Box | HField] = []
    for s in sections:
        rng = np.random.default_rng([seed, s.index])
        mu = s.friction
        if s.type in ("flat", "curb"):
            prims.append(Box(s.x0, s.x1, y0, y1, s.z_start, bottom, mu, s.index))
        elif s.type == "stairs":
            n, tread = int(s.params["n"]), float(s.params["tread_m"])
            rise = (s.z_end - s.z_start) / max(n - 1, 1) if n > 1 else 0.0
            for k in range(n):
                prims.append(Box(s.x0 + k * tread, s.x0 + (k + 1) * tread, y0, y1,
                                 s.z_start + k * rise, bottom, mu, s.index))
        elif s.type == "ramp":
            h = np.array([[s.z_start, s.z_end], [s.z_start, s.z_end]])
            prims.append(HField(s.x0, s.x1, y0, y1, h, bottom, mu, s.index))
        elif s.type == "wave":
            step = min(res, float(s.params["wavelength_m"]) / 16)
            xs = _grid(s.x0, s.x1, step)
            z = nominal_height([s], xs)
            z[-1] = s.z_end                               # sin(2 pi k) rounding
            prims.append(HField(s.x0, s.x1, y0, y1, np.stack([z, z]), bottom, mu, s.index))
        elif s.type == "rough":
            cell = float(s.params.get("cell_m", 0.1))
            a = float(s.params["amplitude_m"])
            xs, ys = _grid(s.x0, s.x1, cell), _grid(y0, y1, cell)
            noise = rng.uniform(-a, a, (len(ys), len(xs)))
            noise[:, 0] = noise[:, -1] = 0.0              # meet the neighbours exactly
            prims.append(HField(s.x0, s.x1, y0, y1, s.z_start + noise, bottom, mu, s.index))
        elif s.type == "blocks":
            b, hmax = float(s.params["block_m"]), float(s.params["h_max_m"])
            nx, ny = max(int(round(s.length / b)), 1), max(int(round(w / b)), 1)
            xs, ys = np.linspace(s.x0, s.x1, nx + 1), np.linspace(y0, y1, ny + 1)
            tops = s.z_start + rng.uniform(0.0, hmax, (ny, nx))
            for j in range(ny):
                for i in range(nx):
                    prims.append(Box(xs[i], xs[i + 1], ys[j], ys[j + 1], float(tops[j, i]),
                                     bottom, mu, s.index))
    return prims


# ---- beacon field -------------------------------------------------------------------

def _beacon_grid_shape(course: dict) -> tuple[int, int]:
    """(tiles per side, height samples per tile side - 1)."""
    size, tile = float(course["field_size_m"]), float(course["tile_size_m"])
    res = float(course.get("hfield_resolution_m", 0.1))
    tiles, per_tile = size / tile, tile / res
    if abs(tiles - round(tiles)) > 1e-6 or abs(per_tile - round(per_tile)) > 1e-6:
        raise ValueError("beacon: field_size_m / tile_size_m and tile_size_m / "
                         "hfield_resolution_m must be whole numbers")
    return int(round(tiles)), int(round(per_tile))


def _value_noise(rng: np.random.Generator, xs: np.ndarray, ys: np.ndarray,
                 wavelength: float) -> np.ndarray:
    """Smooth value noise in [-1, 1] on the grid xs x ys (random lattice values every
    ``wavelength`` m, interpolated with a quintic fade)."""
    gx = int(np.ceil(xs.max() / wavelength)) + 2
    gy = int(np.ceil(ys.max() / wavelength)) + 2
    lattice = rng.uniform(-1.0, 1.0, (gy, gx))
    u, v = xs / wavelength, ys / wavelength
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - i0, v - j0

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    su, sv = fade(fu)[None, :], fade(fv)[:, None]
    l00, l01 = lattice[np.ix_(j0, i0)], lattice[np.ix_(j0, i0 + 1)]
    l10, l11 = lattice[np.ix_(j0 + 1, i0)], lattice[np.ix_(j0 + 1, i0 + 1)]
    return (l00 * (1 - su) + l01 * su) * (1 - sv) + (l10 * (1 - su) + l11 * su) * sv


def max_slope_deg(heights: np.ndarray, res: float) -> float:
    gy, gx = np.gradient(heights, res)
    return float(np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2).max())))


@dataclass
class BeaconField:
    heights: np.ndarray          # [n, n] absolute z on the global grid, row = y, col = x
    resolution: float
    tile_size: float
    friction: np.ndarray         # [tiles, tiles], row = y, col = x
    rocks: np.ndarray            # [k, 4]: x, y, height, radius

    @property
    def size(self) -> float:
        return (self.heights.shape[0] - 1) * self.resolution

    @property
    def tiles(self) -> int:
        return self.friction.shape[0]

    def height(self, x, y) -> np.ndarray:
        """Surface height at field coordinates (x, y), metres from the field's corner."""
        n = self.heights.shape[0] - 1
        u = np.clip(np.asarray(x, float) / self.resolution, 0, n)
        v = np.clip(np.asarray(y, float) / self.resolution, 0, n)
        return _bilinear(self.heights, u, v)

    def slope_deg(self, x, y) -> np.ndarray:
        gy, gx = np.gradient(self.heights, self.resolution)
        n = self.heights.shape[0] - 1
        u = np.clip(np.asarray(x, float) / self.resolution, 0, n)
        v = np.clip(np.asarray(y, float) / self.resolution, 0, n)
        g = np.sqrt(_bilinear(gx, u, v) ** 2 + _bilinear(gy, u, v) ** 2)
        return np.degrees(np.arctan(g))

    def tile_index(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        c = np.clip(np.floor(np.asarray(x, float) / self.tile_size).astype(int), 0, self.tiles - 1)
        r = np.clip(np.floor(np.asarray(y, float) / self.tile_size).astype(int), 0, self.tiles - 1)
        return r, c

    def friction_at(self, x, y) -> np.ndarray:
        r, c = self.tile_index(x, y)
        return self.friction[r, c]

    def tile_heights(self, r: int, c: int) -> np.ndarray:
        """Tile (row r along y, column c along x): its slice of the global grid, edge
        nodes included, so neighbouring tiles share their edge vertices."""
        k = int(round(self.tile_size / self.resolution))
        return self.heights[r * k: (r + 1) * k + 1, c * k: (c + 1) * k + 1]


def beacon_field(course: dict) -> BeaconField:
    """The field's global height grid and per-tile friction, from ``course["field"]`` and
    ``course["seed"]``:

    heights = hills (value noise, rescaled so the steepest hill slope is max_slope_deg)
            + texture (mid and high frequency noise, at most texture_amplitude_m)
            + rocks (smooth cosine bumps, rock_height_m high, rock_radius_m wide,
                     rocks_per_100m2 of them)
    """
    tiles, per_tile = _beacon_grid_shape(course)
    f = course.get("field") or {}
    res = float(course.get("hfield_resolution_m", 0.1))
    size = float(course["field_size_m"])
    n = tiles * per_tile + 1
    xs = np.linspace(0.0, size, n)
    rng = np.random.default_rng(int(course.get("seed", 0)))

    hills = np.zeros((n, n))
    for k, wl in enumerate(f.get("hill_wavelengths_m", [16.0, 8.0])):
        hills += 0.5 ** k * _value_noise(rng, xs, xs, float(wl))
    slope = max_slope_deg(hills, res)
    target = float(f.get("max_slope_deg", 17.0))
    if slope > 0:
        hills *= math.tan(math.radians(target)) / math.tan(math.radians(slope))
    texture = np.zeros((n, n))
    weights = [float(w) for w in f.get("texture_weights", [0.6, 0.4])]
    for wl, w in zip(f.get("texture_wavelengths_m", [1.0, 0.4]), weights):
        texture += w * _value_noise(rng, xs, xs, float(wl))
    texture *= float(f.get("texture_amplitude_m", 0.03)) / max(sum(weights), 1e-9)

    count = int(round(float(f.get("rocks_per_100m2", 3.0)) * size * size / 100.0))
    h_lo, h_hi = (float(v) for v in f.get("rock_height_m", [0.05, 0.12]))
    r_lo, r_hi = (float(v) for v in f.get("rock_radius_m", [0.2, 0.4]))
    rocks = np.column_stack([rng.uniform(0, size, count), rng.uniform(0, size, count),
                             rng.uniform(h_lo, h_hi, count), rng.uniform(r_lo, r_hi, count)]) \
        if count else np.zeros((0, 4))
    bumps = np.zeros((n, n))
    for x, y, h, r in rocks:
        i0, i1 = max(int((x - r) / res), 0), min(int((x + r) / res) + 2, n)
        j0, j1 = max(int((y - r) / res), 0), min(int((y + r) / res) + 2, n)
        dx = xs[i0:i1][None, :] - x
        dy = xs[j0:j1][:, None] - y
        d = np.sqrt(dx ** 2 + dy ** 2)
        bumps[j0:j1, i0:i1] += np.where(d < r, h * 0.5 * (1 + np.cos(np.pi * d / r)), 0.0)

    heights = hills + texture + bumps
    heights -= np.median(heights)

    lo, hi = (float(v) for v in f.get("friction_range", [0.4, 1.0]))
    friction = rng.uniform(lo, hi, (tiles, tiles))
    share = float(f.get("low_friction_share", 0.0))
    if share > 0:
        llo, lhi = (float(v) for v in f.get("low_friction_range", [0.3, 0.4]))
        k = int(round(share * tiles * tiles))
        idx = rng.choice(tiles * tiles, size=k, replace=False)
        friction.flat[idx] = rng.uniform(llo, lhi, k)
    return BeaconField(heights, res, float(course["tile_size_m"]), friction, rocks)


def _segments_cross(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return (orient(p1, p2, q1) != orient(p1, p2, q2)) and (orient(q1, q2, p1) != orient(q1, q2, p2))


def beacon_pairs(course: dict, k: int, seed: int, tries: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """K (start, goal) pairs in field coordinates for one episode seed.

    Goals are goal_distance_m from their start at a random bearing; starts and goals stay
    margin_m inside the field; starts are at least min_start_separation_m apart. A path
    that crosses no earlier path is preferred; after ``tries`` samples the one with the
    fewest crossings is taken.
    """
    size = float(course["field_size_m"])
    ep = course.get("episodes") or {}
    dist = float(ep.get("goal_distance_m", 20.0))
    margin = float(ep.get("margin_m", 3.0))
    sep = float(ep.get("min_start_separation_m", 5.0))
    if dist > (size - 2 * margin) * math.sqrt(2):
        raise ValueError("beacon: goal_distance_m does not fit inside the field")
    rng = np.random.default_rng(seed)
    starts, goals = [], []
    for i in range(k):
        best = None
        for _ in range(tries):
            s = rng.uniform(margin, size - margin, 2)
            if any(np.linalg.norm(s - o) < sep for o in starts):
                continue
            b = rng.uniform(-np.pi, np.pi)
            g = s + dist * np.array([np.cos(b), np.sin(b)])
            if not np.all((g >= margin) & (g <= size - margin)):
                continue
            crossings = sum(_segments_cross(s, g, a, c) for a, c in zip(starts, goals))
            if best is None or crossings < best[0]:
                best = (crossings, s, g)
            if crossings == 0:
                break
        if best is None:
            raise ValueError(f"beacon: could not place robot {i} (field too small for "
                             f"{k} robots {sep} m apart?)")
        starts.append(best[1])
        goals.append(best[2])
    return np.array(starts), np.array(goals)


def spawn_height(fld: BeaconField, xy: np.ndarray, radius: float = 0.4) -> np.ndarray:
    """Highest ground within ``radius`` of each start, so no foot spawns inside a rock."""
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    out = []
    for x, y in np.atleast_2d(xy):
        px = np.concatenate([[x], x + radius * np.cos(ang), x + radius / 2 * np.cos(ang)])
        py = np.concatenate([[y], y + radius * np.sin(ang), y + radius / 2 * np.sin(ang)])
        out.append(float(fld.height(px, py).max()))
    return np.array(out)


def beacon_primitives(course: dict, fld: BeaconField | None = None) -> list[HField]:
    """One heightfield per tile, in field coordinates."""
    fld = fld or beacon_field(course)
    bottom = float(fld.heights.min()) - BASE_DEPTH_M
    t = fld.tile_size
    return [HField(c * t, (c + 1) * t, r * t, (r + 1) * t, fld.tile_heights(r, c), bottom,
                   float(fld.friction[r, c]), r * fld.tiles + c)
            for r in range(fld.tiles) for c in range(fld.tiles)]


# ---- world frame --------------------------------------------------------------------

def course_frame(course: dict, num_lanes: int = 1) -> dict:
    """Where mjlab's generator puts the course (its grid is centred on the world origin).

    gauntlet: ``x0`` is the world x of the course start, ``lane_y`` the world y of each
    lane's centre line. beacon: ``x0``, ``y0`` are the world coordinates of the field's
    corner (field coordinates = world - corner).
    """
    if course["kind"] == "gauntlet":
        length, w = course_length(course), float(course.get("lane_width_m", 2.0))
        lane_y = [-num_lanes * w / 2 + (i + 0.5) * w for i in range(num_lanes)]
        return {"x0": -length / 2, "y0": 0.0, "lane_y": lane_y}
    size = float(course["field_size_m"])
    return {"x0": -size / 2, "y0": -size / 2, "lane_y": None}


# ---- MuJoCo geometry (mjlab terrain) ------------------------------------------------

def friction_rgba(mu: float, shade: float = 1.0) -> tuple[float, float, float, float]:
    """Warm grey for grippy ground, pale blue for slippery (mu 0.3 and below)."""
    t = float(np.clip((mu - 0.3) / 0.7, 0.0, 1.0))
    ice, rock = np.array([0.55, 0.72, 0.92]), np.array([0.58, 0.52, 0.45])
    rgb = (ice + (rock - ice) * t) * shade
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def add_primitives(spec, body, prims: list, offset=(0.0, 0.0, 0.0), prefix: str = "course",
                   shade_sections: bool = True) -> list:
    """Add ``prims`` to ``body`` of ``spec`` as box and heightfield geoms with the course
    contact settings. Returns the new geoms."""
    import mujoco

    ox, oy, oz = (float(v) for v in offset)
    geoms = []
    for k, p in enumerate(prims):
        shade = 0.9 if (shade_sections and p.section % 2) else 1.0
        if isinstance(p, Box):
            g = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[(p.x1 - p.x0) / 2, (p.y1 - p.y0) / 2, (p.z_top - p.z_bottom) / 2],
                pos=[ox + (p.x0 + p.x1) / 2, oy + (p.y0 + p.y1) / 2, oz + (p.z_top + p.z_bottom) / 2])
        else:
            h = np.asarray(p.heights, float)
            zmin, zmax = float(h.min()), float(h.max())
            rz = max(zmax - zmin, 1e-3)
            norm = (h - zmin) / rz
            hf = spec.add_hfield(name=f"{prefix}_hf{k}",
                                 size=[(p.x1 - p.x0) / 2, (p.y1 - p.y0) / 2, rz, zmin - p.z_bottom],
                                 nrow=h.shape[0], ncol=h.shape[1],
                                 userdata=norm.astype(np.float32).flatten().tolist())
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=hf.name,
                              pos=[ox + (p.x0 + p.x1) / 2, oy + (p.y0 + p.y1) / 2, oz + zmin])
        g.name = f"{prefix}_{k}"
        g.friction = [p.friction, TORSIONAL_FRICTION, ROLLING_FRICTION]
        g.priority = COURSE_PRIORITY
        g.condim = COURSE_CONDIM
        g.solref = list(COURSE_SOLREF)
        g.rgba = friction_rgba(p.friction, shade)
        geoms.append(g)
    return geoms


@functools.cache
def _course_sub_terrain_cls():
    from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

    @dataclass(kw_only=True)
    class CourseSubTerrainCfg(SubTerrainCfg):
        """One tile of mjlab's terrain grid that builds course geometry: a gauntlet lane
        (one tile per lane, all identical) or the whole beacon field (one tile)."""

        course: dict
        calls: int = 0

        def function(self, difficulty, spec, rng) -> TerrainOutput:
            del difficulty, rng                        # the course is seeded by its YAML
            body = spec.body("terrain")
            lane = self.calls
            self.calls += 1
            if self.course["kind"] == "gauntlet":
                w = float(self.course.get("lane_width_m", 2.0))
                geoms = add_primitives(spec, body, lane_primitives(self.course),
                                       offset=(0.0, w / 2, 0.0), prefix=f"course_l{lane}")
                origin = np.array([float(self.course.get("spawn_x_m", 1.0)), w / 2, 0.0])
            else:
                fld = beacon_field(self.course)
                geoms = add_primitives(spec, body, beacon_primitives(self.course, fld),
                                       prefix="tile", shade_sections=False)
                c = fld.size / 2
                origin = np.array([c, c, float(fld.height(c, c))])
            # listed so the generator moves them to this tile's place in the grid
            return TerrainOutput(origin=origin, geometries=[TerrainGeometry(geom=g) for g in geoms])

    return CourseSubTerrainCfg


def make_course_terrain(terrain_cfg: dict):
    """TerrainEntityCfg for ``terrain: {type: course, course: <course dict>, lanes: N}``."""
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

    course = copy.deepcopy(terrain_cfg["course"])
    cls = _course_sub_terrain_cls()
    if course["kind"] == "gauntlet":
        lanes = int(terrain_cfg.get("lanes", 1))
        size = (course_length(course), float(course.get("lane_width_m", 2.0)))
        rows, cols = 1, lanes
    else:
        size = (float(course["field_size_m"]),) * 2
        rows = cols = 1
    generator = TerrainGeneratorCfg(
        size=size, border_width=float(course.get("border_width_m", 10.0)),
        num_rows=rows, num_cols=cols, sub_terrains={"course": cls(course=course)},
        curriculum=False, difficulty_range=(0.0, 0.0), add_lights=True, seed=0,
        color_scheme="none")
    return TerrainEntityCfg(terrain_type="generator", terrain_generator=generator,
                            max_init_terrain_level=0)
