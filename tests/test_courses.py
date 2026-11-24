"""Evaluation courses (envs/courses.py, evaluation/courses.py, scripts/course.py).

Geometry, controllers and metrics are tested on plain numpy; the MuJoCo tests compile
the course into a model on the CPU (no mjlab). The ``sim`` tests build mjlab
environments or run scripts/course.py (``pytest -m sim``).
"""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from quadruped_rl.envs import courses as C

ROOT = Path(__file__).resolve().parents[1]


# ---- gauntlet geometry ------------------------------------------------------------------

def test_gauntlet_sections_and_lengths():
    course = C.load_course("gauntlet")
    sections = C.gauntlet_sections(course)
    labels = [s.label for s in sections]
    assert labels[0] == "start" and labels[-1] == "finish"
    for want in ("low friction", "incline 10", "decline 10", "rough", "stairs up", "stairs down",
                 "incline 20", "decline 20", "blocks", "curb up", "curb down",
                 "slippery decline", "wave"):
        assert want in labels
    # back to back, each as long as its YAML entry says
    for spec, s, nxt in zip(course["sections"], sections, sections[1:] + [None]):
        want = spec["n"] * spec["tread_m"] if spec["type"] == "stairs" else spec["length_m"]
        assert s.length == pytest.approx(want)
        if nxt is not None:
            assert nxt.x0 == s.x1
    assert sections[0].x0 == 0.0
    assert C.course_length(course) == pytest.approx(sum(s.length for s in sections))
    assert 50.0 < C.course_length(course) < 70.0
    by = {s.label: s for s in sections}
    assert by["low friction"].friction == 0.35 and by["slippery decline"].friction == 0.4
    assert by["incline 10"].z_end == pytest.approx(4.0 * math.tan(math.radians(10)))
    assert by["incline 20"].z_end - by["incline 20"].z_start == pytest.approx(
        3.0 * math.tan(math.radians(20)))
    assert by["stairs up"].z_end - by["rough"].z_end == pytest.approx(0.40)
    assert by["stairs down"].z_end == pytest.approx(by["rough"].z_end)
    assert by["curb up"].z_start - by["blocks"].z_end == pytest.approx(0.10)


def _profile_jumps(course, y: float, threshold: float = 0.005):
    """x of every vertical jump above ``threshold`` along the line at lane y."""
    prims = C.lane_primitives(course)
    x = np.arange(0.0, C.course_length(course) + 1e-9, 0.001)
    z = C.surface_height(prims, x, np.full_like(x, y))
    assert not np.isnan(z).any(), "holes in the track"
    dz = np.abs(np.diff(z))
    return x[1:][dz > threshold], z


@pytest.mark.parametrize("y", [0.0, -0.55, 0.83])
def test_gauntlet_profile_is_continuous_except_at_jumps(y):
    course = C.load_course("gauntlet")
    sections = C.gauntlet_sections(course)
    jumps, _ = _profile_jumps(course, y)
    allowed = [s for s in sections if s.type in C.JUMP_TYPES]
    for xj in jumps:
        assert any(s.x0 - 0.002 <= xj <= s.x1 + 0.002 for s in allowed), \
            f"unexpected step at x = {xj:.3f} (y = {y})"
    # with 1 mm sampling the steepest continuous slope (20 deg ramp, rough cells) moves
    # well under 5 mm per sample, so the threshold only catches real steps
    assert len(jumps) > 0
    # the nominal end height of every section is the start height of the next one
    x = np.arange(0.0, C.course_length(course), 0.001)
    nominal = C.nominal_height(sections, x)
    for a, b in zip(sections, sections[1:]):
        if b.type not in ("stairs", "curb"):
            assert b.z_start == pytest.approx(a.z_end)
        i = int(round(b.x0 / 0.001))
        if a.type not in C.JUMP_TYPES and b.type not in C.JUMP_TYPES:
            assert abs(nominal[i] - nominal[i - 1]) < 0.002


def test_gauntlet_continuity_check_catches_a_gap():
    course = C.load_course("gauntlet")
    course["sections"] = [dict(s) for s in course["sections"]]
    # a wave that ends half a wavelength early would leave a step: refused
    course["sections"][18] = dict(course["sections"][18], length_m=3.5)
    with pytest.raises(ValueError, match="whole number of wavelengths"):
        C.gauntlet_sections(course)
    with pytest.raises(ValueError, match="unknown type"):
        C.gauntlet_sections({"sections": [{"type": "lava", "length_m": 1}]})
    with pytest.raises(ValueError, match="needs"):
        C.gauntlet_sections({"sections": [{"type": "ramp", "length_m": 1}]})


def test_gauntlet_lanes_are_identical_and_seeded():
    course = C.load_course("gauntlet")
    a, b = C.lane_primitives(course), C.lane_primitives(course)
    assert len(a) == len(b)
    for p, q in zip(a, b):
        if isinstance(p, C.HField):
            assert np.array_equal(p.heights, q.heights)
        else:
            assert p.z_top == q.z_top
    other = dict(course, seed=course["seed"] + 1)
    rough = [p for p in C.lane_primitives(other) if isinstance(p, C.HField) and p.heights.shape[0] > 2]
    assert not np.array_equal(rough[0].heights, [p for p in a if isinstance(p, C.HField)
                                                  and p.heights.shape[0] > 2][0].heights)


def test_course_frame_matches_mjlab_grid_centering():
    course = C.load_course("gauntlet")
    f = C.course_frame(course, num_lanes=4)
    assert f["x0"] == pytest.approx(-C.course_length(course) / 2)
    assert f["lane_y"] == pytest.approx([-3.0, -1.0, 1.0, 3.0])
    assert C.course_frame(C.load_course("beacon"))["x0"] == -30.0


# ---- beacon geometry --------------------------------------------------------------------

def _small_beacon():
    course = C.load_course("beacon")
    course.update(field_size_m=12.0, tile_size_m=4.0)
    course["field"] = dict(course["field"], rocks_per_100m2=20.0)
    return course


def _stored_tile_heights(p: C.HField) -> np.ndarray:
    """Heights as MuJoCo will see them: float32 normalized data x size + position."""
    h = np.asarray(p.heights, float)
    zmin, rz = float(h.min()), max(float(h.max() - h.min()), 1e-3)
    norm = ((h - zmin) / rz).astype(np.float32)
    return zmin + norm.astype(float) * rz


def test_beacon_tiles_share_edges():
    course = C.load_course("beacon")
    fld = C.beacon_field(course)
    prims = C.beacon_primitives(course, fld)
    n = fld.tiles
    assert n == 10 and len(prims) == 100
    tiles = {(p.section // n, p.section % n): _stored_tile_heights(p) for p in prims}
    worst = 0.0
    for (r, c), h in tiles.items():
        if c + 1 < n:          # right edge of (r, c) = left edge of (r, c + 1)
            worst = max(worst, np.abs(h[:, -1] - tiles[(r, c + 1)][:, 0]).max())
        if r + 1 < n:          # top edge of (r, c) = bottom edge of (r + 1, c)
            worst = max(worst, np.abs(h[-1, :] - tiles[(r + 1, c)][0, :]).max())
    assert worst < 1e-3, f"tile edge mismatch {worst * 1000:.3f} mm"
    # and the tiles reproduce the global grid
    assert np.abs(tiles[(3, 7)] - fld.tile_heights(3, 7)).max() < 1e-5


def test_beacon_field_parameters():
    course = C.load_course("beacon")
    fld = C.beacon_field(course)
    assert fld.size == 60.0 and fld.heights.shape == (601, 601)
    f = course["field"]
    lo, hi = f["friction_range"]
    low = fld.friction < lo
    assert low.mean() == pytest.approx(f["low_friction_share"], abs=0.011)
    assert np.all((fld.friction[low] >= 0.3) & (fld.friction[low] <= 0.4))
    assert np.all((fld.friction[~low] >= lo) & (fld.friction[~low] <= hi))
    assert len(fld.rocks) == round(f["rocks_per_100m2"] * 36)
    assert np.all((fld.rocks[:, 2] >= 0.05) & (fld.rocks[:, 2] <= 0.12))
    # the field is seeded
    again = C.beacon_field(course)
    assert np.array_equal(fld.heights, again.heights) and np.array_equal(fld.friction, again.friction)
    assert not np.array_equal(fld.heights, C.beacon_field(dict(course, seed=1)).heights)
    # hills alone have the configured steepest slope
    flat = dict(course, field=dict(f, texture_amplitude_m=0.0, rocks_per_100m2=0.0))
    assert C.max_slope_deg(C.beacon_field(flat).heights, 0.1) == pytest.approx(
        f["max_slope_deg"], abs=0.5)


def test_beacon_pairs():
    course = C.load_course("beacon")
    s, g = C.beacon_pairs(course, 16, seed=1000)
    assert s.shape == g.shape == (16, 2)
    assert np.allclose(np.linalg.norm(g - s, axis=1), 20.0)
    for p in (s, g):
        assert np.all((p >= 3.0) & (p <= 57.0))
    d = np.linalg.norm(s[:, None] - s[None], axis=-1) + np.eye(16) * 99
    assert d.min() >= 5.0
    crossings = sum(C._segments_cross(s[i], g[i], s[j], g[j]) for i in range(16) for j in range(i))
    assert crossings <= 2
    s2, _ = C.beacon_pairs(course, 16, seed=1000)
    assert np.array_equal(s, s2)
    assert not np.array_equal(s, C.beacon_pairs(course, 16, seed=1001)[0])


def test_beacon_lookups():
    fld = C.beacon_field(_small_beacon())
    assert fld.tiles == 3
    assert fld.friction_at(5.0, 9.0) == fld.friction[2, 1]
    x, y = 6.05, 2.3
    assert fld.height(x, y) == pytest.approx(C.surface_height(C.beacon_primitives(_small_beacon(), fld),
                                                              np.array([x]), np.array([y]))[0])
    assert 0.0 <= float(fld.slope_deg(x, y)) < 60.0
    assert C.spawn_height(fld, np.array([[6.0, 6.0]]))[0] >= fld.height(6.0, 6.0)


# ---- the compiled MuJoCo geometry matches the numpy description ---------------------------

def _compile(prims, priority_check=False):
    mujoco = pytest.importorskip("mujoco")
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="terrain")
    C.add_primitives(spec, body, prims)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def _ray_heights(mujoco, model, data, x, y):
    gid = np.zeros(1, np.int32)
    out = []
    for xi, yi in zip(x, y):
        d = mujoco.mj_ray(model, data, np.array([xi, yi, 10.0]), np.array([0.0, 0.0, -1.0]),
                          None, 1, -1, gid)
        out.append(10.0 - d if d >= 0 else np.nan)
    return np.array(out)


def test_compiled_gauntlet_matches_profile():
    course = C.load_course("gauntlet")
    prims = C.lane_primitives(course)
    mujoco, model, data = _compile(prims)
    rng = np.random.default_rng(0)
    sections = C.gauntlet_sections(course)
    smooth = [s for s in sections if s.type in ("flat", "ramp", "wave", "curb", "stairs", "blocks")]
    for s in smooth:            # away from the vertical edges of boxes
        x = rng.uniform(s.x0 + 0.02, s.x1 - 0.02, 20)
        y = rng.uniform(-0.95, 0.95, 20)
        if s.type in ("stairs", "blocks"):
            keep = [not any(abs(xi - e) < 0.01 for e in np.arange(s.x0, s.x1, 0.1)) for xi in x]
            x, y = x[keep], y[keep]
        np.testing.assert_allclose(_ray_heights(mujoco, model, data, x, y),
                                   C.surface_height(prims, x, y), atol=1e-6)
    rough = next(p for p in prims if p.section == next(s.index for s in sections if s.type == "rough"))
    ny, nx = rough.heights.shape          # nodes: MuJoCo's triangles and our bilinear agree
    jx, jy = rng.integers(1, nx - 1, 30), rng.integers(1, ny - 1, 30)   # interior nodes
    x = rough.x0 + jx * (rough.x1 - rough.x0) / (nx - 1)
    y = rough.y0 + jy * (rough.y1 - rough.y0) / (ny - 1)
    np.testing.assert_allclose(_ray_heights(mujoco, model, data, x, y), rough.heights[jy, jx], atol=1e-5)


def test_compiled_course_contact_settings():
    course = C.load_course("gauntlet")
    prims = C.lane_primitives(course)
    mujoco, model, _ = _compile(prims)
    low = next(s.index for s in C.gauntlet_sections(course) if s.label == "low friction")
    for i, p in enumerate(prims):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"course_{i}")
        assert model.geom_priority[g] == C.COURSE_PRIORITY
        assert model.geom_condim[g] == C.COURSE_CONDIM
        assert model.geom_friction[g, 0] == pytest.approx(p.friction)
        assert tuple(model.geom_solref[g]) == pytest.approx(C.COURSE_SOLREF)
        if p.section == low:
            assert model.geom_friction[g, 0] == pytest.approx(0.35)


def test_compiled_beacon_tile_edges():
    course = _small_beacon()
    fld = C.beacon_field(course)
    mujoco, model, data = _compile(C.beacon_primitives(course, fld))
    # sample just either side of every interior tile edge
    rng = np.random.default_rng(1)
    for e in (4.0, 8.0):
        y = rng.uniform(0.2, 11.8, 25)
        left = _ray_heights(mujoco, model, data, np.full(25, e - 1e-6), y)
        right = _ray_heights(mujoco, model, data, np.full(25, e + 1e-6), y)
        assert np.abs(left - right).max() < 1e-3
        x = rng.uniform(0.2, 11.8, 25)
        below = _ray_heights(mujoco, model, data, x, np.full(25, e - 1e-6))
        above = _ray_heights(mujoco, model, data, x, np.full(25, e + 1e-6))
        assert np.abs(below - above).max() < 1e-3


# ---- controllers ------------------------------------------------------------------------

from quadruped_rl.evaluation import courses as E  # noqa: E402

RANGES = {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)}
LANE = dict(speed=0.8, k_y=1.0, vy_max=0.3, k_yaw=1.5, wz_max=0.8, ranges=RANGES)


def test_lane_command_signs_and_clipping():
    y = np.array([0.1, -0.1, 2.0, 0.0, 0.0, 0.0])
    yaw = np.array([0.0, 0.0, 0.0, 0.2, -0.2, 3.0])
    cmd = E.lane_command(y, yaw, np.zeros(6), **LANE)
    assert np.allclose(cmd[:, 0], 0.8)
    assert cmd[0, 1] == pytest.approx(-0.1) and cmd[1, 1] == pytest.approx(0.1)   # back to the line
    assert cmd[2, 1] == pytest.approx(-0.3)                                       # vy_max
    assert cmd[3, 2] == pytest.approx(-0.3) and cmd[4, 2] == pytest.approx(0.3)   # turn back to +x
    assert cmd[5, 2] == pytest.approx(-0.8)                                       # wz_max
    # the lane is per robot, and the trained command range wins over the gains
    cmd = E.lane_command([2.1], [0.0], [2.0], **dict(LANE, speed=1.6, vy_max=1.0, wz_max=2.0))
    assert cmd[0, 0] == 1.0 and cmd[0, 1] == pytest.approx(-0.1)
    tight = dict(LANE, ranges={**RANGES, "lin_vel_y": (-0.05, 0.05)})
    assert E.lane_command([1.0], [0.0], [0.0], **tight)[0, 1] == pytest.approx(-0.05)


def test_heading_error_wraps():
    assert E.wrap_pi(3 * np.pi / 2) == pytest.approx(-np.pi / 2)
    cmd = E.lane_command([0.0], [-3.0], [0.0], **LANE)     # facing almost -x, turned right
    assert cmd[0, 2] == pytest.approx(0.8)                 # turns left (the short way)


BEACON = dict(v_max=0.8, k_yaw=1.5, wz_max=1.0, slow_radius_m=1.5, ranges=RANGES)


def test_beacon_command():
    goal = np.array([[10.0, 0.0]] * 5)
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [9.25, 0.0]])
    yaw = np.array([0.0, 0.3, -0.3, np.pi, 0.0])
    cmd = E.beacon_command(xy, yaw, goal, **BEACON)
    assert cmd[0].tolist() == pytest.approx([0.8, 0.0, 0.0])
    assert cmd[1, 2] == pytest.approx(-0.45) and cmd[2, 2] == pytest.approx(0.45)
    assert cmd[1, 0] == pytest.approx(0.8 * np.cos(0.3))
    assert cmd[3, 0] == 0.0 and abs(cmd[3, 2]) == pytest.approx(1.0)   # behind: turn, no walking
    assert cmd[4, 0] == pytest.approx(0.8 * 0.75 / 1.5)                # slowing near the goal
    assert np.all(cmd[:, 1] == 0.0)
    # goal to the left (+y) of a robot facing +x: turn left
    assert E.beacon_command([[0, 0]], [0.0], [[0, 5]], **BEACON)[0, 2] == pytest.approx(1.0)


def test_goal_check():
    assert E.goal_reached([[0.0, 0.49], [0.0, 0.51]], [[0.0, 0.0]] * 2, 0.5).tolist() == [True, False]


def _beacon_ctrl(n=3, limit=None):
    course = C.load_course("beacon")
    starts = np.zeros((n, 3))
    goals = np.column_stack([np.full(n, 20.0), np.zeros(n), np.zeros(n)])
    return E.BeaconController(course, starts, goals, RANGES, dt=0.02, time_limit_s=limit)


def test_beacon_outcomes_and_fall_masking():
    ctrl = _beacon_ctrl(3)
    assert ctrl.time_limit_s == pytest.approx(3 * 20 / 0.8 + 10)
    z = np.zeros(3, bool)
    pos = np.zeros((3, 3))
    for t in range(1100):
        pos[0, 0] = min(20.0, t * 0.02 * 1.0)      # robot 0 walks to the goal at 1 m/s
        pos[1, 0] = t * 0.02 * 0.5                 # robot 1 walks, then falls at t = 100
        pos[2, 0] = 0.01 * (t % 3)                 # robot 2 shuffles in place
        fell = z.copy()
        fell[1] = t == 100
        if t == 100:
            pos[1] = 0.0                           # the env already reset it to the start
        ctrl.update(t, pos, np.zeros(3), fell, z)
        if t == 100:
            pos[1] = 50.0                          # anything after the fall must not count
    st, end = ctrl.attempts.status, ctrl.attempts.end
    assert st[1] == "fell" and end[1] == 100       # steps 0..99 count, the reset step does not
    assert st[2] == "stuck" and end[2] == round(10.0 / 0.02) + 1
    assert st[0] == "success" and end[0] == pytest.approx(19.5 / 0.02, abs=2)


def test_gauntlet_outcomes_and_results():
    course = C.load_course("gauntlet")
    ctrl = E.GauntletController(course, 2, RANGES, dt=0.02, time_limit_s=200.0)
    assert ctrl.time_limit_s == 200.0
    default = E.GauntletController(course, 2, RANGES, dt=0.02)
    assert default.time_limit_s == pytest.approx(3 * C.course_length(course) / 0.8)
    T, dt = 3000, 0.02
    sections = C.gauntlet_sections(course)
    fall_x = sections[2].x0 + 1.0                  # robot 1 falls 1 m into "incline 10"
    qpos = np.zeros((T, 2, 19), np.float32)
    qpos[..., 3] = 1.0
    x = 1.0 + np.arange(T) * dt * 1.0              # 1 m/s from the spawn point
    qpos[:, :, 0] = (ctrl.x0 + x)[:, None]
    qpos[:, :, 1] = np.array(ctrl.lane_y)[None] + 0.1
    fell = np.zeros((T, 2), bool)
    t_fall = int(np.argmax(x >= fall_x))
    fell[t_fall, 1] = True
    qpos[t_fall:, 1, 0] = ctrl.x0 + 1.0            # reset to the start, then walks on (masked)
    z = np.zeros(2, bool)
    for t in range(T):
        pos = qpos[t, :, :3].astype(float)
        if ctrl.update(t, pos, np.zeros(2), fell[t], z):
            break
    T = t + 1
    rollout = {"qpos": qpos[:T], "dt": np.float32(dt),
               "torques": np.ones((T, 2, 12), np.float32), "joint_vel": np.ones((T, 2, 12), np.float32),
               "base_lin_vel_b": np.tile(np.array([1.0, 0, 0], np.float32), (T, 2, 1)),
               "foot_vel_w": np.zeros((T, 2, 4, 3), np.float32), "foot_contact": np.ones((T, 2, 4), bool)}
    rows, robots = E.gauntlet_results(rollout, ctrl, mass=10.0, legs=["FR", "FL", "RR", "RL"],
                                      continuous=[5.0] * 12)
    assert robots[0]["finished"] and robots[0]["status"] == "finished"
    finish = C.course_length(course) - 1.0
    assert robots[0]["total_time_s"] == pytest.approx((finish - 1.0) / 1.0, abs=0.05)
    assert robots[1]["fell"] and robots[1]["fall_section"] == "incline 10"
    assert robots[1]["fall_x_m"] == pytest.approx(fall_x, abs=0.05)      # not the reset position
    assert robots[1]["x_reached_m"] < fall_x + 0.05                    # later steps masked
    r1 = {r["label"]: r for r in rows if r["robot"] == 1}
    assert r1["low friction"]["exited"] and r1["incline 10"]["entered"] and not r1["incline 10"]["exited"]
    assert not r1["rough"]["entered"]
    r0 = {r["label"]: r for r in rows if r["robot"] == 0}
    assert r0["low friction"]["time_s"] == pytest.approx(4.0, abs=0.03)
    assert r0["low friction"]["mean_speed"] == pytest.approx(1.0, abs=0.01)
    assert r0["low friction"]["lateral_rms_m"] == pytest.approx(0.1, abs=1e-4)
    assert r0["low friction"]["energy_J"] == pytest.approx(12 * 4.0, rel=0.01)
    assert r0["low friction"]["cot"] == pytest.approx(12 / (10.0 * 9.81))
    assert r0["low friction"]["hip_rms_Nm"] == pytest.approx(1.0)
    assert r0["start"]["time_s"] == pytest.approx(2.0, abs=0.03)     # spawned 1 m in
    agg = {a["label"]: a for a in E.aggregate_gauntlet(rows)}
    assert agg["incline 10"]["entered"] == 2 and agg["incline 10"]["completion"] == 0.5
    assert agg["incline 10"]["falls"] == 1 and agg["rough"]["entered"] == 1
    text = E.format_gauntlet("test", rows, robots)
    assert "incline 10" in text and "50%" in text
    assert "incline 10" in E.compare_gauntlet({"a": rows, "b": rows})


def test_eval_robot_strips_randomization():
    from quadruped_rl.robots import load_robot

    robot = load_robot("astra_stance2")
    assert "friction" in robot.domain_randomization
    ev = E.eval_robot(robot)
    lo, hi = robot.domain_randomization.get("action_delay_steps", (0, 0))
    want = (lo + hi) // 2
    assert ev.domain_randomization == ({"action_delay_steps": [want, want]} if want else {})
    assert robot.domain_randomization["friction"]                       # input left alone
    task = E.course_task({"commands": {}, "terrain": {"type": "flat"}}, {"kind": "beacon"}, 4)
    assert task["terrain"]["type"] == "course" and task["fixed_reset"] and task["terrain"]["lanes"] == 4


def test_tracking_runs_are_refused():
    with pytest.raises(ValueError, match="velocity"):
        E.check_compatible("supermimic", {"kind": "tracking", "name": "motion_tracking"})
    with pytest.raises(ValueError, match="velocity"):
        E.check_compatible("deepmimic", {"name": "x"})
    E.check_compatible("ppo", {"name": "rough_velocity"})


def test_fixed_reset_events():
    pytest.importorskip("mjlab")
    from quadruped_rl.envs.domain_randomization import make_events
    from quadruped_rl.robots import load_robot

    robot = E.eval_robot(load_robot("astra_stance2"))
    events = make_events(robot, play=True, fixed_reset=True)
    assert events["reset_base"].params["pose_range"] == {}
    assert events["reset_joints"].params["position_range"] == (0.0, 0.0)
    assert set(events) == {"reset_base", "reset_joints"}             # no startup randomization
    assert make_events(robot, play=True)["reset_base"].params["pose_range"]["yaw"][1] > 3


def test_preview_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    paths = E.preview_gauntlet(C.load_course("gauntlet"), tmp_path)
    paths += E.preview_beacon(C.load_course("beacon"), tmp_path, k=4, seed=1000)
    assert [p.name for p in paths] == ["gauntlet_top.png", "gauntlet_side.png", "beacon_top.png"]
    assert all(p.stat().st_size > 10_000 for p in paths)


# ---- in simulation ------------------------------------------------------------------------

def _course_env(course_name: str, num_envs: int = 2):
    from quadruped_rl.envs.builder import make_env
    from quadruped_rl.robots import load_robot
    from quadruped_rl.utils.config import load_group

    course = C.load_course(course_name)
    task = E.course_task(load_group("tasks", "flat_velocity"), course, num_envs)
    robot = E.eval_robot(load_robot("go1"))
    return course, make_env(robot, task, num_envs=num_envs, device="cpu", play=True, seed=0)


@pytest.mark.sim
def test_friction_in_effect_on_a_low_friction_section():
    """MuJoCo would take the larger of the two frictions (the foot's 1.0); the course
    geoms' priority makes the course's 0.35 the friction of every foot contact on it."""
    import mujoco
    import torch

    course, env = _course_env("gauntlet")
    try:
        frame = C.course_frame(course, 2)
        assert torch.allclose(env.scene.env_origins[:, 1].cpu(), torch.tensor(frame["lane_y"]))
        assert torch.allclose(env.scene.env_origins[:, 0].cpu(),
                              torch.full((2,), frame["x0"] + course["spawn_x_m"]))
        low = next(s for s in C.gauntlet_sections(course) if s.label == "low friction")
        env.scene.terrain.env_origins[0, 0] = frame["x0"] + (low.x0 + low.x1) / 2
        env.reset()
        for _ in range(20):
            env.step(torch.zeros(2, env.action_manager.total_action_dim))
        m = env.sim.mj_model
        n = int(env.sim.data.nacon[0])
        con = env.sim.data.contact
        geoms = con.geom[:n].cpu().numpy()
        friction = con.friction[:n].cpu().numpy()
        world = con.worldid[:n].cpu().numpy()
        dim = con.dim[:n].cpu().numpy()
        names = [[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in pair] for pair in geoms]
        foot = [i for i, pair in enumerate(names) if any("foot" in g for g in pair)]
        on_low = [i for i in foot if world[i] == 0]
        on_start = [i for i in foot if world[i] == 1]
        assert len(on_low) == 4 and len(on_start) == 4           # every foot is down
        foot_mu = m.geom_friction[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                                    "robot/FR_foot_collision"), 0]
        assert foot_mu == pytest.approx(1.0)                      # max() would give 1.0
        assert np.allclose(friction[on_low, 0], 0.35) and np.allclose(friction[on_low, 1], 0.35)
        assert np.allclose(friction[on_start, 0], 1.0)
        # torsional and rolling friction, condim and solref as for the feet in training
        assert np.allclose(friction[on_low, 2:], [5e-3, 5e-4, 5e-4])
        assert np.all(dim[foot] == 6)
        fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "robot/FR_foot_collision")
        cid = int(geoms[on_low[0]][1] if "course" in names[on_low[0]][1] else geoms[on_low[0]][0])
        assert m.geom_condim[cid] == m.geom_condim[fid]
        assert np.allclose(m.geom_solref[cid], m.geom_solref[fid])
        assert np.allclose(m.geom_friction[cid, 1:], m.geom_friction[fid, 1:])
    finally:
        env.close()


def _course_script(tmp_path, course, *extra):
    out = tmp_path / course
    proc = subprocess.run(
        [sys.executable, "scripts/course.py", "--robot", "go1", "--course", course,
         "--num-envs", "2", "--time-limit", "1.0", "--device", "cpu", "--out", str(out), *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return out, proc


@pytest.mark.sim
def test_gauntlet_smoke(tmp_path):
    import csv
    import json

    out, _ = _course_script(tmp_path, "gauntlet")
    with open(out / "results.csv") as f:
        rows = list(csv.DictReader(f))
    sections = C.gauntlet_sections(C.load_course("gauntlet"))
    assert len(rows) == 2 * len(sections)
    assert {r["run"] for r in rows} == {"zero_actions_go1"} and {r["seed"] for r in rows} == {"1000"}
    assert rows[0]["label"] == "start" and rows[0]["entered"] == "True"
    text = (out / "summary.txt").read_text()
    assert "low friction" in text and "Comparison" in text
    meta = json.loads((out / "meta.json").read_text())
    assert meta["course"]["kind"] == "gauntlet" and meta["seeds"] == [1000]
    assert meta["runs"][0]["conditions"]["domain_randomization"] == "none"
    assert meta["git"]


@pytest.mark.sim
def test_beacon_smoke_with_video(tmp_path):
    import csv
    import json

    out, _ = _course_script(tmp_path, "beacon", "--seeds", "5", "--video")
    with open(out / "results.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and {r["seed"] for r in rows} == {"5"}
    starts, goals = C.beacon_pairs(C.load_course("beacon"), 2, 5)
    assert float(rows[1]["start_x"]) == pytest.approx(starts[1, 0], abs=1e-3)
    assert float(rows[1]["goal_y"]) == pytest.approx(goals[1, 1], abs=1e-3)
    assert all(r["outcome"] in ("timeout", "fell", "stuck") for r in rows)
    assert "success rate" in (out / "summary.txt").read_text()
    rec = out / "recordings" / "zero_actions_go1" / "seed5"
    for name in ("rollout.npz", "model.mjb.gz", "meta.json", "video_robot0.mp4"):
        assert (rec / name).exists(), name
    meta = json.loads((rec / "meta.json").read_text())
    assert len(meta["goals_w"]) == 2
    # robots start where the course put them, facing their goal
    from quadruped_rl.evaluation import render

    _, rollout, _ = render.load_recording(rec)
    frame = C.course_frame(C.load_course("beacon"))
    xy0 = rollout["qpos"][0, :, :2] - [frame["x0"], frame["y0"]]
    assert np.allclose(xy0, starts, atol=0.1)
    yaw = E.yaw_from_quat(rollout["qpos"][0, :, 3:7])
    want = np.arctan2(*(goals - starts)[:, ::-1].T)
    assert np.allclose(E.wrap_pi(yaw - want), 0.0, atol=0.1)


def test_course_script_refuses_tracking_runs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text("algo: {name: supermimic}\ntask: {kind: tracking, name: motion_tracking}\n")
    proc = subprocess.run([sys.executable, "scripts/course.py", str(run), "--course", "gauntlet"],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0 and "velocity" in proc.stderr
