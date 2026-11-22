"""Terrain curriculum promotion, play-mode difficulty range, spawn-tile logging, astra_J_rough.

The unit tests use a stand-in env object; the ``sim`` tests build a small CPU environment
or run scripts/record.py (``pytest -m sim``).
"""

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from quadruped_rl.envs import terrain as T  # noqa: E402
from quadruped_rl.utils.config import apply_dotted, load_group, load_preset, parse_yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# ---- curriculum promotion ---------------------------------------------------------------

class _Scene(dict):
    """env.scene: entities by name, plus .terrain and .env_origins."""


def fake_curriculum_env(walked_m, time_outs, terminated, step=100):
    n = len(walked_m)
    calls = []

    def update_env_origins(env_ids, move_up, move_down):
        calls.append((move_up.clone(), move_down.clone()))

    terrain = SimpleNamespace(
        cfg=SimpleNamespace(terrain_generator=SimpleNamespace(
            size=(8.0, 8.0), sub_terrains={"flat": None, "hf_pyramid_slope": None})),
        update_env_origins=update_env_origins,
        terrain_levels=torch.full((n,), 3), terrain_types=torch.zeros(n, dtype=torch.long),
        terrain_origins=torch.zeros(10, 2, 3))
    pos = torch.zeros(n, 3)
    pos[:, 0] = torch.tensor(walked_m, dtype=torch.float)
    scene = _Scene(robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=pos)))
    scene.terrain, scene.env_origins = terrain, torch.zeros(n, 3)
    env = SimpleNamespace(
        scene=scene, common_step_counter=step, max_episode_length_s=20.0,
        command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([[0.5, 0.0, 0.0]] * n)),
        termination_manager=SimpleNamespace(time_outs=torch.tensor(time_outs),
                                            terminated=torch.tensor(terminated)))
    return env, calls


def test_mjlab_promotes_a_robot_that_fell_after_half_a_tile():
    # robot 0: timed out after 5 m; robot 1: fell after 5 m; robot 2: fell after 1 m
    env, calls = fake_curriculum_env([5.0, 5.0, 1.0], [True, False, False], [False, True, True])
    T.terrain_levels_vel(env, torch.arange(3), "twist")          # default: mjlab's term
    up, down = calls[-1]
    assert up.tolist() == [True, True, False]                    # the fall is promoted
    assert down.tolist() == [False, False, True]


def test_promote_requires_timeout():
    env, calls = fake_curriculum_env([5.0, 5.0, 1.0, 5.0], [True, False, False, True],
                                     [False, True, True, True])
    out = T.terrain_levels_vel(env, torch.arange(4), "twist", promote_requires_timeout=True)
    up, down = calls[-1]
    # robot 3 hit the time-out and fell on the same step: a fall, not promoted
    assert up.tolist() == [True, False, False, False]
    # demotion as before: only robots that covered less than half the commanded distance
    assert down.tolist() == [False, False, True, False]
    assert set(out) == {"mean", "max", "flat"}                   # every fake robot is in column 0

    env, calls = fake_curriculum_env([5.0], [True], [False], step=0)   # first reset: frozen
    T.terrain_levels_vel(env, torch.arange(1), "twist", promote_requires_timeout=True)
    assert calls[-1][0].tolist() == [False] and calls[-1][1].tolist() == [False]


def test_builder_passes_promote_requires_timeout():
    from quadruped_rl.envs.builder import make_env_cfg
    from quadruped_rl.robots import load_robot

    task = load_group("tasks", "rough_velocity")
    robot = load_robot("go1")
    term = make_env_cfg(robot, task, num_envs=2).curriculum["terrain_levels"]
    assert term.func is T.terrain_levels_vel and term.params["promote_requires_timeout"] is False
    task["curriculum"]["promote_requires_timeout"] = True
    term = make_env_cfg(robot, task, num_envs=2).curriculum["terrain_levels"]
    assert term.params["promote_requires_timeout"] is True
    assert "terrain_levels" not in make_env_cfg(robot, task, num_envs=2, play=True).curriculum


# ---- play-mode terrain --------------------------------------------------------------------

def test_play_difficulty_range_config():
    terrain = dict(load_group("tasks", "rough_velocity")["terrain"])
    assert terrain["play_difficulty_range"] == [0.0, 1.0]
    terrain["play_difficulty_range"] = [0.2, 0.4]
    play = T.make_terrain(terrain, play=True).terrain_generator
    assert play.difficulty_range == (0.2, 0.4) and not play.curriculum
    assert all(isinstance(s, T._LoggedSubTerrain) for s in play.sub_terrains.values())
    train = T.make_terrain(terrain).terrain_generator             # training ignores it
    assert train.difficulty_range == (0.0, 1.0) and train.curriculum
    del terrain["play_difficulty_range"]
    assert T.make_terrain(terrain, play=True).terrain_generator.difficulty_range == (0.0, 1.0)
    for bad in ([0.5, 0.2], [-0.1, 0.5], [0.0, 1.5]):
        terrain["play_difficulty_range"] = bad
        with pytest.raises(ValueError, match="play_difficulty_range"):
            T.make_terrain(terrain, play=True)


def test_slope_degrees():
    slope = T._sub_terrain("hf_pyramid_slope", {"proportion": 0.2, "slope_range": [0.0, 0.6]})
    inv = T._sub_terrain("hf_pyramid_slope_inv", {"proportion": 0.2, "slope_range": [0.0, 1.0]})
    assert T._slope_deg(slope, 1.0) == pytest.approx(math.degrees(math.atan(0.6)))   # 30.96
    assert T._slope_deg(slope, 0.5) == pytest.approx(math.degrees(math.atan(0.3)))
    assert T._slope_deg(inv, 1.0) == pytest.approx(45.0)
    assert T._slope_deg(T._sub_terrain("flat", 0.1), 1.0) is None


# ---- astra_J_rough ------------------------------------------------------------------------

def j_rough_task():
    preset = load_preset("astra_J_rough")
    return preset, apply_dotted(load_group("tasks", preset["task_name"]), preset["task"])


def test_astra_j_rough_preset():
    preset, task = j_rough_task()
    h_preset, h_task = (load_preset("astra_H_rough"),
                        apply_dotted(load_group("tasks", "rough_velocity"), load_preset("astra_H_rough")["task"]))
    assert preset["robot"] == h_preset["robot"] == "astra_stance2"
    assert preset["algo_set"] == {"max_iterations": 3000}
    assert task["rewards"] == h_task["rewards"] and task["reward_params"] == h_task["reward_params"]
    assert task["curriculum"]["promote_requires_timeout"] is True
    assert task["terrain"]["play_difficulty_range"] == [0.0, 1.0]
    sub = task["terrain"]["sub_terrains"]
    props = {n: s if isinstance(s, (int, float)) else s["proportion"] for n, s in sub.items()}
    assert props == {"flat": 0.1, "pyramid_stairs": 0.15, "pyramid_stairs_inv": 0.15,
                     "hf_pyramid_slope": 0.2, "hf_pyramid_slope_inv": 0.2, "random_rough": 0.1,
                     "wave_terrain": 0.1}
    assert sub["pyramid_stairs"]["step_height_range"] == [0.0, 0.1]          # kept from the task
    # The steepest slope anywhere (training rows or play tiles, difficulty <= 1) is 31 deg.
    for mode in (False, True):
        gen = T.make_terrain(task["terrain"], play=mode).terrain_generator
        hi = max(gen.difficulty_range)
        for name in ("hf_pyramid_slope", "hf_pyramid_slope_inv"):
            cfg = getattr(gen.sub_terrains[name], "inner", gen.sub_terrains[name])
            assert cfg.slope_range == (0.0, 0.6)
            assert T._slope_deg(cfg, hi) == pytest.approx(30.96, abs=0.01)


def test_train_dry_run_astra_j_rough():
    out = subprocess.run(
        [sys.executable, "scripts/train.py", "--preset", "astra_J_rough", "--dry-run"],
        cwd=ROOT, capture_output=True, text=True, timeout=300, check=True).stdout
    resolved = parse_yaml(out)
    assert resolved["algo"]["max_iterations"] == 3000
    assert resolved["task"]["curriculum"]["promote_requires_timeout"] is True
    out = subprocess.run(
        [sys.executable, "scripts/train.py", "--preset", "astra_J_rough", "--dry-run",
         "--max-iterations", "7"],
        cwd=ROOT, capture_output=True, text=True, timeout=300, check=True).stdout
    assert parse_yaml(out)["algo"]["max_iterations"] == 7                     # command line wins


# ---- in simulation ------------------------------------------------------------------------

@pytest.mark.sim
def test_play_grid_respects_difficulty_range_and_logs_tiles():
    from quadruped_rl.envs.builder import make_env
    from quadruped_rl.robots import load_robot

    _, task = j_rough_task()
    task["terrain"]["play_difficulty_range"] = [0.5, 0.75]
    env = make_env(load_robot("go1"), task, num_envs=6, device="cpu", play=True, seed=0)
    try:
        gen = env.scene.terrain.cfg.terrain_generator
        log = next(iter(gen.sub_terrains.values())).log
        assert len(log) == 25 and all(0.5 <= d <= 0.75 for _, d in log)
        tiles = T.spawn_tiles(env)
        assert len(tiles) == 6
        names = list(task["terrain"]["sub_terrains"])
        for i, t in enumerate(tiles):
            assert t["robot"] == i and t["type"] in names and 0.5 <= t["difficulty"] <= 0.75
            assert (t["slope_deg"] is not None) == t["type"].startswith("hf_pyramid_slope")
            # the robot's origin is the centre of the tile named
            origin = env.scene.terrain.terrain_origins[t["row"], t["col"]]
            assert torch.allclose(env.scene.env_origins[i], origin)
        for name, d in log:
            if name.startswith("hf_pyramid_slope"):
                assert T._slope_deg(gen.sub_terrains[name].inner, d) <= 31.0
    finally:
        env.close()


@pytest.mark.sim
def test_record_writes_spawn_tiles(tmp_path):
    out = tmp_path / "rec"
    proc = subprocess.run(
        [sys.executable, "scripts/record.py", "--robot", "go1", "--task", "rough_velocity",
         "--command", "0", "0", "0", "--duration", "0.2", "--num-envs", "3", "--no-video",
         "--device", "cpu", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    meta = json.loads((out / "meta.json").read_text())
    tiles = meta["terrain_tiles"]
    assert [t["robot"] for t in tiles] == [0, 1, 2]
    for t in tiles:
        assert set(t) == {"robot", "row", "col", "type", "difficulty", "slope_deg"}
        assert 0.0 <= t["difficulty"] <= 1.0
    assert "Spawn tiles" in proc.stdout and "Spawn tiles" in (out / "summary.txt").read_text()
