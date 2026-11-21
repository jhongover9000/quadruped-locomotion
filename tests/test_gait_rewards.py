"""Gait-shaping reward terms, reward weight ramps, experiment presets and YAML floats.

The term tests use hand-built states (a stand-in env object holding tensors); the last
test builds a real environment with every new term on (``pytest -m sim``).
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from quadruped_rl.envs import rewards as R
from quadruped_rl.robots import load_robot
from quadruped_rl.utils.config import (
    apply_dotted,
    apply_overrides,
    available,
    load_group,
    load_preset,
    parse_yaml,
)

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ["astra_D", "astra_E", "astra_F", "astra_G"]


def fake_env(**data):
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(**data))})


# ---- terms ----------------------------------------------------------------------------

def test_knee_bend_hinge():
    # calves at indices 2, 5, 8, 11; knee bends negative in the sim convention
    q = torch.zeros(3, 12)
    q[0, [2, 5, 8, 11]] = -1.1                 # bent past min_bend: no cost
    q[1, [2, 5, 8, 11]] = -0.5                 # 0.4 short on each leg
    q[2, [2, 5, 8, 11]] = torch.tensor([0.0, -0.9, -0.3, -2.0])
    q[:, [0, 1, 3, 4]] = -3.0                  # other joints are ignored
    ids = R.calf_joint_ids(load_robot("astra_stance2").joint_names)
    assert ids == [2, 5, 8, 11]
    out = R.knee_bend(fake_env(joint_pos=q), ids, 0.9)
    assert torch.allclose(out, torch.tensor([0.0, 1.6, 0.9 + 0.0 + 0.6 + 0.0]))


def test_band_penalty():
    h = torch.tensor([0.40, 0.43, 0.45, 0.47, 0.50])
    out = R.band_penalty(h, 0.43, 0.47)
    assert torch.allclose(out, torch.tensor([0.03 ** 2, 0.0, 0.0, 0.0, 0.03 ** 2]), atol=1e-7)


def test_height_above_feet_uses_feet_in_contact():
    base = torch.tensor([0.5, 0.5])
    feet = torch.tensor([[0.0, 0.0, 0.1, 0.1], [0.02, 0.04, 0.06, 0.08]])
    contact = torch.tensor([[True, True, False, False], [False, False, False, False]])
    h = R.height_above_feet(base, feet, contact)
    assert torch.allclose(h, torch.tensor([0.5, 0.5 - 0.05]))   # no contact: all feet


def test_torques_rated_with_and_without_continuous_torque():
    astra = load_robot("astra_stance2")
    assert R.rated_torque(astra) == [5.0] * 12                   # continuous_torque
    go1 = load_robot("go1")
    assert go1.actuator.continuous_torque is None
    assert R.rated_torque(go1) == go1.actuator.torque_limit      # falls back to the limit

    tau = torch.zeros(2, 12)
    tau[0, 0] = 5.0                     # at the rating: 1
    tau[1, :] = 10.0                    # twice the rating everywhere: 12 * 4
    assert torch.allclose(R.torques_rated(fake_env(actuator_force=tau), R.rated_torque(astra)),
                          torch.tensor([1.0, 48.0]))
    ref = R.rated_torque(go1)
    out = R.torques_rated(fake_env(actuator_force=tau), ref)
    assert torch.allclose(out[1], sum((10.0 / r) ** 2 for r in ref) * torch.ones(()))


def test_clearance_kernel():
    h = torch.tensor([[0.07, 0.07, 0.0, 0.0]])
    fast = torch.tensor([[1.0, 1.0, 0.0, 0.0]])      # stance feet (speed 0) do not count
    assert torch.allclose(R.clearance_kernel(h, fast, 0.07, 0.05, 2.0), torch.ones(1))
    low = torch.zeros(1, 4)
    expected = torch.exp(-torch.tensor(2 * 0.07 ** 2 * torch.tanh(torch.tensor(2.0)).item() / 0.05))
    assert torch.allclose(R.clearance_kernel(low, fast, 0.07, 0.05, 2.0), expected)


def test_stumble():
    f = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 10.0], [0, 0, 0], [0, 0, 0]],   # fine
                      [[30.0, 0.0, 10.0], [0, 0, 5.0], [0, 0, 0], [0, 0, 0]]])     # 30 > 2 x 10
    assert torch.equal(R.stumble(f, 2.0), torch.tensor([0.0, 1.0]))


def test_torque_rate_is_zero_after_reset():
    ref = [5.0] * 12
    data = SimpleNamespace(actuator_force=torch.zeros(2, 12))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
    term = R.torque_rate(None, env)
    assert torch.equal(term(env, ref), torch.zeros(2))          # first call: no history
    data.actuator_force = torch.zeros(2, 12)
    data.actuator_force[:, 0] = 10.0
    assert torch.allclose(term(env, ref), torch.tensor([4.0, 4.0]))   # (10 / 5)^2
    term.reset(torch.tensor([1]))
    data.actuator_force = torch.full((2, 12), 5.0)
    out = term(env, ref)
    assert out[1] == 0.0                                            # reset env: no spike
    assert out[0] == pytest.approx(1.0 * 11 + ((5.0 - 10.0) / 5.0) ** 2)
    assert torch.allclose(term(env, ref), torch.zeros(2))           # steady torque: 0


def test_feet_impact_threshold():
    weight = 17.0 * 9.81
    quarter = weight / 4.0
    fz = torch.tensor([[quarter, 1.5 * quarter, 0.0, 0.0],              # at/below k mg/4: 0
                       [-(1.5 * quarter + weight), 0.0, 0.0, 0.0]])    # sign ignored: 1 mg over
    out = R.impact(fz, weight, 1.5)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(1.0)


# ---- ramps ----------------------------------------------------------------------------

def test_ramp_schedule():
    s = lambda step: R.ramp_scale(step, 7200, 24000, 0.0)
    assert s(0) == 0.0 and s(7200) == 0.0
    assert s(15600) == pytest.approx(0.5)
    assert s(24000) == 1.0 and s(10 ** 7) == 1.0
    assert R.ramp_scale(100, 0, 200, 0.2) == pytest.approx(0.6)


def test_ramp_curriculum_term_scales_the_live_weight():
    term = SimpleNamespace(weight=-0.01)
    env = SimpleNamespace(common_step_counter=0,
                          reward_manager=SimpleNamespace(get_term_cfg=lambda name: term))
    params = {"reward_name": "torques_rated", "start_steps": 100, "end_steps": 300,
              "start_scale": 0.0}
    ramp = R.reward_weight_ramp(SimpleNamespace(params=params), env)
    ramp(env, None, **params)
    assert term.weight == 0.0
    env.common_step_counter = 200
    ramp(env, None, **params)
    assert term.weight == pytest.approx(-0.005)
    env.common_step_counter = 1000
    ramp(env, None, **params)
    assert term.weight == pytest.approx(-0.01)


def test_builder_registers_ramps_only_for_enabled_terms():
    from quadruped_rl.envs.builder import make_reward_ramps, make_rewards

    preset = load_preset("astra_E")["task"]
    task = apply_dotted(load_group("tasks", "flat_velocity"), preset)
    rewards = make_rewards(task, load_robot("astra_stance2"))
    assert "torques" not in rewards
    assert rewards["torques_rated"].weight == preset["rewards.torques_rated"]
    ramps = make_reward_ramps(task, rewards)
    assert ramps["ramp_torques_rated"].params == {
        "reward_name": "torques_rated", "start_steps": 7200, "end_steps": 24000,
        "start_scale": 0.0}
    task["rewards"]["torques_rated"] = 0.0
    assert make_reward_ramps(task, make_rewards(task, load_robot("astra_stance2"))) == {}


def test_default_tasks_keep_the_old_reward_terms():
    from quadruped_rl.envs.builder import make_rewards

    old = {"track_lin_vel_xy", "track_ang_vel_z", "lin_vel_z", "ang_vel_xy", "flat_orientation",
           "torques", "joint_acc", "action_rate", "joint_pos_limits", "feet_air_time",
           "undesired_contacts"}
    for task in ("flat_velocity", "rough_velocity"):
        for robot in ("go1", "astra_stance2"):
            assert set(make_rewards(load_group("tasks", task), load_robot(robot))) == old


# ---- YAML floats, presets -------------------------------------------------------------

def test_yaml_exponent_floats():
    assert parse_yaml("-1e-3") == -0.001 and isinstance(parse_yaml("1e-3"), float)
    assert parse_yaml("2E4") == 20000.0 and parse_yaml("1.0e3") == 1000.0
    assert parse_yaml("10") == 10 and isinstance(parse_yaml("10"), int)
    assert parse_yaml("abc") == "abc" and parse_yaml("1e") == "1e"
    cfg = apply_overrides({}, ["rewards.torques=-1e-3", "x=[1e-3, 2]"])
    assert cfg["rewards"]["torques"] == -0.001
    assert cfg["x"] == [0.001, 2]


def test_presets_load_and_chain():
    assert set(PRESETS) <= set(available("experiments"))
    d, e, f, g = (load_preset(n) for n in PRESETS)
    assert d["robot"] == "astra_stance2"
    assert d["task"]["reward_params.base_height_band"] == [0.43, 0.47]
    assert e["task"]["rewards.foot_clearance"] == d["task"]["rewards.foot_clearance"]   # E = D + ...
    assert e["task"]["rewards.torques"] == 0.0
    assert f["task"]["action.scale_mode"] == "absolute" and "rewards.torques_rated" in f["task"]
    assert g["task_name"] == "rough_velocity" and g["task"] == f["task"]
    for preset in (d, e, f, g):
        assert preset["task"]["reward_params.foot_clearance_std"] == 0.01
        for value in preset["task"].values():
            assert not isinstance(value, str) or value == "absolute"


def test_preset_then_set_precedence():
    preset = load_preset("astra_G")
    task = apply_dotted(load_group("tasks", preset["task_name"]), preset["task"])
    task = apply_overrides(task, ["rewards.feet_slip=-1e-3"])
    assert task["rewards"]["feet_slip"] == -0.001                       # --set wins
    assert task["rewards"]["knee_bend"] == preset["task"]["rewards.knee_bend"]
    assert task["rewards"]["flat_orientation"] == -0.2                   # rough's own change
    assert task["action"]["scale_mode"] == "absolute"


def test_train_dry_run_with_preset():
    out = subprocess.run(
        [sys.executable, "scripts/train.py", "--algo", "ppo", "--preset", "astra_G", "--dry-run",
         "--set", "rewards.feet_slip=-1e-3"],
        cwd=ROOT, capture_output=True, text=True, timeout=300, check=True).stdout
    resolved = parse_yaml(out)
    assert resolved["preset"] == "astra_G"
    assert resolved["robot"]["name"] == "astra_stance2"
    assert resolved["task"]["name"] == "rough_velocity"
    assert resolved["task"]["rewards"]["feet_slip"] == -0.001
    assert resolved["task"]["reward_ramps"]["torques_rated"]["end_steps"] == 24000


# ---- in simulation --------------------------------------------------------------------

@pytest.mark.sim
def test_new_terms_run_in_simulation():
    from quadruped_rl.envs.builder import make_env

    task = apply_dotted(load_group("tasks", "flat_velocity"), load_preset("astra_E")["task"])
    task["rewards"]["feet_stumble"] = -0.1
    env = make_env(load_robot("astra_stance2"), task, num_envs=4, device="cpu")
    try:
        env.reset()
        active = set(env.reward_manager.active_terms)
        assert {"foot_clearance", "feet_slip", "feet_stumble", "base_height_band", "knee_bend",
                "torques_rated"} <= active
        assert "ramp_torques_rated" in env.curriculum_manager.active_terms
        assert env.reward_manager.get_term_cfg("torques_rated").weight == 0.0   # ramp at step 0
        for _ in range(10):
            env.step(torch.zeros(4, 12))
        for name in ("foot_clearance", "feet_slip", "base_height_band", "knee_bend"):
            term = env.reward_manager.get_term_cfg(name)
            value = term.func(env, **term.params)
            assert value.shape == (4,) and torch.isfinite(value).all(), name
        # standing at the default pose: knees bent 1.1 > 0.9, trunk ~0.48 above the feet
        knee = env.reward_manager.get_term_cfg("knee_bend")
        assert torch.allclose(knee.func(env, **knee.params), torch.zeros(4), atol=1e-3)
    finally:
        env.close()
