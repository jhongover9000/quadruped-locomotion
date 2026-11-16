"""Every algorithm, task, checkpoint format and log format fits the shared interfaces.

These are the contracts scripts/train.py, play.py, record.py, evaluate.py and
plot_training.py rely on; a new algorithm or task that breaks one fails here.
"""

import numpy as np
import pytest

from quadruped_rl.algorithms import (
    IncompatibleError,
    algorithm_names,
    check_compatible,
    compatibility_table,
    get_algorithm,
)
from quadruped_rl.algorithms.base import Agent, task_kind
from quadruped_rl.utils.config import available, load_group

IMPLEMENTED = ["ppo", "ppo_rslrl", "supermimic"]


# ---- algorithms x tasks -------------------------------------------------------------

def test_registry_matches_configs():
    assert sorted(algorithm_names()) == available("algorithms")
    for name in algorithm_names():
        assert load_group("algorithms", name)["name"] == name


def test_every_algorithm_declares_the_interface():
    for name in algorithm_names():
        cls = get_algorithm(name)
        assert issubclass(cls, Agent)
        assert cls.task_kinds and set(cls.task_kinds) <= {"velocity", "tracking"}
        assert cls.checkpoint_patterns
        assert cls.implemented == (name in IMPLEMENTED), name


def test_every_task_has_a_known_kind():
    for name in available("tasks"):
        assert task_kind(load_group("tasks", name)) in ("velocity", "tracking")
    assert task_kind({"name": "saved before kinds existed"}) == "velocity"


def test_implemented_algorithms_run_on_their_default_task():
    for name in IMPLEMENTED:
        check_compatible(name, load_group("tasks", load_group("algorithms", name)["default_task"]))


@pytest.mark.parametrize("algo,task", [("ppo", "motion_tracking"), ("ppo_rslrl", "motion_tracking"),
                                       ("supermimic", "flat_velocity"), ("supermimic", "rough_velocity")])
def test_wrong_kind_of_task_is_refused_with_alternatives(algo, task):
    with pytest.raises(IncompatibleError) as e:
        check_compatible(algo, load_group("tasks", task))
    assert "Algorithms for it" in str(e.value)


@pytest.mark.parametrize("algo", ["amp", "deepmimic", "mpc"])
def test_planned_algorithms_refuse_to_train(algo, tmp_path):
    """A stub must never train silently as something else (AMP used to run plain PPO)."""
    with pytest.raises(IncompatibleError, match="not implemented"):
        check_compatible(algo, load_group("tasks", load_group("algorithms", algo)["default_task"]))
    agent = get_algorithm(algo)(robot=None, task_cfg={}, algo_cfg={}, run_dir=tmp_path)
    with pytest.raises(NotImplementedError):
        agent.train()
    with pytest.raises(NotImplementedError):
        agent.make_policy(None, tmp_path / "x.pt")


def test_table_lists_everything():
    table = compatibility_table()
    for name in algorithm_names() + available("tasks"):
        assert name in table


# ---- checkpoints --------------------------------------------------------------------

def test_checkpoints_are_found_in_every_trainers_layout(tmp_path):
    from quadruped_rl.algorithms.checkpoints import checkpoint_number, list_checkpoints

    for rel in ("checkpoints/model_100.pt", "checkpoints/model_1500.pt",      # ours
                "model_0.pt", "model_299.pt",                                 # rsl_rl
                "ckpt_0002000.pt", "ckpt_0010000.pt", "ckpt_latest.pt"):      # SuperMimic
        p = tmp_path / rel
        p.parent.mkdir(exist_ok=True)
        p.touch()
    assert [checkpoint_number(p) for p in list_checkpoints(tmp_path)] == [0, 100, 299, 1500, 2000, 10000]
    for name, last in (("ppo", 1500), ("supermimic", 10000)):
        found = list_checkpoints(tmp_path, get_algorithm(name).checkpoint_patterns)
        assert checkpoint_number(found[-1]) == last
        assert all(p.name != "ckpt_latest.pt" for p in found)


def test_checkpoint_from_another_trainer_is_named_in_the_error(tmp_path):
    from quadruped_rl.algorithms.checkpoints import checkpoint_format, require_format

    ours = {"model": {}, "optimizer": {}, "iteration": 1, "learning_rate": 1e-3}
    rsl = {"actor_state_dict": {}, "iter": 3}
    sm = {"wm": {}, "policy": {}, "critic": {}, "config": {}}
    assert [checkpoint_format(c) for c in (ours, rsl, sm, {})] == ["ppo", "ppo_rslrl", "supermimic", "unknown"]
    require_format(ours, "ppo", tmp_path / "a.pt")
    with pytest.raises(ValueError, match="written by 'ppo_rslrl'"):
        require_format(rsl, "ppo", tmp_path / "a.pt")
    with pytest.raises(ValueError, match="written by 'ppo'"):
        require_format(ours, "supermimic", tmp_path / "a.pt")


PPO_ALGO = {"steps_per_env": 8, "learning_rate": 1e-3, "num_epochs": 1, "num_minibatches": 1,
            "clip_ratio": 0.2, "gamma": 0.99, "gae_lambda": 0.95, "entropy_coef": 0.0,
            "value_loss_coef": 1.0, "max_grad_norm": 1.0, "init_noise_std": 0.5,
            "actor_hidden": [16], "critic_hidden": [16], "activation": "elu"}


def test_ppo_policy_loads_where_only_the_critic_differs(tmp_path):
    """A flat-ground policy must play on rough terrain: same actor, larger critic input."""
    torch = pytest.importorskip("torch")
    from quadruped_rl.algorithms.ppo.agent import PPOAgent

    trained = PPOAgent(robot=None, task_cfg={}, algo_cfg=PPO_ALGO, run_dir=tmp_path)
    trained.build(actor_dim=10, critic_dim=12, action_dim=3)
    trained.save(tmp_path / "model_5.pt")
    obs = torch.randn(4, 10)

    other = PPOAgent(robot=None, task_cfg={}, algo_cfg=PPO_ALGO, run_dir=tmp_path)
    other.build(actor_dim=10, critic_dim=40, action_dim=3)
    with pytest.raises(RuntimeError):                  # the full state does not fit...
        other.load(tmp_path / "model_5.pt")
    other.load(tmp_path / "model_5.pt", actor_only=True)   # ...the actor does
    assert torch.allclose(other.act(obs), trained.act(obs))

    wrong = PPOAgent(robot=None, task_cfg={}, algo_cfg=PPO_ALGO, run_dir=tmp_path)
    wrong.build(actor_dim=11, critic_dim=12, action_dim=3)
    with pytest.raises(ValueError, match="actor observations"):
        wrong.load(tmp_path / "model_5.pt", actor_only=True)


# ---- training curves ----------------------------------------------------------------

def _fake_run(tmp_path, name, algo, tags, steps=40, extra_cfg=None):
    yaml = pytest.importorskip("yaml")
    writer_mod = pytest.importorskip("torch.utils.tensorboard")
    run = tmp_path / name
    run.mkdir()
    cfg = {"algo": {"name": algo, "num_envs": 100, "steps_per_env": 24, **(extra_cfg or {})},
           "task": {"name": "t"}}
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    w = writer_mod.SummaryWriter(str(run / ("tb" if algo == "supermimic" else "")))
    for i in range(1, steps + 1):
        for k, tag in enumerate(tags):
            w.add_scalar(tag, float(i * (k + 1)), i)
    w.close()
    return run


def test_curves_put_every_trainer_on_the_same_axes(tmp_path):
    from quadruped_rl.evaluation import curves as C

    ours = C.load_curves(_fake_run(tmp_path, "a", "ppo", ["Train/mean_return", "Loss/value_loss",
                                                         "Episode_Reward/x", "Policy/learning_rate"]))
    rsl = C.load_curves(_fake_run(tmp_path, "b", "ppo_rslrl", ["Train/mean_reward", "Loss/value",
                                                              "Loss/learning_rate"]))
    sm = C.load_curves(_fake_run(tmp_path, "c", "supermimic", [
        "train/episode_norm_return", "policy/critic_loss", "lr", "wm/online_nse_1step",
        "time/sim_s_per_iter", "time/wm_s_per_iter", "time/policy_s_per_iter"]))
    for run in (ours, rsl, sm):
        for quantity in ("return", "value_loss", "learning_rate"):
            assert run.get(quantity) is not None, (run.algo, quantity)
    # One PPO iteration is steps_per_env steps in every environment; one SuperMimic iteration is one.
    assert ours.env_steps_per_iteration == rsl.env_steps_per_iteration == 2400
    assert sm.env_steps_per_iteration == 100
    assert sm.get("steps_per_s") is not None           # derived from the per-phase timers
    s = ours.get("return")
    assert np.allclose(ours.x(s, "env_steps"), s.step * 2400)

    assert C.plot_dashboard(ours, tmp_path / "ours.png").stat().st_size > 10_000
    assert C.plot_dashboard(sm, tmp_path / "sm.png").stat().st_size > 10_000
    assert C.plot_comparison([ours, rsl, sm], tmp_path / "cmp.png").stat().st_size > 10_000
    C.export_csv(sm, tmp_path / "sm.csv")
    header = (tmp_path / "sm.csv").read_text().splitlines()[0]
    assert header.startswith("iteration,env_steps,") and "wm/online_nse_1step" in header
    assert C.summarize(ours)["iterations"] == 40


def test_comparison_refuses_more_runs_than_colours(tmp_path):
    from quadruped_rl.evaluation import curves as C

    run = C.load_curves(_fake_run(tmp_path, "a", "ppo", ["Train/mean_return"], steps=3))
    with pytest.raises(ValueError, match="At most"):
        C.plot_comparison([run] * (C.MAX_SERIES + 1), tmp_path / "x.png")


# ---- stall watchdog -----------------------------------------------------------------

def test_watchdog_counts_env_steps_as_progress(monkeypatch):
    from quadruped_rl.utils import watchdog

    class Env:
        def step(self, a):
            return a + 1

    env = Env()
    assert watchdog.watch_env(env) is env and env.step(1) == 2      # not installed: untouched
    monkeypatch.setitem(watchdog._state, "installed", True)
    monkeypatch.setitem(watchdog._state, "last", 0.0)
    watchdog.watch_env(env)
    watchdog.watch_env(env)                                         # wrapping twice is a no-op
    assert env.step(1) == 2
    assert watchdog._state["last"] > 0.0 and watchdog._state["what"] == "env.step"


def test_supervisor_ends_a_run_that_hangs_holding_the_interpreter_lock(tmp_path):
    """The case a watchdog thread cannot see: one C call that never returns (in practice,
    torch.compile stuck inside sympy). The supervising process must end it with code 75."""
    import subprocess
    import sys
    import textwrap

    from quadruped_rl.utils.paths import REPO_ROOT
    from quadruped_rl.utils.watchdog import STALL_EXIT_CODE

    script = tmp_path / "job.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT / "src")!r})
        from quadruped_rl.utils import watchdog
        watchdog.supervise(2.0)
        watchdog.install(stall_s=0)
        watchdog.heartbeat("start")
        if sys.argv[1] == "hang":
            sum(range(10**13))
        print("finished")
    """))
    ok = subprocess.run([sys.executable, str(script), "ok"], capture_output=True, text=True,
                        check=False)
    assert ok.returncode == 0 and "finished" in ok.stdout
    hung = subprocess.run([sys.executable, str(script), "hang"], capture_output=True, text=True,
                          timeout=60, check=False)
    assert hung.returncode == STALL_EXIT_CODE
    assert "made no progress" in hung.stderr and "job.py" in hung.stderr     # its stack was printed


# ---- SuperMimic backend -------------------------------------------------------------

def _backend():
    from quadruped_rl.algorithms.supermimic import backend

    if backend.find_root(load_group("algorithms", "supermimic")) is None:
        pytest.skip("SuperMimic reference repository not found (see docs/supermimic.md)")
    pytest.importorskip("torch")
    return backend


def test_supermimic_config_translates_to_the_reference_trainer(tmp_path):
    backend = _backend()
    algo = load_group("algorithms", "supermimic")
    task = load_group("tasks", "motion_tracking")
    algo.update(num_envs=512, max_iterations=1234, save_interval=77, max_hours=1.5)
    cfg = backend.build_train_config("go1", task, algo, tmp_path / "run", "cpu", seed=3)
    assert (cfg.robot, cfg.seed, cfg.env.num_envs) == ("go1", 3, 512)
    assert cfg.schedule.total_iterations == 1234 and cfg.schedule.checkpoint_interval == 77
    assert cfg.schedule.max_hours == 1.5
    assert (cfg.wm_trainer.batch_size, cfg.policy_trainer.batch_size) == (256, 512)
    assert cfg.log_root / cfg.run_name == tmp_path / "run"          # logs land in our run directory
    assert cfg.data.group == task["motions"]["train_groups"]
    assert cfg.data.npz_dir.name == "go1" and cfg.data.manifest.name == "manifest.csv"
    assert cfg.env.clip_weight_alpha == task["motions"]["sampling"]["clip_weight_alpha"]
    assert cfg.world_model.hidden == tuple(algo["world_model"]["hidden"])

    algo["policy_trainer"]["no_such_field"] = 1
    with pytest.raises(KeyError, match="no_such_field"):
        backend.build_train_config("go1", task, algo, tmp_path / "run", "cpu", seed=0)
    with pytest.raises(KeyError, match="no SuperMimic RobotSpec"):
        backend.build_train_config("other_robot", task, load_group("algorithms", "supermimic"),
                                   tmp_path / "run", "cpu", seed=0)


def test_supermimic_robot_matches_our_robot_config():
    """Tracking and velocity policies must be trained on the same robot. Known difference:
    the hip angle of the default pose (ours +-0.1 rad, the spec 0). Anything else is new."""
    backend = _backend()
    backend.ensure_importable()
    from supermimic.robots import get_robot_spec, set_robot_overrides

    from quadruped_rl.robots import load_robot

    set_robot_overrides()
    diffs = backend.robot_spec_differences(load_robot("go1"), get_robot_spec("go1"),
                                           load_group("tasks", "flat_velocity")["action"])
    unexpected = [d for d in diffs if not (d.startswith("default pose") and "hip" in d)]
    assert not unexpected, unexpected
