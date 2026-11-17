"""DeepMimic: PPO in the SuperMimic backend's tracking environment.

The fast tests need no simulator: the observation adapter, the checkpoint identity, that
DeepMimic's environment is configured like SuperMimic's, and the motion sampler on the
real clip library. The ``sim`` tests build the tracking environment on the CPU (reference-
state resets, the tracking reward, early termination) and train for a few iterations.
Tests that need the reference repository or its clip library skip without them.
"""

import ast
from pathlib import Path

import pytest

from quadruped_rl.utils.config import load_group

torch = pytest.importorskip("torch")

from quadruped_rl.algorithms.deepmimic.agent import (
    OBS_GROUPS,
    DeepMimicAgent,
    TrackingObservations,
)
from quadruped_rl.algorithms.ppo.agent import PPOAgent

TINY_PPO = {"steps_per_env": 4, "learning_rate": 1e-3, "adaptive_lr_target_kl": 0.01,
            "num_epochs": 1, "num_minibatches": 1, "clip_ratio": 0.2, "gamma": 0.95,
            "gae_lambda": 0.95, "entropy_coef": 0.0, "value_loss_coef": 1.0, "max_grad_norm": 1.0,
            "init_noise_std": 0.5, "actor_hidden": [32], "critic_hidden": [32], "activation": "elu",
            "save_interval": 1}


def _backend():
    from quadruped_rl.algorithms.supermimic import backend

    root = backend.find_root(load_group("algorithms", "deepmimic"))
    if root is None:
        pytest.skip("SuperMimic reference repository not found (see docs/supermimic.md)")
    return backend, root


def _clips_or_skip(agent, groups):
    try:
        return agent.clip_files(groups)
    except FileNotFoundError as e:
        pytest.skip(str(e))


# ---- observation adapter ------------------------------------------------------------

class FakeTrackingEnv:
    """The backend environment's interface, with recognizable observations."""

    def __init__(self, n=3):
        self.num_envs, self.max_episode_length, self.step_dt = n, 50, 0.02
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.closed = False

    def _obs(self):
        return {"policy": torch.ones(self.num_envs, 4), "goal": torch.full((self.num_envs, 2), 2.0)}

    def get_observations(self):
        return self._obs()

    def reset(self):
        return self._obs(), {"log": {}}

    def step(self, actions):
        z = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), torch.ones(self.num_envs), z, z, {"log": {"x": 1.0}}

    def close(self):
        self.closed = True


def test_actor_and_critic_see_proprioception_then_goal():
    assert OBS_GROUPS == ("policy", "goal")
    raw = FakeTrackingEnv()
    env = TrackingObservations(raw)
    expected = torch.tensor([[1.0, 1.0, 1.0, 1.0, 2.0, 2.0]] * 3)
    obs, _ = env.reset()
    assert set(obs) == {"actor", "critic"}
    assert torch.equal(obs["actor"], expected) and torch.equal(obs["critic"], expected)
    obs, _, _, _, extras = env.step(torch.zeros(3, 12))
    assert torch.equal(obs["actor"], expected) and extras["log"]["x"] == 1.0
    assert torch.equal(env.get_observations()["actor"], expected)
    # Everything else is the wrapped environment's, including PPO's episode desynchronization.
    assert env.num_envs == 3 and env.max_episode_length == 50 and env.unwrapped is raw
    env.episode_length_buf = torch.full((3,), 7)
    assert raw.episode_length_buf.tolist() == [7, 7, 7]
    env.close()
    assert raw.closed


def test_ppo_learn_runs_unchanged_on_the_adapter(tmp_path):
    agent = DeepMimicAgent(robot=None, task_cfg={}, algo_cfg=TINY_PPO, run_dir=tmp_path)
    raw = FakeTrackingEnv(n=8)
    raw.num_actions = 12
    agent.learn(TrackingObservations(raw), max_iterations=2)
    assert agent.iteration == 2
    assert (tmp_path / "checkpoints" / "model_2.pt").exists()


# ---- checkpoints --------------------------------------------------------------------

class FakeSampler:
    def __init__(self, n_bins):
        self.command = object()
        self.state = {"bin_return": torch.arange(n_bins, dtype=torch.float32),
                      "bin_memory": torch.ones(n_bins)}

    def state_dict(self):
        return {k: v.clone() for k, v in self.state.items()}

    def load_state_dict(self, state):
        if state["bin_return"].shape != self.state["bin_return"].shape:
            return False
        self.state = {k: v.clone() for k, v in state.items()}
        return True


def test_checkpoint_round_trip_and_identity(tmp_path):
    from quadruped_rl.algorithms.checkpoints import checkpoint_format, load_checkpoint

    trained = DeepMimicAgent(robot=None, task_cfg={}, algo_cfg=TINY_PPO, run_dir=tmp_path)
    trained.build(actor_dim=10, critic_dim=10, action_dim=3)
    trained.sampler = FakeSampler(5)
    trained.iteration = 7
    trained.save(tmp_path / "model_7.pt")
    ckpt = load_checkpoint(tmp_path / "model_7.pt")          # weights_only: plain tensors and lists
    assert checkpoint_format(ckpt) == "deepmimic"
    assert ckpt["deepmimic"]["observations"] == ["policy", "goal"]

    # Resume: weights, iteration and the reset sampler's statistics come back.
    resumed = DeepMimicAgent(robot=None, task_cfg={}, algo_cfg=TINY_PPO, run_dir=tmp_path)
    resumed.build(actor_dim=10, critic_dim=10, action_dim=3)
    resumed.sampler = FakeSampler(5)
    resumed.sampler.state["bin_return"].zero_()
    resumed.load(tmp_path / "model_7.pt")
    assert resumed.iteration == 7
    assert torch.equal(resumed.sampler.state["bin_return"], torch.arange(5, dtype=torch.float32))
    obs = torch.randn(4, 10)
    assert torch.allclose(resumed.act(obs), trained.act(obs))

    # A different clip set keeps its own sampler state instead of failing.
    other = DeepMimicAgent(robot=None, task_cfg={}, algo_cfg=TINY_PPO, run_dir=tmp_path)
    other.build(actor_dim=10, critic_dim=10, action_dim=3)
    other.sampler = FakeSampler(9)
    other.load(tmp_path / "model_7.pt")
    assert other.sampler.state["bin_return"].shape == (9,)

    # Networks can be sized from the file alone (the benchmark has no training env).
    fresh = DeepMimicAgent(robot=None, task_cfg={}, algo_cfg=TINY_PPO, run_dir=tmp_path)
    fresh._build_from_checkpoint(tmp_path / "model_7.pt")
    assert torch.allclose(fresh.act(obs), trained.act(obs))


def test_checkpoints_do_not_cross_between_velocity_ppo_and_deepmimic(tmp_path):
    ppo = PPOAgent(robot=None, task_cfg={}, algo_cfg=TINY_PPO, run_dir=tmp_path)
    ppo.build(actor_dim=10, critic_dim=10, action_dim=3)
    ppo.save(tmp_path / "ppo_1.pt")
    dm = DeepMimicAgent(robot=None, task_cfg={}, algo_cfg=TINY_PPO, run_dir=tmp_path)
    dm.build(actor_dim=10, critic_dim=10, action_dim=3)
    dm.save(tmp_path / "dm_1.pt")

    with pytest.raises(ValueError, match="written by 'ppo'; load it with --algo ppo"):
        dm.load(tmp_path / "ppo_1.pt")
    with pytest.raises(ValueError, match="written by 'ppo'"):
        dm.load(tmp_path / "ppo_1.pt", actor_only=True)        # play / record
    with pytest.raises(ValueError, match="written by 'deepmimic'; load it with --algo deepmimic"):
        ppo.load(tmp_path / "dm_1.pt")


# ---- same environment as SuperMimic -------------------------------------------------

def test_deepmimic_is_configured_like_supermimic():
    dm, sm = load_group("algorithms", "deepmimic"), load_group("algorithms", "supermimic")
    assert dm["default_task"] == sm["default_task"] == "motion_tracking"
    assert dm["backend"] == sm["backend"]
    assert dm["env"]["action_noise_std"] == 0.0               # PPO explores by itself
    # Every other environment setting SuperMimic makes, DeepMimic makes identically
    # (replay_capacity sizes SuperMimic's world-model buffer, which DeepMimic does not have).
    shared = {k: v for k, v in sm["env"].items() if k not in ("action_noise_std", "replay_capacity")}
    assert {k: dm["env"].get(k) for k in shared} == shared
    assert set(dm["env"]) <= set(shared) | {"action_noise_std"}


def _call_kwargs(tree, func_name: str, callee: str) -> dict[str, str]:
    """keyword -> source of the first ``callee(...)`` call inside function ``func_name``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and ast.unparse(call.func) == callee:
                    return {k.arg: ast.unparse(k.value) for k in call.keywords}
    raise AssertionError(f"no {callee}(...) in {func_name}")


def test_training_env_is_built_exactly_like_the_reference_trainer():
    """The reference Trainer builds its environment inline; tracking.train_env_args must
    pass the same arguments, or DeepMimic and SuperMimic would train in different worlds."""
    from quadruped_rl.algorithms.supermimic import tracking

    _, root = _backend()
    trainer = ast.parse((root / "src/supermimic/training/loop.py").read_text())
    ours = ast.parse(Path(tracking.__file__).read_text())
    theirs = _call_kwargs(trainer, "__init__", "SuperMimicEnvCfgArgs")
    mine = _call_kwargs(ours, "train_env_args", "SuperMimicEnvCfgArgs")
    assert theirs.pop("seed") == "cfg.seed + 1000 * self.rank" and mine.pop("seed") == "cfg.seed"
    assert mine.pop("record_transitions") == "record_transitions"   # the reference's default: True
    assert mine == theirs
    assert (_call_kwargs(trainer, "__init__", "discover_npz")
            == _call_kwargs(ours, "train_env_args", "discover_npz"))


def test_task_translates_to_the_same_environment_as_supermimic(tmp_path):
    backend, _ = _backend()
    from quadruped_rl.robots import load_robot

    task = load_group("tasks", "motion_tracking")
    agent = DeepMimicAgent(robot=load_robot("go1"), task_cfg=task,
                           algo_cfg=load_group("algorithms", "deepmimic"), run_dir=tmp_path / "dm",
                           seed=3)
    dm = agent.train_config(num_envs=64)
    sm_algo = load_group("algorithms", "supermimic")
    sm = backend.build_train_config("go1", task, {**sm_algo, "num_envs": 64}, tmp_path / "sm", "cpu", 3)
    assert dm.data == sm.data                                    # clips, groups, manifest
    assert (dm.robot, dm.seed, dm.env.num_envs) == ("go1", 3, 64)
    assert dm.env.action_noise_std == 0.0 and sm.env.action_noise_std == sm_algo["env"]["action_noise_std"]
    dm.env.action_noise_std = sm.env.action_noise_std
    dm.env.replay_capacity = sm.env.replay_capacity
    assert dm.env == sm.env          # reset sampling, terminations, noise, randomization, episode length


def test_motion_sampler_on_the_clip_library(tmp_path):
    """Frames are valid starts (a following frame in the same clip) and states match the clip."""
    backend, _ = _backend()
    from quadruped_rl.robots import load_robot

    agent = DeepMimicAgent(robot=load_robot("go1"), task_cfg=load_group("tasks", "motion_tracking"),
                           algo_cfg=load_group("algorithms", "deepmimic"), run_dir=tmp_path)
    files = _clips_or_skip(agent, "holdout")[:3]
    backend.ensure_importable()
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.robots import get_robot_spec

    from quadruped_rl.algorithms.supermimic.tracking import MotionSampler

    spec = get_robot_spec("go1")
    home = backend._spec_home(spec.entity_cfg_fn(False), spec.joint_names)
    lib = MotionLibrary(files, spec.tracked_bodies, torch.tensor([h or 0.0 for h in home]),
                        device="cpu", root_body=spec.root_body, robot="go1")
    sampler = MotionSampler(lib)
    frames = sampler.sample_frames(2000, generator=torch.Generator().manual_seed(0))
    clip = lib.frame_clip[frames]
    assert (frames >= lib.clip_start[clip]).all() and (frames + 1 <= lib.clip_last[clip]).all()
    assert len(set(clip.tolist())) == lib.num_clips             # every clip gets starts
    state = sampler.reference_state(frames[:5])
    assert state.joint_pos.shape == (5, 12) and state.root_quat_w.shape == (5, 4)
    qpos, qvel = sampler.reference_qpos_qvel(frames[:5])
    assert qpos.shape == (5, 19) and qvel.shape == (5, 18)
    import numpy as np

    first = np.load(files[0])
    assert np.allclose(sampler.reference_state(lib.clip_start[:1]).joint_pos[0].numpy(),
                       first["joint_pos"][0], atol=1e-5)
    with pytest.raises(ValueError, match="no environment"):
        sampler.state_dict()


# ---- simulated (CPU) ----------------------------------------------------------------

def _sim_agent(tmp_path, **env_overrides):
    pytest.importorskip("mjlab")
    _backend()
    from quadruped_rl.robots import load_robot
    from quadruped_rl.utils.config import deep_merge

    task = load_group("tasks", "motion_tracking")
    task["motions"]["limit"] = 4
    algo = deep_merge(load_group("algorithms", "deepmimic"), {**TINY_PPO, "env": env_overrides})
    agent = DeepMimicAgent(robot=load_robot("go1"), task_cfg=task, algo_cfg=algo,
                           run_dir=tmp_path, device="cpu")
    _clips_or_skip(agent, task["motions"]["train_groups"])
    return agent


@pytest.mark.sim
def test_episodes_start_on_the_reference_and_end_when_the_robot_strays(tmp_path):
    agent = _sim_agent(tmp_path, obs_noise=False, reset_noise=False, domain_randomization=False)
    env = agent.make_env(4)
    try:
        raw = env.unwrapped
        obs, _ = env.reset()
        cmd = raw.command_manager.get_term("motion")
        # 10 frames x (q 12, qd 12, ang vel 3, gravity 3, last action 12) + 10 x 34 goal features
        assert obs["actor"].shape == (4, 420 + 340)
        # Reference-state initialization: the robot is put on the frame each episode starts
        # at (the reset already advanced the command to the next frame, its first target).
        start = cmd.library.ref(cmd.frame - 1)
        limits = cmd.robot.data.soft_joint_pos_limits
        ref_q = torch.clip(start.joint_pos, limits[..., 0], limits[..., 1])
        assert torch.allclose(cmd.robot.data.joint_pos, ref_q, atol=1e-4)
        assert torch.allclose(raw.sim.data.qpos[:, :3], start.root_pos_w + raw.scene.env_origins,
                              atol=1e-4)
        # Starting on the reference, the first step earns most of the maximum tracking reward.
        _, reward, terminated, _, _ = env.step(torch.zeros(4, 12))
        max_r = _max_step_reward(raw)
        assert not terminated.any()
        assert (reward > 0.5 * max_r).all() and (reward <= max_r + 1e-6).all()
        # Early termination: a trunk far above the reference ends the episode as a failure.
        pos = raw.sim.data.qpos[:, :3].clone()
        pos[:, 2] += 1.0
        state = torch.cat([pos, raw.sim.data.qpos[:, 3:7], torch.zeros(4, 6)], dim=1)
        cmd.robot.write_root_state_to_sim(state)
        _, _, terminated, truncated, _ = env.step(torch.zeros(4, 12))
        assert terminated.all() and not truncated.any()
    finally:
        env.close()


def _max_step_reward(raw) -> float:
    from supermimic.envs.mdp.rewards import TrackingRewardCfg
    from supermimic.robots import get_robot_spec

    spec = get_robot_spec("go1")
    return TrackingRewardCfg.from_robot(spec, dt=spec.sim.control_dt).max_step_reward


@pytest.mark.sim
def test_deepmimic_trains_resumes_plays_and_benchmarks(tmp_path):
    import numpy as np

    from quadruped_rl.algorithms.checkpoints import checkpoint_format, load_checkpoint

    agent = _sim_agent(tmp_path)
    agent.train(num_envs=8, max_iterations=2)
    ckpt = tmp_path / "checkpoints" / "model_2.pt"
    saved = load_checkpoint(ckpt)
    assert checkpoint_format(saved) == "deepmimic" and saved["deepmimic"]["sampler"]["bin_return"].numel()

    resumed = _sim_agent(tmp_path)
    resumed.train(num_envs=8, max_iterations=1, resume=ckpt)
    assert resumed.iteration == 3 and (tmp_path / "checkpoints" / "model_3.pt").exists()

    # Play: the evaluation environment SuperMimic plays in, through PPO's deterministic actor.
    files = sorted(_clips_or_skip(agent, "holdout"), key=lambda f: len(np.load(f)["joint_pos"]))[:2]
    player = _sim_agent(tmp_path)
    env = player.make_env(2, play=True, clips=files)
    try:
        viewer_env, policy = player.make_policy(env, tmp_path / "checkpoints" / "model_3.pt")
        for _ in range(3):
            actions = policy(viewer_env.get_observations())
            viewer_env.step(actions)
        assert actions.shape == (2, 12) and torch.isfinite(actions).all()
        assert torch.equal(policy(viewer_env.get_observations()), policy(viewer_env.get_observations()))
    finally:
        env.close()

    # The reference benchmark scores this policy (same columns as a SuperMimic run).
    player.clip_files = lambda groups=None, names=None, mirrors=False: files
    summary = player.benchmark(tmp_path / "checkpoints" / "model_3.pt", "holdout", num_envs=2,
                               out=tmp_path / "bench.json")
    assert summary["clips"] == 2 and 0 <= summary["failed"] <= 2
    assert {"mean_norm_return", "step_weighted_norm_return", "worst"} <= set(summary)
    assert (tmp_path / "bench.json").exists()
