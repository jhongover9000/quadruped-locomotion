"""AMP pieces on CPU without the simulator: the discriminator features, the clip and replay
buffers, the discriminator loss, reset masking and the checkpoint. The last test (marked
sim) writes a clip frame into the simulator and compares features from both paths."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quadruped_rl.algorithms.amp.agent import AMPAgent  # noqa: E402
from quadruped_rl.algorithms.amp.discriminator import Discriminator  # noqa: E402
from quadruped_rl.algorithms.amp.features import (  # noqa: E402
    amp_features,
    clip_features,
    env_features,
    feature_dim,
    quat_rotate_inverse,
)
from quadruped_rl.algorithms.amp.motion_buffer import MotionBuffer, ReplayBuffer  # noqa: E402
from quadruped_rl.motions.format import Clip  # noqa: E402

J = 12
D = feature_dim(J)


def yaw_quat(yaw):
    return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def random_quats(n, gen):
    q = torch.randn(n, 4, generator=gen)
    return q / q.norm(dim=-1, keepdim=True)


# ---- features -----------------------------------------------------------------------

def test_rotation_matches_mujoco():
    mujoco = pytest.importorskip("mujoco")
    gen = torch.Generator().manual_seed(0)
    q, v = random_quats(20, gen).double(), torch.randn(20, 3, generator=gen).double()
    expected = []
    for qi, vi in zip(q.numpy(), v.numpy()):
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, qi)
        expected.append(mat.reshape(3, 3).T @ vi)
    assert torch.allclose(quat_rotate_inverse(q, v), torch.tensor(np.array(expected)), atol=1e-10)


def test_features_layout_and_heading_invariance():
    gen = torch.Generator().manual_seed(1)
    q_j, qd_j = torch.randn(J, generator=gen), torch.randn(J, generator=gen)
    lin, ang = torch.tensor([0.7, 0.1, 0.0]), torch.tensor([0.0, 0.2, 0.5])
    f = amp_features(q_j, qd_j, yaw_quat(0.0), lin, ang, torch.tensor(0.3))
    assert f.shape == (D,)
    assert torch.equal(f[:J], q_j) and torch.equal(f[J:2 * J], qd_j)
    assert torch.allclose(f[2 * J:], torch.tensor([0.7, 0.1, 0.0, 0.0, 0.2, 0.5, 0, 0, -1, 0.3]))

    # The same motion facing another way (world velocities rotated with it) looks the same.
    yaw = 2.0
    rot = torch.tensor([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    g = amp_features(q_j, qd_j, yaw_quat(yaw), rot @ lin, rot @ ang, torch.tensor(0.3))
    assert torch.allclose(f, g, atol=1e-6)

    # Pitched nose-down by 90 degrees: gravity points along the body's +x.
    pitch = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])
    h = amp_features(q_j, qd_j, pitch, lin, ang, torch.tensor(0.3))
    assert torch.allclose(h[2 * J + 6:2 * J + 9], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


def make_clip(frames=30, yaw=0.0, start_x=0.0, name="c"):
    t = np.arange(frames) / 50.0
    bodies = ["trunk", "FR_hip"]
    quat = np.zeros((frames, 2, 4))
    quat[..., 0], quat[..., 3] = math.cos(yaw / 2), math.sin(yaw / 2)
    vel = np.zeros((frames, 2, 3))
    vel[..., 0], vel[..., 1] = 0.5 * math.cos(yaw), 0.5 * math.sin(yaw)
    pos = np.zeros((frames, 2, 3))
    pos[..., 0] = start_x + (0.5 * t * math.cos(yaw))[:, None]
    pos[..., 1] = (0.5 * t * math.sin(yaw))[:, None]
    pos[..., 2] = 0.3
    joint_pos = np.tile(np.linspace(-1, 1, J), (frames, 1)) + np.sin(t)[:, None]
    return Clip(name, 50.0, [f"j{i}" for i in range(J)], bodies, joint_pos,
                np.cos(t)[:, None].repeat(J, 1), pos, quat, vel, np.zeros_like(vel))


def test_clip_features_ignore_position_and_heading():
    a = clip_features(make_clip(yaw=0.0), "trunk")
    b = clip_features(make_clip(yaw=1.3, start_x=5.0), "trunk")
    assert a.shape == (30, D) and a.dtype == torch.float32
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(a[:, 2 * J:2 * J + 3], torch.tensor([0.5, 0.0, 0.0]).expand(30, 3), atol=1e-6)
    assert torch.allclose(a[:, -1], torch.full((30,), 0.3))


# ---- buffers ------------------------------------------------------------------------

def tagged_clip(clip_id, frames):
    """Features whose first two numbers say which clip and which frame they come from."""
    f = torch.zeros(frames, 4)
    f[:, 0], f[:, 1] = clip_id, torch.arange(frames)
    return f


def test_motion_buffer_windows_stay_inside_one_clip():
    torch.manual_seed(0)
    buffer = MotionBuffer([tagged_clip(0, 5), tagged_clip(1, 2), tagged_clip(2, 40)], num_frames=3)
    assert buffer.num_clips == 2 and buffer.dim == 12           # the 2-frame clip is too short
    w = buffer.sample(5000).view(5000, 3, 4)
    assert torch.all(w[:, :, 0] == w[:, :1, 0])                  # one clip per window
    assert torch.all(w[:, 1:, 1] - w[:, :-1, 1] == 1)            # consecutive frames
    assert set(w[:, 0, 0].tolist()) == {0.0, 2.0}


@pytest.mark.parametrize("alpha,expected", [(0.0, 0.5), (1.0, 10 / 50)])
def test_motion_buffer_clip_weights(alpha, expected):
    """alpha = 0: every clip equally often; alpha = 1: every frame equally often."""
    torch.manual_seed(0)
    buffer = MotionBuffer([tagged_clip(0, 10), tagged_clip(1, 40)], num_frames=1,
                          clip_weight_alpha=alpha)
    share = (buffer.sample(40000)[:, 0] == 0).float().mean().item()
    assert abs(share - expected) < 0.01


def test_replay_buffer_keeps_the_newest():
    buffer = ReplayBuffer(capacity=5, dim=1)
    buffer.insert(torch.arange(3.0)[:, None])
    assert len(buffer) == 3 and set(buffer.sample(200)[:, 0].tolist()) == {0.0, 1.0, 2.0}
    buffer.insert(torch.arange(3.0, 7.0)[:, None])
    assert len(buffer) == 5 and set(buffer.sample(500)[:, 0].tolist()) == {2.0, 3.0, 4.0, 5.0, 6.0}
    buffer.insert(torch.arange(10.0, 20.0)[:, None])             # more than fits at once
    assert set(buffer.sample(500)[:, 0].tolist()) == {15.0, 16.0, 17.0, 18.0, 19.0}


# ---- discriminator ------------------------------------------------------------------

def test_style_reward_formula():
    disc = Discriminator(2, [8])
    for target, reward in ((1.0, 1.0), (-1.0, 0.0), (0.0, 0.75), (3.0, 0.0)):
        torch.nn.init.zeros_(disc.net[-1].weight)
        torch.nn.init.constant_(disc.net[-1].bias, target)
        assert disc.style_reward(torch.zeros(1, 2)).item() == pytest.approx(reward)


def test_discriminator_learns_to_separate():
    torch.manual_seed(0)
    disc = Discriminator(4, [32, 32])
    clip = torch.randn(2000, 4) + torch.tensor([2.0, 0, 0, 0])
    policy = torch.randn(2000, 4) - torch.tensor([2.0, 0, 0, 0])
    disc.norm.update(torch.cat([clip, policy]))
    opt = torch.optim.Adam(disc.parameters(), lr=1e-3)
    for _ in range(300):
        loss, info = disc.loss(clip, policy, grad_penalty=10.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # The penalty limits D's slope, so the two sides stay well inside the +-1 targets.
    assert info["d_clip"] > 0.15 and info["d_policy"] < -0.15 and info["grad_penalty"] > 0
    assert disc.style_reward(clip).mean() > disc.style_reward(policy).mean() + 0.2


# ---- the agent: reset masking, rewards, checkpoints ---------------------------------

ROBOT = SimpleNamespace(name="toy", num_joints=J, base_body="trunk",
                        joint_names=[f"j{i}" for i in range(J)])
ALGO = {"steps_per_env": 4, "learning_rate": 1e-3, "num_epochs": 1, "num_minibatches": 2,
        "clip_ratio": 0.2, "gamma": 0.99, "gae_lambda": 0.95, "entropy_coef": 0.0,
        "value_loss_coef": 1.0, "max_grad_norm": 1.0, "init_noise_std": 0.5,
        "actor_hidden": [16], "critic_hidden": [16], "activation": "elu", "save_interval": 1000,
        "amp": {"disc_hidden": [32], "disc_obs_frames": 2, "disc_learning_rate": 1e-3,
                "grad_penalty": 10.0, "replay_buffer_size": 1000, "task_reward_weight": 0.5,
                "style_reward_weight": 0.5}}


class FakeScene(dict):
    env_origins: torch.Tensor


class FakeRobotEnv:
    """Batched stand-in for the mjlab environment: just what AMP and PPO read. Each
    environment's joint positions all equal its episode count, so a window's episodes
    can be read back from its features."""

    def __init__(self, num_envs=3, max_steps=6):
        self.num_envs, self.num_actions, self.max_episode_length = num_envs, J, max_steps
        self.step_dt = 0.02
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.episode = torch.zeros(num_envs)
        self.time = torch.zeros(num_envs)
        self.done_next = torch.zeros(num_envs, dtype=torch.bool)    # tests force resets with this
        self.data = SimpleNamespace()
        self.scene = FakeScene(robot=SimpleNamespace(data=self.data))
        self.scene.env_origins = torch.zeros(num_envs, 3)
        self._refresh()

    def _refresh(self):
        n = self.num_envs
        self.data.joint_pos = self.episode[:, None].repeat(1, J)
        self.data.joint_vel = self.time[:, None].repeat(1, J)
        self.data.root_link_pos_w = torch.zeros(n, 3) + torch.tensor([0.0, 0.0, 0.3])
        self.data.root_link_quat_w = torch.tensor([1.0, 0, 0, 0]).repeat(n, 1)
        self.data.root_link_lin_vel_w = torch.zeros(n, 3)
        self.data.root_link_ang_vel_w = torch.zeros(n, 3)

    def _obs(self):
        o = torch.cat([self.episode[:, None], self.time[:, None]], dim=1)
        return {"actor": o, "critic": o}

    def reset(self):
        return self._obs(), {}

    def step(self, action):
        self.time += 1
        self.episode_length_buf += 1
        truncated = (self.episode_length_buf >= self.max_episode_length) | self.done_next
        terminated = torch.zeros_like(truncated)
        self.episode[truncated] += 1                # reset in place, as mjlab does
        self.time[truncated] = 0
        self.episode_length_buf[truncated] = 0
        self.done_next[:] = False
        self._refresh()
        return self._obs(), torch.full((self.num_envs,), 0.01), terminated, truncated, {"log": {}}


def make_agent(tmp_path, env):
    agent = AMPAgent(robot=ROBOT, task_cfg={}, algo_cfg=ALGO, run_dir=tmp_path)
    agent.build(actor_dim=2, critic_dim=2, action_dim=J)
    # Stand-in for the clip library (style_clips() needs the real one).
    agent.motions = MotionBuffer([torch.randn(50, D)], num_frames=2)
    agent._on_learn_start(env)
    return agent


def test_windows_across_a_reset_are_dropped(tmp_path):
    torch.manual_seed(0)
    env = FakeRobotEnv(num_envs=3)
    agent = make_agent(tmp_path, env)

    def step():
        _, r, term, trunc, _ = env.step(torch.zeros(3, J))
        return agent._process_rewards(env, r, term, trunc), term | trunc

    reward, done = step()
    assert not done.any() and len(agent._windows) == 1 and agent._windows[0].shape == (3, 2 * D)
    env.done_next[1] = True
    reward, done = step()
    assert done.tolist() == [False, True, False]
    assert reward[1].item() == pytest.approx(0.5 * 0.01)      # task reward only: no style reward
    assert (reward[[0, 2]] > 0.5 * 0.01).all()                # the others got some style reward
    assert agent._windows[-1].shape == (2, 2 * D)             # env 1's window never stored
    reward, done = step()                                     # env 1's first post-reset window
    assert agent._windows[-1].shape == (3, 2 * D) and reward[1] > 0.5 * 0.01

    windows = torch.cat(agent._windows).view(-1, 2, D)
    # Every stored window lies in one episode (joint positions carry the episode count) ...
    assert torch.all(windows[:, 0, 0] == windows[:, 1, 0])
    # ... and consists of consecutive states (joint velocities carry the step in the episode).
    assert torch.all(windows[:, 1, J] - windows[:, 0, J] == 1)
    assert agent._extra_scalars()["AMP/valid_window_fraction"] == pytest.approx(8 / 9)


def test_learn_runs_and_checkpoint_restores_the_discriminator(tmp_path):
    torch.manual_seed(0)
    env = FakeRobotEnv(num_envs=8)
    agent = AMPAgent(robot=ROBOT, task_cfg={}, algo_cfg=ALGO, run_dir=tmp_path)
    agent.build(actor_dim=2, critic_dim=2, action_dim=J)
    agent.motions = MotionBuffer([torch.randn(50, D)], num_frames=2)
    agent.learn(env, max_iterations=3)

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(tmp_path))
    events.Reload()
    for tag in ("AMP/disc_loss", "AMP/disc_clip_mean", "AMP/disc_policy_mean",
                "AMP/mean_style_reward", "AMP/mean_task_reward"):
        assert len(events.Scalars(tag)) == 3, tag
    # 12 rollout steps in 8 environments, each reset twice in that time (every 6 steps).
    assert len(agent.replay) == 12 * 8 - 2 * 8

    path = tmp_path / "checkpoints" / "model_3.pt"
    ckpt = torch.load(path, weights_only=True)
    assert {"discriminator", "disc_optimizer", "model"} <= set(ckpt)
    assert ckpt["discriminator"]["norm.count"] > 0

    restored = AMPAgent(robot=ROBOT, task_cfg={}, algo_cfg=ALGO, run_dir=tmp_path)
    restored.build(actor_dim=2, critic_dim=2, action_dim=J)
    restored.load(path)
    for k, v in agent.discriminator.state_dict().items():
        assert torch.equal(restored.discriminator.state_dict()[k], v), k
    assert restored.disc_optimizer.state_dict()["state"].keys() == \
        agent.disc_optimizer.state_dict()["state"].keys()
    assert restored.iteration == 3

    # play.py / record.py / evaluate.py: the actor alone loads; PPO refuses the file.
    player = AMPAgent(robot=ROBOT, task_cfg={}, algo_cfg=ALGO, run_dir=tmp_path)
    player.build(actor_dim=2, critic_dim=2, action_dim=J)
    player.load(path, actor_only=True)
    obs = torch.randn(4, 2)
    assert torch.allclose(player.act(obs), agent.act(obs))
    from quadruped_rl.algorithms.ppo.agent import PPOAgent

    ppo = PPOAgent(robot=ROBOT, task_cfg={}, algo_cfg=ALGO, run_dir=tmp_path)
    ppo.build(actor_dim=2, critic_dim=2, action_dim=J)
    with pytest.raises(ValueError, match="written by 'amp'"):
        ppo.load(path, actor_only=True)
    ppo.save(tmp_path / "ppo.pt")
    with pytest.raises(ValueError, match="written by 'ppo'"):
        player.load(tmp_path / "ppo.pt")


# ---- the simulator gives the clip's features ----------------------------------------

@pytest.mark.sim
def test_clip_frame_in_the_simulator_gives_the_same_features():
    pytest.importorskip("mjlab")
    from quadruped_rl.algorithms.supermimic import backend
    from quadruped_rl.envs.builder import make_env
    from quadruped_rl.motions.motion_lib import MotionLibrary
    from quadruped_rl.robots import load_robot
    from quadruped_rl.utils.config import load_group

    root = backend.find_root()
    directory = backend.resolve_path("data/npz/go1", root, "go1")
    if not directory.is_dir():
        pytest.skip("Go1 motion library not found (see docs/supermimic.md)")
    lib = MotionLibrary(directory)
    clip = lib.load(next(r["name"] for r in lib.kept() if r.get("category") == "trot"))
    robot = load_robot("go1")
    i = clip.body_names.index(robot.base_body)
    frames = [5, clip.frames // 2]                      # one frame per environment
    expected = clip_features(clip, robot.base_body)[frames]

    env = make_env(robot, load_group("tasks", "flat_velocity"), num_envs=len(frames), device="cpu")
    try:
        env.reset()
        entity = env.scene["robot"]

        def t(x):
            return torch.as_tensor(np.asarray(x, dtype=np.float32))

        state = torch.cat([t(clip.body_pos_w[frames, i]), t(clip.body_quat_w[frames, i]),
                           t(clip.body_lin_vel_w[frames, i]), t(clip.body_ang_vel_w[frames, i])], dim=-1)
        entity.write_root_state_to_sim(state)
        entity.write_joint_state_to_sim(t(clip.joint_pos[frames]), t(clip.joint_vel[frames]))
        env.sim.forward()
        got = env_features(env)
    finally:
        env.close()
    assert torch.allclose(got.cpu(), expected, atol=1e-4), (got.cpu() - expected).abs().max()
