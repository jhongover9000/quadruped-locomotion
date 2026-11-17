"""AMP agent (Peng et al., 2021): PPO on a velocity task whose reward mixes the task reward
with a style reward from a discriminator trained alongside the policy,

    r = w_task * r_task + w_style * dt * r_style

r_task is the environment's reward, which mjlab already integrates over the control step
dt (each term is weight * value * dt), and r_style is in [0, 1] per step (discriminator.py).
Multiplying r_style by dt puts both on the same per-second scale, so the weights mean
what they mean in the paper, where both rewards are per step in [0, 1].

Each iteration:
  1. Rollout as in PPO. After every step the features (features.py) of the last
     ``disc_obs_frames`` states form a window; its style reward joins the task reward,
     and the window is kept for the discriminator. A window that crosses a reset (the
     environment was done this step, so its state is already the post-reset one) is
     dropped and gets no style reward.
  2. PPO update.
  3. Discriminator update: the rollout's windows enter a replay buffer; then as many
     steps as PPO took, each on a PPO-minibatch-sized batch of clip windows (motion
     buffer) and policy windows (replay buffer). The input normalizer is then refreshed
     with the rollout's windows and as many clip windows.

The checkpoint is PPO's plus the discriminator (with its normalizer) and its optimizer;
the replay buffer is not saved and refills after --continue-run. play.py, record.py and
evaluate.py load only the actor, as for PPO.
"""

from __future__ import annotations

import torch

from quadruped_rl.algorithms.amp.discriminator import Discriminator
from quadruped_rl.algorithms.amp.features import clip_features, env_features, feature_dim
from quadruped_rl.algorithms.amp.motion_buffer import MotionBuffer, ReplayBuffer
from quadruped_rl.algorithms.ppo.agent import PPOAgent
from quadruped_rl.algorithms.ppo.storage import RolloutStorage


class AMPAgent(PPOAgent):
    task_kinds = ("velocity",)
    checkpoint_kind = "amp"
    motions: MotionBuffer | None = None
    replay: ReplayBuffer | None = None

    # ---- setup ----------------------------------------------------------------------

    def build(self, actor_dim: int, critic_dim: int, action_dim: int) -> None:
        super().build(actor_dim, critic_dim, action_dim)
        a = self.algo_cfg["amp"]
        self.num_frames = int(a["disc_obs_frames"])
        dim = self.num_frames * feature_dim(self.robot.num_joints)
        self.discriminator = Discriminator(dim, a["disc_hidden"]).to(self.device)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                               lr=float(a["disc_learning_rate"]))

    def style_clips(self) -> list:
        """The clips that define the style: manifest groups and categories from amp.motions,
        looked up like SuperMimic's library (algorithms/supermimic/backend.py)."""
        from quadruped_rl.algorithms.supermimic import backend
        from quadruped_rl.motions.format import validate_clip
        from quadruped_rl.motions.motion_lib import MotionLibrary

        m = self.algo_cfg["amp"]["motions"]
        root = backend.find_root(self.algo_cfg)
        directory = backend.resolve_path(m["dir"], root, self.robot.name)
        lib = MotionLibrary(directory, backend.resolve_path(m.get("manifest"), root, self.robot.name))
        categories = set(m["categories"]) if m.get("categories") else None
        category = {r["name"]: r.get("category") for r in lib.rows}
        names = [n for n in lib.names(m.get("groups"), exclude=m.get("exclude"))
                 if categories is None or category.get(n) in categories]
        if not names:
            raise FileNotFoundError(
                f"No style clips in {directory} for groups={m.get('groups')} "
                f"categories={m.get('categories')} (see amp.motions in configs/algorithms/amp.yaml)")
        clips = [lib.load(n) for n in names]
        for clip in clips:
            validate_clip(clip, self.robot.joint_names)
        return clips

    def _load_motions(self, step_dt: float) -> MotionBuffer:
        clips = self.style_clips()
        wrong = [c.name for c in clips if abs(c.fps * step_dt - 1.0) > 1e-6]
        if wrong:
            raise ValueError(f"AMP needs clips at the control rate ({1.0 / step_dt:g} Hz); "
                             f"{len(wrong)} are not, e.g. {wrong[0]}")
        m = self.algo_cfg["amp"]["motions"]
        buffer = MotionBuffer([clip_features(c, self.robot.base_body) for c in clips],
                              self.num_frames, self.device,
                              float(m.get("clip_weight_alpha", 0.5)))
        print(f"[amp] {buffer.num_clips} style clips, {buffer.features.shape[0]} frames", flush=True)
        return buffer

    # ---- PPO extension points -------------------------------------------------------

    def _on_learn_start(self, env) -> None:
        a = self.algo_cfg["amp"]
        # The control step: a clip window must span it, and it scales the style reward.
        self.step_dt = float(env.step_dt)
        if self.motions is None:
            self.motions = self._load_motions(self.step_dt)
        if self.replay is None:
            self.replay = ReplayBuffer(int(a["replay_buffer_size"]), self.motions.dim, self.device)
        if self.discriminator.norm.count == 0:      # fresh run: start from the clips' statistics
            self.discriminator.norm.update(self.motions.sample(100_000))
        self.w_task, self.w_style = float(a["task_reward_weight"]), float(a["style_reward_weight"])
        features = env_features(env)
        self._history = features.unsqueeze(1).repeat(1, self.num_frames, 1)   # [N, frames, D]
        self._age = torch.ones(env.num_envs, dtype=torch.long, device=self.device)
        self._windows: list[torch.Tensor] = []
        self._sums = {"style": 0.0, "task": 0.0, "valid": 0, "steps": 0}

    def _process_rewards(self, env, rewards, terminated, truncated):
        done = terminated | truncated
        features = env_features(env)
        self._history = torch.cat([self._history[:, 1:], features.unsqueeze(1)], dim=1)
        # States since the last reset, the current one included: a done environment's
        # current state is already its post-reset state.
        self._age = torch.where(done, torch.ones_like(self._age), self._age + 1)
        valid = (self._age >= self.num_frames) & ~done
        style = torch.zeros_like(rewards)
        if valid.any():
            windows = self._history[valid].flatten(1)
            style[valid] = self.discriminator.style_reward(windows)
            self._windows.append(windows)
        if done.any():                              # the window restarts at the new episode
            self._history[done] = features[done].unsqueeze(1)
        s = self._sums
        s["style"] += style[valid].sum().item()
        s["task"] += rewards.sum().item()
        s["valid"] += int(valid.sum().item())
        s["steps"] += rewards.numel()
        return self.w_task * rewards + self.w_style * self.step_dt * style

    def _extra_update(self, storage: RolloutStorage) -> dict[str, float]:
        c, a = self.algo_cfg, self.algo_cfg["amp"]
        policy = torch.cat(self._windows) if self._windows else None
        self._windows = []
        if policy is not None:
            self.replay.insert(policy)
        if len(self.replay) == 0:
            return {}
        steps = int(c["num_epochs"]) * int(c["num_minibatches"])
        batch = storage.num_steps * storage.num_envs // int(c["num_minibatches"])
        totals = {"loss": 0.0, "lsgan": 0.0, "grad_penalty": 0.0, "d_clip": 0.0, "d_policy": 0.0}
        for _ in range(steps):
            loss, info = self.discriminator.loss(self.motions.sample(batch), self.replay.sample(batch),
                                                 float(a["grad_penalty"]))
            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()
            totals["loss"] += loss.item()
            for k, v in info.items():
                totals[k] += v
        if policy is not None:
            self.discriminator.norm.update(torch.cat([policy, self.motions.sample(policy.shape[0])]))
        return {
            "AMP/disc_loss": totals["loss"] / steps,
            "AMP/disc_lsgan_loss": totals["lsgan"] / steps,
            "AMP/disc_grad_penalty": totals["grad_penalty"] / steps,
            "AMP/disc_clip_mean": totals["d_clip"] / steps,
            "AMP/disc_policy_mean": totals["d_policy"] / steps,
        }

    def _extra_scalars(self) -> dict[str, float]:
        s = self._sums
        out = {"AMP/mean_task_reward": s["task"] / max(s["steps"], 1),
               "AMP/mean_style_reward": s["style"] / max(s["valid"], 1),
               "AMP/valid_window_fraction": s["valid"] / max(s["steps"], 1)}
        self._sums = {"style": 0.0, "task": 0.0, "valid": 0, "steps": 0}
        return out

    def _extra_state(self) -> dict:
        return {"discriminator": self.discriminator.state_dict(),
                "disc_optimizer": self.disc_optimizer.state_dict()}

    def _load_extra_state(self, ckpt: dict) -> None:
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
