"""PPO agent: clipped surrogate objective with GAE advantages.

Each iteration:
  1. Rollout: every environment runs T steps with the current (stochastic) policy.
  2. Advantages: GAE over the rollout (storage.py).
  3. Update: several epochs of minibatch gradient steps on

        L = L_policy + c_v * L_value - c_e * entropy

        L_policy = -E[ min( rho * A, clip(rho, 1 - eps, 1 + eps) * A ) ],
        rho = pi_new(a | o) / pi_old(a | o)

     The clip stops one update from moving the policy far from the one that collected
     the data. The learning rate adapts to keep the KL divergence between old and new
     policies near a target.

Asymmetric actor-critic: the actor reads the "actor" observations, the critic the
privileged "critic" observations.

Subclasses extend the loop through the hooks under "extension points": they change the
reward after each step, add their own optimizer steps to the update, add scalars to the
log and add state to the checkpoint. AMP uses all of them; DeepMimic only the checkpoint
hooks. The hooks do nothing here.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import torch

from quadruped_rl.algorithms.base import Agent
from quadruped_rl.algorithms.ppo.networks import ActorCritic
from quadruped_rl.algorithms.ppo.storage import RolloutStorage

CSV_COLUMNS = ["iteration", "env_steps", "mean_return", "mean_episode_length", "policy_loss",
               "value_loss", "entropy", "learning_rate", "action_std", "steps_per_s"]


class PPOAgent(Agent):
    model: ActorCritic | None = None
    checkpoint_kind = "ppo"         # the format name in algorithms/checkpoints.py

    # ---- setup ----------------------------------------------------------------------

    def build(self, actor_dim: int, critic_dim: int, action_dim: int) -> None:
        c = self.algo_cfg
        self.model = ActorCritic(actor_dim, critic_dim, action_dim, c["actor_hidden"],
                                 c["critic_hidden"], c["activation"], c["init_noise_std"]).to(self.device)
        self.learning_rate = float(c["learning_rate"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.iteration = 0

    def _build_for_env(self, env) -> None:
        obs, _ = env.reset()
        action_dim = env.action_manager.total_action_dim if hasattr(env, "action_manager") \
            else env.num_actions
        self.build(obs["actor"].shape[1], obs["critic"].shape[1], action_dim)

    # ---- training -------------------------------------------------------------------

    def train(self, num_envs: int | None = None, max_iterations: int | None = None,
              resume: Path | None = None) -> None:
        env = self.make_env(num_envs or self.algo_cfg["num_envs"])
        self._build_for_env(env)
        if resume is not None:
            self.load(resume)
        self.learn(env, max_iterations or self.algo_cfg["max_iterations"])
        env.close()

    def learn(self, env, max_iterations: int) -> None:
        """Main loop. ``env`` follows QuadrupedEnv's interface (see envs/base.py)."""
        from quadruped_rl.utils.logging import RunLogger

        c = self.algo_cfg
        T, N = int(c["steps_per_env"]), env.num_envs
        gamma, lam = float(c["gamma"]), float(c["gae_lambda"])
        if self.model is None:
            self._build_for_env(env)
        model = self.model

        obs, _ = env.reset()
        # Desynchronize episodes so resets (and their reward dips) don't all happen together.
        env.episode_length_buf = torch.randint_like(env.episode_length_buf,
                                                    high=int(env.max_episode_length))
        actor_obs, critic_obs = obs["actor"].to(self.device), obs["critic"].to(self.device)
        if model.actor_norm.count == 0:
            actor_obs, critic_obs = self._warm_up_normalizers(env, actor_obs, critic_obs, T)
        if self.iteration > 0 and hasattr(env, "common_step_counter"):
            # Resumed (--resume / --continue-run): put the env step counter where the run
            # left off, so step-based curricula (reward_ramps) don't start over. Set after
            # the reset above, so the terrain curriculum's first-reset freeze still applies.
            env.common_step_counter = self.iteration * T
        storage = RolloutStorage(N, T, actor_obs.shape[1], critic_obs.shape[1],
                                 model.log_std.shape[0], self.device)
        ep_return = torch.zeros(N, device=self.device)
        ep_length = torch.zeros(N, device=self.device)
        logger = RunLogger(self.run_dir, CSV_COLUMNS)
        save_interval = int(c.get("save_interval", 100))
        end = self.iteration + max_iterations
        max_hours, t_begin = c.get("max_hours"), time.perf_counter()
        self._on_learn_start(env)

        while self.iteration < end:
            t_start = time.perf_counter()
            finished_returns, finished_lengths = [], []
            env_log = defaultdict(list)

            # 1. Rollout
            model.eval()
            with torch.no_grad():
                for _ in range(T):
                    actions, log_probs, mean, std = model.act(actor_obs)
                    values = model.evaluate(critic_obs)
                    obs, rewards, terminated, truncated, extras = env.step(actions)
                    rewards = rewards.to(self.device)
                    dones = (terminated | truncated).to(self.device)

                    ep_return += rewards
                    ep_length += 1
                    if dones.any():
                        finished_returns.append(ep_return[dones].clone())
                        finished_lengths.append(ep_length[dones].clone())
                        ep_return[dones] = 0.0
                        ep_length[dones] = 0.0
                    for key, value in (extras.get("log") or {}).items():
                        env_log[key].append(value.float().mean().item()
                                            if torch.is_tensor(value) else float(value))

                    # The return above is the environment's; learning uses the hook's reward.
                    rewards = self._process_rewards(env, rewards, terminated.to(self.device),
                                                    truncated.to(self.device))
                    # Time-outs aren't failures: bootstrap with the critic's estimate.
                    rewards = rewards + gamma * values * truncated.to(self.device).float()
                    storage.add(actor_obs, critic_obs, actions, log_probs, values, rewards,
                                dones.float(), mean, std)
                    actor_obs = obs["actor"].to(self.device)
                    critic_obs = obs["critic"].to(self.device)
                last_values = model.evaluate(critic_obs)
            t_collect = time.perf_counter() - t_start

            # 2. Advantages, 3. Update
            storage.compute_returns(last_values, gamma, lam)
            losses = self._update(storage)
            extra_scalars = self._extra_update(storage)
            # Refresh the normalizers only now, so the update above saw exactly the
            # normalization the rollout used (otherwise the KL check reads drift as a step).
            model.update_normalization(storage.actor_obs.flatten(0, 1), storage.critic_obs.flatten(0, 1))
            storage.clear()
            self.iteration += 1
            t_total = time.perf_counter() - t_start

            # Logging
            scalars = {
                "Train/env_steps": self.iteration * T * N,
                "Loss/policy_loss": losses["policy"],
                "Loss/value_loss": losses["value"],
                "Loss/entropy": losses["entropy"],
                "Policy/learning_rate": self.learning_rate,
                "Policy/action_std": model.std.mean().item(),
                "Perf/steps_per_s": T * N / t_total,
                "Perf/collection_s": t_collect,
                "Perf/learning_s": t_total - t_collect,
            }
            if finished_returns:
                scalars["Train/mean_return"] = torch.cat(finished_returns).mean().item()
                scalars["Train/mean_episode_length"] = torch.cat(finished_lengths).mean().item()
            for key, values in env_log.items():
                scalars[key] = sum(values) / len(values)
            scalars.update(extra_scalars)
            scalars.update(self._extra_scalars())
            logger.log(self.iteration, scalars)
            self._print(scalars, end)

            if self.iteration % save_interval == 0 or self.iteration == end:
                self.save(self.run_dir / "checkpoints" / f"model_{self.iteration}.pt")
            if max_hours is not None and (time.perf_counter() - t_begin) / 3600.0 > float(max_hours):
                print(f"max_hours ({max_hours}) reached at iteration {self.iteration}: stopping")
                self.save(self.run_dir / "checkpoints" / f"model_{self.iteration}.pt")
                break
        logger.close()

    @torch.no_grad()
    def _warm_up_normalizers(self, env, actor_obs, critic_obs, steps: int):
        """Run one rollout without learning so the first update sees normalized inputs.

        Fresh normalizers pass raw observations through; joint velocities of tens of rad/s
        would make the very first gradient step move the policy far too much.
        """
        seen_actor, seen_critic = [actor_obs], [critic_obs]
        for _ in range(steps):
            actions, *_ = self.model.act(actor_obs)
            obs, *_ = env.step(actions)
            actor_obs, critic_obs = obs["actor"].to(self.device), obs["critic"].to(self.device)
            seen_actor.append(actor_obs)
            seen_critic.append(critic_obs)
        self.model.update_normalization(torch.cat(seen_actor), torch.cat(seen_critic))
        return actor_obs, critic_obs

    def _update(self, storage: RolloutStorage) -> dict[str, float]:
        c = self.algo_cfg
        clip, target_kl = float(c["clip_ratio"]), c.get("adaptive_lr_target_kl")
        model = self.model
        model.train()
        totals = defaultdict(float)
        n = 0
        for batch in storage.mini_batches(int(c["num_minibatches"]), int(c["num_epochs"])):
            dist = model.distribution(batch.actor_obs)
            log_probs = dist.log_prob(batch.actions).sum(-1)
            values = model.evaluate(batch.critic_obs)
            entropy = dist.entropy().sum(-1).mean()

            # Adaptive learning rate from KL(old || new) between diagonal Gaussians.
            if target_kl is not None:
                with torch.no_grad():
                    kl = torch.sum(
                        torch.log(dist.stddev / batch.old_std)
                        + (batch.old_std**2 + (batch.old_mean - dist.mean) ** 2) / (2 * dist.stddev**2)
                        - 0.5, dim=-1).mean().item()
                if kl > 2.0 * target_kl:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif 0.0 < kl < target_kl / 2.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate

            ratio = torch.exp(log_probs - batch.old_log_probs)
            surrogate = batch.advantages * ratio
            surrogate_clipped = batch.advantages * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
            policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

            # Clipped value loss: the critic can't jump far from its old prediction either.
            values_clipped = batch.old_values + (values - batch.old_values).clamp(-clip, clip)
            value_loss = torch.max((values - batch.returns) ** 2,
                                   (values_clipped - batch.returns) ** 2).mean()

            loss = policy_loss + float(c["value_loss_coef"]) * value_loss \
                - float(c["entropy_coef"]) * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(c["max_grad_norm"]))
            self.optimizer.step()

            totals["policy"] += policy_loss.item()
            totals["value"] += value_loss.item()
            totals["entropy"] += entropy.item()
            n += 1
        return {k: v / n for k, v in totals.items()}

    # ---- extension points (no-ops for plain PPO) -------------------------------------

    def _on_learn_start(self, env) -> None:
        """Called once in learn(), right before the first rollout."""

    def _process_rewards(self, env, rewards: torch.Tensor, terminated: torch.Tensor,
                         truncated: torch.Tensor) -> torch.Tensor:
        """The reward PPO learns from, given the environment's reward [N] after each step.
        Environments that are done have already been reset in ``env``."""
        return rewards

    def _extra_update(self, storage: RolloutStorage) -> dict[str, float]:
        """Further optimizer steps after the PPO update; returns scalars to log by tag."""
        return {}

    def _extra_scalars(self) -> dict[str, float]:
        """More scalars for this iteration's log (TensorBoard tags)."""
        return {}

    def _extra_state(self) -> dict:
        """More entries for the checkpoint dict."""
        return {}

    def _load_extra_state(self, ckpt: dict) -> None:
        """Restore what _extra_state() saved (full loads only, not actor_only)."""

    def _print(self, s: dict, end: int) -> None:
        ret = s.get("Train/mean_return", float("nan"))
        length = s.get("Train/mean_episode_length", float("nan"))
        print(f"[{self.iteration:5d}/{end}] return {ret:8.3f} | ep len {length:6.1f} | "
              f"value loss {s['Loss/value_loss']:.4f} | std {s['Policy/action_std']:.3f} | "
              f"lr {s['Policy/learning_rate']:.1e} | {s['Perf/steps_per_s']:,.0f} steps/s",
              flush=True)

    # ---- inference and checkpoints --------------------------------------------------

    @torch.no_grad()
    def act(self, actor_obs: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model.act_inference(actor_obs.to(self.device))

    def make_policy(self, env, checkpoint: Path):
        # Inference needs only the actor, so a policy can be played on a task whose critic
        # sees different observations (a flat-ground policy on rough terrain).
        self._build_for_env(env)
        self.load(checkpoint, actor_only=True)
        return env, lambda obs: self.act(obs["actor"])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "iteration": self.iteration, "learning_rate": self.learning_rate,
                    **self._extra_state()}, path)

    def load(self, path: Path, actor_only: bool = False) -> None:
        """Restore a checkpoint. ``actor_only`` loads just what inference needs (actor,
        its normalizer, the noise std) and leaves the critic and optimizer untouched."""
        from quadruped_rl.algorithms.checkpoints import load_checkpoint, require_format

        ckpt = load_checkpoint(path, self.device)
        require_format(ckpt, self.checkpoint_kind, Path(path))
        state = ckpt["model"]
        saved_dim = state["actor_norm.mean"].shape[0]
        if saved_dim != self.model.actor_norm.mean.shape[0]:
            raise ValueError(
                f"{path} was trained with {saved_dim} actor observations, but this task gives "
                f"{self.model.actor_norm.mean.shape[0]}: the policy cannot run on it. Use a task "
                "with the same observations.actor list.")
        self.iteration = int(ckpt["iteration"])
        self.learning_rate = float(ckpt["learning_rate"])
        if actor_only:
            actor = {k: v for k, v in state.items()
                     if k.startswith(("actor.", "actor_norm.")) or k == "log_std"}
            self.model.load_state_dict(actor, strict=False)
            return
        self.model.load_state_dict(state)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._load_extra_state(ckpt)
