"""Reference PPO: rsl_rl's implementation (the trainer mjlab uses), on our environment.

Same environment, same hyperparameters (from configs/algorithms/ppo.yaml), different
PPO code. If this learns and ours does not, the bug is in our PPO; if neither learns,
look at the environment or rewards.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from quadruped_rl.algorithms.base import Agent


class RslRlPPOAgent(Agent):
    def _runner(self, env, log_dir: str | None):
        from mjlab.rl import (MjlabOnPolicyRunner, RslRlModelCfg, RslRlOnPolicyRunnerCfg,
                              RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper)

        c = self.algo_cfg
        cfg = RslRlOnPolicyRunnerCfg(
            actor=RslRlModelCfg(
                hidden_dims=tuple(c["actor_hidden"]),
                activation=c["activation"],
                obs_normalization=True,
                distribution_cfg={"class_name": "GaussianDistribution",
                                  "init_std": float(c["init_noise_std"]), "std_type": "scalar"},
            ),
            critic=RslRlModelCfg(hidden_dims=tuple(c["critic_hidden"]), activation=c["activation"],
                                 obs_normalization=True),
            algorithm=RslRlPpoAlgorithmCfg(
                value_loss_coef=float(c["value_loss_coef"]),
                use_clipped_value_loss=True,
                clip_param=float(c["clip_ratio"]),
                entropy_coef=float(c["entropy_coef"]),
                num_learning_epochs=int(c["num_epochs"]),
                num_mini_batches=int(c["num_minibatches"]),
                learning_rate=float(c["learning_rate"]),
                schedule="adaptive" if c.get("adaptive_lr_target_kl") else "fixed",
                gamma=float(c["gamma"]),
                lam=float(c["gae_lambda"]),
                desired_kl=float(c.get("adaptive_lr_target_kl") or 0.01),
                max_grad_norm=float(c["max_grad_norm"]),
            ),
            num_steps_per_env=int(c["steps_per_env"]),
            max_iterations=int(c["max_iterations"]),
            save_interval=int(c.get("save_interval", 100)),
            logger="tensorboard",
            seed=self.seed,
            experiment_name="ppo_rslrl",
        )
        wrapped = RslRlVecEnvWrapper(env)
        return wrapped, MjlabOnPolicyRunner(wrapped, asdict(cfg), log_dir, self.device)

    def train(self, num_envs: int | None = None, max_iterations: int | None = None,
              resume: Path | None = None) -> None:
        from quadruped_rl.envs.builder import make_env

        env = make_env(self.robot, self.task_cfg, num_envs or self.algo_cfg["num_envs"],
                       self.device, seed=self.seed)
        wrapped, runner = self._runner(env, str(self.run_dir))
        if resume is not None:
            runner.load(str(resume))
        runner.learn(num_learning_iterations=max_iterations or int(self.algo_cfg["max_iterations"]),
                     init_at_random_ep_len=True)
        wrapped.close()

    def make_policy(self, env, checkpoint: Path):
        wrapped, runner = self._runner(env, None)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=self.device)
        return wrapped, runner.get_inference_policy(device=self.device)
