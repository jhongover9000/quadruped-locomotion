"""DeepMimic agent (Peng et al., SIGGRAPH 2018): PPO on the imitation reward of a tracking
task, with reference-state initialization and early termination. It is the model-free
counterpart of SuperMimic on the same clips.

Route: our PPO (algorithms/ppo/, ``learn()`` unchanged) trains in the SuperMimic backend's
tracking environment (algorithms/supermimic/tracking.py), built from the same task YAML
by the same code. Both methods therefore see the same clips, tracking reward,
terminations and return-weighted reference-state resets. What differs:

* no action noise in the environment (SuperMimic needs it for exploration; PPO explores
  through its Gaussian policy). Set in configs/algorithms/deepmimic.yaml;
* no replay recorder (only SuperMimic's world model reads it);
* actor and critic both observe the backend's ``policy`` group (10 frames of
  proprioception) followed by its ``goal`` group (the next reference frames). The goal
  window stands in for DeepMimic's phase variable.

Checkpoints are PPO checkpoints plus a ``deepmimic`` entry (tag and reset-sampler state),
so a velocity PPO checkpoint and a DeepMimic one are never mistaken for each other.
"""

from __future__ import annotations

from pathlib import Path

import torch

from quadruped_rl.algorithms.ppo.agent import PPOAgent
from quadruped_rl.algorithms.supermimic import backend, tracking

OBS_GROUPS = ("policy", "goal")     # backend observation groups, in the order PPO sees them


def tracking_obs(obs) -> torch.Tensor:
    """The backend's observation groups as one vector per robot."""
    return torch.cat([obs[g] for g in OBS_GROUPS], dim=-1)


class TrackingObservations:
    """The backend environment as PPO expects it: observations ``{"actor", "critic"}``
    (both ``tracking_obs``). Everything else is the wrapped environment's."""

    def __init__(self, env):
        object.__setattr__(self, "env", env)

    @staticmethod
    def convert(obs) -> dict:
        x = tracking_obs(obs)
        return {"actor": x, "critic": x}

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value) -> None:        # PPO desynchronizes episodes
        self.env.episode_length_buf = value

    def reset(self, *args, **kwargs):
        obs, extras = self.env.reset(*args, **kwargs)
        return self.convert(obs), extras

    def step(self, actions):
        obs, rewards, terminated, truncated, extras = self.env.step(actions)
        return self.convert(obs), rewards, terminated, truncated, extras

    def get_observations(self) -> dict:
        return self.convert(self.env.get_observations())

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


class DeepMimicAgent(PPOAgent):
    task_kinds = ("tracking",)
    checkpoint_kind = "deepmimic"
    sampler: tracking.MotionSampler | None = None

    # ---- environment ----------------------------------------------------------------

    def train_config(self, num_envs: int):
        """The reference TrainConfig for this run (only its data and env parts are used):
        the task YAML translated exactly as for SuperMimic."""
        keys = ("backend", "env", "max_iterations", "save_interval")
        algo = {k: self.algo_cfg[k] for k in keys if k in self.algo_cfg}
        algo["num_envs"] = num_envs
        return backend.build_train_config(self.robot.name, self.task_cfg, algo, self.run_dir,
                                          self.device, self.seed)

    def make_env(self, num_envs: int, play: bool = False, seed: int | None = None,
                 clips: list[Path] | None = None, terminations: bool = True):
        """Training: the backend's training environment wrapped for PPO. ``play``: the raw
        evaluation environment, exactly SuperMimic's (make_policy wraps it)."""
        if play:
            return tracking.play_env(self.robot.name, self.task_cfg, self.algo_cfg, num_envs,
                                     self.device, seed=self.seed if seed is None else seed,
                                     clips=clips, terminations=terminations)
        from quadruped_rl.utils import watchdog

        cfg = self.train_config(num_envs)
        if seed is not None:
            cfg.seed = int(seed)
        self.warn_robot_differences(cfg.robot)
        env = tracking.make_train_env(cfg, self.device)
        self.sampler = tracking.MotionSampler.from_env(env)
        print(f"[deepmimic] {self.sampler.library.num_clips} clips, {env.num_envs} envs, "
              f"reset sampling '{cfg.env.sampling_mode}'", flush=True)
        return TrackingObservations(watchdog.watch_env(env))

    def warn_robot_differences(self, spec_name: str) -> list[str]:
        from supermimic.robots import get_robot_spec

        diffs = backend.robot_spec_differences(self.robot, get_robot_spec(spec_name))
        if diffs:
            print(f"[deepmimic] WARNING: configs/robots/{self.robot.name}.yaml and the SuperMimic "
                  f"spec '{spec_name}' differ:\n{backend.collapse(diffs)}", flush=True)
        return diffs

    def clip_files(self, groups: str | None = None, names: list[str] | None = None,
                   mirrors: bool = False) -> list[Path]:
        return tracking.clip_files(self.robot.name, self.task_cfg, self.algo_cfg, groups, names, mirrors)

    # ---- checkpoints ----------------------------------------------------------------

    def _extra_state(self) -> dict:
        extra = {"observations": list(OBS_GROUPS)}
        if self.sampler is not None and self.sampler.command is not None:
            extra["sampler"] = self.sampler.state_dict()
        return {"deepmimic": extra}

    def _load_extra_state(self, ckpt: dict) -> None:
        state = (ckpt.get("deepmimic") or {}).get("sampler")
        if self.sampler is None or state is None:
            return
        if not self.sampler.load_state_dict(state):
            print("[deepmimic] checkpoint sampler state is for a different clip set; "
                  "reset sampler restarted", flush=True)

    def _build_from_checkpoint(self, checkpoint: Path) -> None:
        """Networks sized from a checkpoint and its actor loaded, without an environment."""
        from quadruped_rl.algorithms.checkpoints import load_checkpoint, require_format

        ckpt = load_checkpoint(checkpoint, self.device)
        require_format(ckpt, self.checkpoint_kind, Path(checkpoint))
        state = ckpt["model"]
        self.build(state["actor_norm.mean"].shape[0], state["critic_norm.mean"].shape[0],
                   state["log_std"].shape[0])
        self.load(checkpoint, actor_only=True)

    # ---- play / evaluation ----------------------------------------------------------

    def make_policy(self, env, checkpoint: Path):
        view = env if isinstance(env, TrackingObservations) else TrackingObservations(env)
        return super().make_policy(view, checkpoint)

    def benchmark(self, checkpoint: Path, groups: str | None, num_envs: int = 1024,
                  out: Path | None = None) -> dict:
        """SuperMimic's clip benchmark, run by the same code on this policy: every clip of
        ``groups`` played once from its first frame, deterministically."""
        tracking.set_actuator_overrides(self.algo_cfg)
        self._build_from_checkpoint(checkpoint)
        expected = self.model.actor_norm.mean.shape[0]

        def policy(proprio, goal):           # the reference evaluator's policy signature
            x = tracking_obs({"policy": proprio, "goal": goal})
            if x.shape[-1] != expected:
                raise ValueError(f"{checkpoint} expects {expected} observations, the benchmark "
                                 f"environment gives {x.shape[-1]} (anchor_obs or future_frames differ)")
            return self.act(x)

        env_cfg = self.algo_cfg.get("env") or {}
        motions, term = self.task_cfg["motions"], self.task_cfg.get("terminations") or {}
        return tracking.benchmark(
            policy, self.clip_files(groups), robot=backend.spec_name(self.robot.name, self.algo_cfg),
            device=self.device, num_envs=num_envs,
            future_frames=int(motions.get("future_frames", 10)),
            anchor_obs=bool(env_cfg.get("anchor_obs", False)),
            support_foot_threshold=env_cfg.get("support_foot_threshold"),
            root_height_threshold=term.get("root_height_error_m"),
            root_tilt_threshold=term.get("root_tilt_error"), out=out)
