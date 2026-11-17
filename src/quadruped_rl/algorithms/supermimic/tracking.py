"""The SuperMimic backend's tracking environment, shared by every method that learns to
follow reference clips (SuperMimic, DeepMimic, and later AMP), so they all see the same
clips, rewards, terminations and reference-state resets.

    files = clip_files(robot_name, task_cfg, algo_cfg, groups="holdout")
    env = play_env(robot_name, task_cfg, algo_cfg, num_envs, device, clips=files)  # evaluation settings
    cfg = backend.build_train_config(...)        # task + algorithm YAML -> reference TrainConfig
    env = make_train_env(cfg, device)            # the environment the reference Trainer trains in
    sampler = MotionSampler.from_env(env)        # motion library, frame sampling, reference states

The reference ``Trainer`` builds its own environment; ``train_env_args`` passes the same
arguments it does (tests/test_deepmimic.py compares the two calls in the source, so they
cannot drift apart silently). Everything here needs the reference repository (backend.py).
"""

from __future__ import annotations

from pathlib import Path

from quadruped_rl.algorithms.supermimic import backend

# ---- clips --------------------------------------------------------------------------

def clip_files(robot_name: str, task_cfg: dict, algo_cfg: dict, groups: str | None = None,
               names: list[str] | None = None, mirrors: bool = False) -> list[Path]:
    """Clips of the task's motion library: by manifest group ("train", "holdout", ...),
    by name, or everything the task trains on."""
    root = backend.ensure_importable(algo_cfg)
    from supermimic.envs.mdp.motion_library import discover_npz

    spec = backend.spec_name(robot_name, algo_cfg)
    motions = task_cfg["motions"]
    exclude = motions.get("exclude")
    if not mirrors:
        exclude = "_mirror$" if not exclude else f"(?:{exclude})|_mirror$"
    files = discover_npz(
        backend.resolve_path(motions["dir"], root, spec), names=names,
        manifest=backend.resolve_path(motions.get("manifest"), root, spec),
        group=groups if names is None else None, exclude=exclude)
    if not files:
        raise FileNotFoundError(
            f"No clips for groups={groups} names={names} in "
            f"{backend.resolve_path(motions['dir'], root, spec)} (build the motion library "
            "first: docs/supermimic.md)")
    return files


# ---- environments -------------------------------------------------------------------

def set_actuator_overrides(algo_cfg: dict) -> None:
    """Actuator overrides (env.stiffness / env.action_scale) must be set before the robot
    spec is first used."""
    backend.ensure_importable(algo_cfg)
    from supermimic.robots import set_robot_overrides

    env_cfg = algo_cfg.get("env") or {}
    set_robot_overrides(stiffness=env_cfg.get("stiffness"), action_scale=env_cfg.get("action_scale"))


def play_env(robot_name: str, task_cfg: dict, algo_cfg: dict, num_envs: int, device: str,
             seed: int = 0, clips: list[Path] | None = None, terminations: bool = True):
    """The evaluation settings of the reference: no noise, no randomization, episodes start
    at a clip's first frame and run to its end. ``clips`` default to the task's eval_groups."""
    set_actuator_overrides(algo_cfg)
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env

    env_cfg = algo_cfg.get("env") or {}
    motions = task_cfg["motions"]
    term = task_cfg.get("terminations") or {}
    files = clips or clip_files(robot_name, task_cfg, algo_cfg, motions.get("eval_groups"))
    off = {} if terminations else {"root_height_threshold": 1.0e3, "root_tilt_threshold": 1.0e3,
                                   "support_foot_threshold": 0.0}
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(f) for f in files],
        robot=backend.spec_name(robot_name, algo_cfg),
        num_envs=num_envs,
        future_frames=int(motions.get("future_frames", 10)),
        episode_length_s=1.0e5,              # the clip's end decides
        action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
        sampling_mode="start", record_transitions=False,
        seed=seed,
        anchor_obs=bool(env_cfg.get("anchor_obs", False)),
        **({"root_height_threshold": term.get("root_height_error_m"),
            "root_tilt_threshold": term.get("root_tilt_error"),
            "support_foot_threshold": env_cfg.get("support_foot_threshold")} | off),
    )
    return make_env(args, device=device)


def train_env_args(cfg, record_transitions: bool = False):
    """``SuperMimicEnvCfgArgs`` for a reference ``TrainConfig``: the arguments the reference
    Trainer passes (supermimic/training/loop.py), keyword for keyword. Only the replay
    recorder, which only the world model needs, is off by default."""
    from supermimic.envs import SuperMimicEnvCfgArgs
    from supermimic.envs.mdp.motion_library import discover_npz, manifest_clip_weights

    if cfg.data.library is not None:
        files = [cfg.data.library]
        cfg.env.compact_library = True
    else:
        files = discover_npz(cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips, manifest=cfg.data.manifest, group=cfg.data.group, exclude=cfg.data.exclude)
    if not files:
        raise FileNotFoundError(f"no clips under {cfg.data.npz_dir} (build the motion library "
                                "first: docs/supermimic.md)")
    return SuperMimicEnvCfgArgs(
        npz_files=[str(f) for f in files],
        robot=cfg.robot,
        num_envs=cfg.env.num_envs,
        future_frames=cfg.env.future_frames,
        episode_length_s=cfg.env.episode_length_s,
        action_noise_std=cfg.env.action_noise_std,
        obs_noise=cfg.env.obs_noise,
        reset_noise=cfg.env.reset_noise,
        domain_randomization=cfg.env.domain_randomization,
        sampling_mode=cfg.env.sampling_mode,
        replay_capacity=cfg.env.replay_capacity,
        compact_library=cfg.env.compact_library,
        anchor_obs=cfg.env.anchor_obs,
        immune_probability=cfg.env.immune_probability,
        lie_probability=cfg.env.lie_probability,
        push_interval_s=cfg.env.push_interval_s,
        push_speed=cfg.env.push_speed,
        push_down_prob=cfg.env.push_down_prob,
        clip_weight_alpha=cfg.env.clip_weight_alpha,
        static_bin_factor=cfg.env.static_bin_factor,
        clip_weights=manifest_clip_weights(cfg.data.manifest),
        support_foot_threshold=cfg.env.support_foot_threshold,
        root_height_threshold=cfg.env.root_height_threshold,
        root_tilt_threshold=cfg.env.root_tilt_threshold,
        seed=cfg.seed,
        record_transitions=record_transitions,
    )


def make_train_env(cfg, device: str, record_transitions: bool = False):
    """The training environment of a ``TrainConfig``: multi-clip reference-state resets,
    the tracking reward and the early terminations, exactly as the reference Trainer has it."""
    backend.ensure_importable()
    from supermimic.envs import make_env
    from supermimic.robots import set_robot_overrides

    set_robot_overrides(stiffness=cfg.env.stiffness, action_scale=cfg.env.action_scale)
    return make_env(train_env_args(cfg, record_transitions), device=device)


# ---- motion sampling and reference states -------------------------------------------

class MotionSampler:
    """The motion library of a tracking environment behind a small interface.

    The environment already resets every episode to a reference state (sampled by its
    ``motion`` command, return-weighted by default); this exposes the same library to
    code outside the environment, for example an AMP discriminator's reference data:

    * ``sample_frames(n)``: global frame indices, clip-balanced like the uniform part of the
      environment's reset sampler, each with at least one following frame in its clip;
    * ``reference_state(frames)``: root pose and velocity and joint state at those frames;
    * ``reset_to_frames(env_ids, frames)``: put robots exactly on the reference (no noise);
    * ``state_dict()`` / ``load_state_dict()``: the reset sampler's per-bin statistics, so a
      resumed run keeps sampling where training found the clips hard.
    """

    def __init__(self, library, command=None):
        self.library = library
        self.command = command

    @classmethod
    def from_env(cls, env) -> MotionSampler:
        command = getattr(env, "unwrapped", env).command_manager.get_term("motion")
        return cls(command.library, command)

    def sample_frames(self, n: int, generator=None):
        import torch

        lib = self.library
        bins = torch.multinomial(lib.bin_weight, n, replacement=True, generator=generator)
        u = torch.rand(n, device=lib.bin_weight.device, generator=generator)
        frames = lib.bin_start[bins] + (u * (lib.bin_end[bins] - lib.bin_start[bins]).float()).long()
        first = lib.clip_start[lib.frame_clip[frames]]
        return torch.minimum(frames, lib.last_frame_of(frames) - 1).clamp_min(first)

    def reference_state(self, frames):
        """``RefState`` (root_pos_w, root_quat_w wxyz, joint_pos, root_lin_vel_w,
        root_ang_vel_w, joint_vel), positions relative to the clip's own origin."""
        return self.library.ref(frames)

    def reference_qpos_qvel(self, frames):
        """The same in MuJoCo layout: qpos (N, 7 + nj), qvel (N, 6 + nj)."""
        return self.library.ref_qpos_qvel(frames)

    def reset_to_frames(self, env_ids, frames) -> None:
        self._require_command().reset_to_frame(env_ids, frames)

    def state_dict(self) -> dict:
        cmd = self._require_command()
        return {"bin_return": cmd.bin_return.detach().cpu().clone(),
                "bin_memory": cmd.bin_memory.detach().cpu().clone()}

    def load_state_dict(self, state: dict) -> bool:
        """Restore the sampler statistics. False (and nothing changed) when they belong to a
        different clip set."""
        cmd = self._require_command()
        if tuple(state["bin_return"].shape) != tuple(cmd.bin_return.shape):
            return False
        cmd.bin_return.copy_(state["bin_return"].to(cmd.bin_return.device))
        cmd.bin_memory.copy_(state["bin_memory"].to(cmd.bin_memory.device))
        return True

    def _require_command(self):
        if self.command is None:
            raise ValueError("this MotionSampler has a library but no environment (use from_env)")
        return self.command


# ---- evaluation ---------------------------------------------------------------------

def benchmark(policy, files: list[Path], *, robot: str, device: str, num_envs: int,
              future_frames: int, anchor_obs: bool, support_foot_threshold, root_height_threshold,
              root_tilt_threshold, out: Path | None = None) -> dict:
    """The reference's evaluation: every clip played once from its first frame,
    deterministically. ``policy(obs["policy"], obs["goal"]) -> actions`` may be any method's
    policy, so all methods are scored by the same code. Returns its summary (clips, failed,
    normalized return) and the five worst clips."""
    backend.ensure_importable()
    from supermimic.eval.benchmark import run_benchmark, save_results, summarize

    results = run_benchmark(
        policy, [str(f) for f in files], device=device, num_envs=num_envs,
        future_frames=future_frames, anchor_obs=anchor_obs, robot=robot,
        support_foot_threshold=support_foot_threshold,
        root_height_threshold=root_height_threshold, root_tilt_threshold=root_tilt_threshold,
        verbose=False)
    if out is not None:
        save_results(results, out)
    summary = summarize(results)
    summary["worst"] = [{"name": r.name, "norm_return": r.norm_return, "failed": r.failed}
                        for r in sorted(results, key=lambda r: r.norm_return)[:5]]
    return summary
