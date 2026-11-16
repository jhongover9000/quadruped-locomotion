"""SuperMimic agent (Liao et al., CoRL 2026): a learned differentiable world model and
policy improvement by backpropagation through short imagined rollouts. Training loop:

  1. collect one control step in every simulated environment with the current policy
  2. fit the world model s' = f(s, a) on the replay buffer
  3. freeze it; improve the policy by BPTT through imagined rollouts of the tracking reward
  4. repeat

The implementation is the reference repository's (see backend.py for how it is found);
this class adapts it to the Agent interface, so scripts/train.py, play.py, record.py,
evaluate.py and plot_training.py work on SuperMimic runs like on PPO runs.

One "iteration" here is one control step in every environment (the reference's unit),
not a 24-step rollout as in PPO: compare runs by environment steps, not iterations.
"""

from __future__ import annotations

from pathlib import Path

from quadruped_rl.algorithms.base import Agent
from quadruped_rl.algorithms.supermimic import backend


class SuperMimicAgent(Agent):
    task_kinds = ("tracking",)
    checkpoint_patterns = ("ckpt_[0-9]*.pt",)

    # ---- training -------------------------------------------------------------------

    def train(self, num_envs: int | None = None, max_iterations: int | None = None,
              resume: Path | None = None) -> None:
        """``max_iterations`` counts from the resume point, like the other trainers."""
        algo = dict(self.algo_cfg)
        if num_envs:
            algo["num_envs"] = num_envs
        if max_iterations:
            algo["max_iterations"] = max_iterations
        cfg = backend.build_train_config(self.robot.name, self.task_cfg, algo, self.run_dir,
                                         self.device, self.seed, resume)
        if resume is not None:
            from quadruped_rl.algorithms.checkpoints import load_checkpoint, require_format

            ckpt = load_checkpoint(resume)
            require_format(ckpt, "supermimic", Path(resume))
            cfg.schedule.total_iterations += int(ckpt["iteration"])
            del ckpt
        self.warn_robot_differences(cfg.robot)

        from supermimic.training.loop import Trainer

        from quadruped_rl.utils import watchdog

        trainer = Trainer(cfg)
        watchdog.watch_env(trainer.env)         # every simulated step counts as progress
        print(f"[supermimic] {len(trainer.lib.clips)} clips, {cfg.env.num_envs} envs, "
              f"iterations {trainer.iteration} -> {cfg.schedule.total_iterations}", flush=True)
        try:
            trainer.run()
            # The reference numbers checkpoints only at its save interval; always leave a
            # numbered file for where training stopped so "latest" means the same everywhere.
            final = self.run_dir / f"ckpt_{trainer.iteration:07d}.pt"
            if not final.exists():
                trainer.save(final.name)
        finally:
            trainer.env.close()

    def warn_robot_differences(self, spec_name: str) -> list[str]:
        from supermimic.robots import get_robot_spec

        diffs = backend.robot_spec_differences(self.robot, get_robot_spec(spec_name))
        if diffs:
            print(f"[supermimic] WARNING: configs/robots/{self.robot.name}.yaml and the SuperMimic "
                  f"spec '{spec_name}' differ, so tracking and velocity policies train on "
                  f"slightly different robots:\n{backend.collapse(diffs)}", flush=True)
        return diffs

    # ---- play / evaluation ----------------------------------------------------------

    def clip_files(self, groups: str | None = None, names: list[str] | None = None,
                   mirrors: bool = False) -> list[Path]:
        """Clips of this run's motion library: by manifest group ("train", "holdout", ...),
        by name, or everything the task trains on."""
        root = backend.ensure_importable(self.algo_cfg)
        from supermimic.envs.mdp.motion_library import discover_npz

        spec = backend.spec_name(self.robot.name, self.algo_cfg)
        motions = self.task_cfg["motions"]
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

    def make_env(self, num_envs: int, play: bool = False, seed: int | None = None,
                 clips: list[Path] | None = None, terminations: bool = True):
        """The tracking environment. ``play`` uses the evaluation settings of the reference
        (no noise, no randomization, episodes start at a clip's first frame and run to its end)."""
        backend.ensure_importable(self.algo_cfg)
        from supermimic.envs import SuperMimicEnvCfgArgs, make_env
        from supermimic.robots import set_robot_overrides

        env_cfg = self.algo_cfg.get("env") or {}
        # Actuator overrides must be set before the robot spec is first used.
        set_robot_overrides(stiffness=env_cfg.get("stiffness"), action_scale=env_cfg.get("action_scale"))
        if not play:
            raise NotImplementedError("training environments are built by the reference Trainer")
        motions = self.task_cfg["motions"]
        term = self.task_cfg.get("terminations") or {}
        files = clips or self.clip_files(motions.get("eval_groups"))
        off = {} if terminations else {"root_height_threshold": 1.0e3, "root_tilt_threshold": 1.0e3,
                                       "support_foot_threshold": 0.0}
        args = SuperMimicEnvCfgArgs(
            npz_files=[str(f) for f in files],
            robot=backend.spec_name(self.robot.name, self.algo_cfg),
            num_envs=num_envs,
            future_frames=int(motions.get("future_frames", 10)),
            episode_length_s=1.0e5,              # the clip's end decides
            action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
            sampling_mode="start", record_transitions=False,
            seed=self.seed if seed is None else seed,
            anchor_obs=bool(env_cfg.get("anchor_obs", False)),
            **({"root_height_threshold": term.get("root_height_error_m"),
                "root_tilt_threshold": term.get("root_tilt_error"),
                "support_foot_threshold": env_cfg.get("support_foot_threshold")} | off),
        )
        return make_env(args, device=self.device)

    def make_policy(self, env, checkpoint: Path):
        backend.ensure_importable(self.algo_cfg)
        import torch
        from mjlab.rl import RslRlVecEnvWrapper
        from supermimic.eval.benchmark import load_policy_from_checkpoint

        from quadruped_rl.algorithms.checkpoints import load_checkpoint, require_format

        require_format(load_checkpoint(checkpoint), "supermimic", Path(checkpoint))
        net, _ = load_policy_from_checkpoint(checkpoint, self.device)

        @torch.no_grad()
        def policy(obs):
            return net(obs["policy"], obs["goal"])

        return RslRlVecEnvWrapper(env), policy

    def benchmark(self, checkpoint: Path, groups: str | None, num_envs: int = 1024,
                  out: Path | None = None) -> dict:
        """The reference's evaluation: every clip of ``groups`` played once from its first
        frame, deterministically. Returns its summary (clips, failed, normalized return)."""
        backend.ensure_importable(self.algo_cfg)
        from supermimic.eval.benchmark import (
            load_policy_from_checkpoint,
            run_benchmark,
            save_results,
            summarize,
        )

        files = [str(f) for f in self.clip_files(groups)]
        policy, cfg = load_policy_from_checkpoint(checkpoint, self.device)
        env = cfg["env"]
        results = run_benchmark(
            policy, files, device=self.device, num_envs=num_envs,
            future_frames=env["future_frames"], anchor_obs=env.get("anchor_obs", False),
            robot=cfg["robot"], support_foot_threshold=env.get("support_foot_threshold"),
            root_height_threshold=env.get("root_height_threshold"),
            root_tilt_threshold=env.get("root_tilt_threshold"), verbose=False)
        if out is not None:
            save_results(results, out)
        summary = summarize(results)
        summary["worst"] = [{"name": r.name, "norm_return": r.norm_return, "failed": r.failed}
                            for r in sorted(results, key=lambda r: r.norm_return)[:5]]
        return summary
