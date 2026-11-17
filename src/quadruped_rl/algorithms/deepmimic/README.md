# DeepMimic

DeepMimic (Peng et al. 2018) is PPO on an imitation reward, with episodes that start at a
random frame of a reference clip (reference-state initialization) and end as soon as the
robot strays from it (early termination). Here it is the model-free counterpart of
SuperMimic: the same clips, reward, terminations and reset sampling, a different learner.

How it is built:

- The learner is our PPO (`algorithms/ppo/agent.py`); `DeepMimicAgent` subclasses
  `PPOAgent` and leaves `learn()` unchanged.
- The environment is the SuperMimic backend's tracking environment, built from
  `configs/tasks/motion_tracking.yaml` by the code SuperMimic uses
  (`algorithms/supermimic/tracking.py`, `backend.py`). It needs the reference repository
  `supermimic-quadruped` (see `docs/supermimic.md`). `envs/builder.py` does not build
  tracking tasks.
- Observations: actor and critic both see the backend's `policy` group (10 frames of
  proprioception) followed by its `goal` group (the next 10 reference frames, in place
  of DeepMimic's phase variable).
- Differences from SuperMimic's environment: no action noise (PPO explores through its
  own Gaussian policy) and no replay recorder. Everything else in the `env:` section of
  `configs/algorithms/deepmimic.yaml` must match `supermimic.yaml`; a test checks it.
- Checkpoints: `checkpoints/model_<iteration>.pt`, PPO's format plus a `deepmimic` entry
  (which tells it apart from a velocity PPO checkpoint, and holds the reset sampler's
  state so `--continue-run` resumes sampling where it was).
- Evaluation: `scripts/evaluate.py` scores it with SuperMimic's clip benchmark (the same
  reference code, given this policy), so its row has the same columns as a SuperMimic row.

```bash
python scripts/train.py --algo deepmimic
python scripts/train.py --continue-run outputs/deepmimic/motion_tracking/<run>/
python scripts/record.py   outputs/deepmimic/motion_tracking/<run>/
python scripts/evaluate.py outputs/deepmimic/motion_tracking/<run>/ outputs/supermimic/motion_tracking/<run>/
```

Returns in its training curves are PPO's (sum of per-step rewards), not SuperMimic's
normalized return; compare the two methods with `evaluate.py`.
