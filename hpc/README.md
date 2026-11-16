# Training on the cluster (Slurm)

A chain of Slurm jobs that trains every implemented method, evaluates the results and
collects a report. Each stage starts when the one before it has finished.

```
check ──> ppo (array) ──> motions ──> supermimic 1 ──> supermimic 2 ──> supermimic 3 ──> evaluate
```

| Stage | File | What it does | Time limit |
|---|---|---|---|
| check | `10_check.slurm` | GPU visible, tests pass, a few training iterations, offscreen rendering | 45 min |
| ppo | `20_train_ppo.slurm` | PPO velocity policies; one array task per algorithm × task × seed | 8 h each |
| motions | `30_supermimic_motions.slurm` | Builds the SuperMimic motion library, unless it exists | 3 h |
| supermimic | `40_train_supermimic.slurm` | SuperMimic training, as chained segments of one run | 20 h each |
| evaluate | `50_evaluate.slurm` | Evaluation table, training curves, videos, gathered in one folder | 4 h |

`submit_pipeline.sh` submits them with dependencies; `config.sh` holds every setting;
`lib.sh` has the shared helpers. `nn_refinement_job_example.slurm` is the example these
were modelled on.

## First time

1. **Copy both repositories to scratch.** The defaults in `config.sh` expect
   `/scratch/$USER/quadruped-locomotion` and `/scratch/$USER/supermimic-quadruped`.
   The SuperMimic reference repository is not on GitHub, so copy it from your workstation.
   Including its built motion library saves rebuilding it on the cluster:

   ```bash
   rsync -a --exclude logs --exclude data/raw --exclude '*.pt' \
       ~/code/supermimic-quadruped/ <cluster>:/scratch/$USER/supermimic-quadruped/
   ```

2. **Set up the environment on the login node** (needs internet):

   ```bash
   cd /scratch/$USER/quadruped-locomotion
   bash hpc/00_setup.sh
   ```

   It creates the `mujoco` conda environment, installs this repository, checks that the
   SuperMimic repository and its Go1 spec are there, and downloads the raw motion captures
   only if no built library was copied.

3. **Check your settings** in `hpc/config.sh`: partition, GPU type, seeds, environment
   counts. Put personal overrides in `hpc/config.local.sh` (not committed), for example
   `MAIL_USER=you@university.edu` (by default completion mail goes to your git e-mail).

4. **Dry run** every stage with tiny settings (about 20 minutes of GPU time):

   ```bash
   SMOKE=1 hpc/submit_pipeline.sh
   ```

## Running

```bash
hpc/submit_pipeline.sh                      # everything, one stage after another
hpc/submit_pipeline.sh --parallel           # PPO and SuperMimic branches at the same time
hpc/submit_pipeline.sh --only ppo           # a single stage
hpc/submit_pipeline.sh --skip motions       # leave a stage out (repeatable)
hpc/submit_pipeline.sh --id paper_v1        # name the submission (default: date_time)
hpc/submit_pipeline.sh --dry-run            # print the sbatch commands only
hpc/submit_pipeline.sh --local              # no Slurm: run the stages here in order
```

Watch with `squeue -u $USER`; logs are in `/scratch/$USER/logs/quad_*`.

**Order and failures.** A failed check cancels everything after it. A failed PPO task
does not stop SuperMimic. Evaluation runs on whatever finished training.

**Where results go.** Runs: `outputs/<algo>/<task>/<id>_<robot>_s<seed>/`. The report:
`outputs/pipelines/<id>/report/` with `evaluation.csv`, `training_*.png`,
`compare_<task>.png`, and per run a video, diagnostic plots, a summary and replay files.
Copy it home and watch rollouts offline, without a GPU:

```bash
rsync -a <cluster>:/scratch/$USER/quadruped-locomotion/outputs/pipelines/<id>/report/ ./report/
python scripts/replay.py ./report/<run>/
```

## Long runs, interruptions and hangs

* **SuperMimic takes longer than one job may run.** It trains in `SM_SEGMENTS` chained
  jobs. Each stops cleanly after `SM_SEGMENT_HOURS` and the next continues the same run
  directory from the last checkpoint (`scripts/train.py --continue-run`). A segment that
  finds the run finished exits at once. If the last segment ends before training does, its
  log prints the command that adds more segments.
* **Resubmitting is safe.** Run directories are named after the pipeline id, and a stage
  continues a run that already exists. After a node failure, submit the same stage again
  with the same `--id`.
* **Hangs.** Training runs under a supervisor (`--stall-timeout`): a run that makes no
  progress for `STALL_TIMEOUT_S` has its Python stacks printed to the `.err` log and is
  restarted from its last checkpoint, up to `STALL_RETRIES` times. This exists because
  the first `torch.compile` of a SuperMimic run sometimes never returns (it was seen stuck
  inside sympy for many minutes on a workstation); once the compiled kernels are cached
  (on scratch, `TORCHINDUCTOR_CACHE_DIR`), later starts take seconds.
  `kill -USR1 <pid>` on the node prints the stacks of a running job at any time.

## Sizing SuperMimic

One iteration is one control step in every environment. Measured on an RTX 4080 Laptop:
1024 environments use about 4 GiB of GPU memory and run 3000 iterations in 14 minutes.
The reference run (20480 environments, 50000 iterations) used about 72 GiB. The default
here, `SM_NUM_ENVS=8192`, is sized for a 40 GB A100; on an 80 GB A100 set
`SM_NUM_ENVS=20480`. If a job dies with CUDA out of memory, halve it. The speed on an
A100 has not been measured yet: the first segment's log shows iterations per second, and
`SM_SEGMENTS` should be set from that.

## Not tested here

These scripts were written and exercised on a workstation without Slurm: the stage
scripts ran end to end through `SMOKE=1 hpc/submit_pipeline.sh --local`, and the
`sbatch` commands were checked with `--dry-run`. The Slurm submission itself, the
module/conda paths and EGL rendering on the compute nodes are untested until the first
`SMOKE=1` run on the cluster; stage `check` exists to surface those problems in minutes.
