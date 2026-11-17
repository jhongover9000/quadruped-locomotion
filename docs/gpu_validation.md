# GPU validation of the phase 5 tree

Everything that still has to be run on the laptop's GPU before DeepMimic, AMP, the MPC
baseline and the seven-stage pipeline count as working. None of it has been run on a GPU
yet. The CPU tests pass on this tree (`pytest -q`: 108 passed, `pytest -q -m sim`: 7
passed, 2026-11-16), but those tests do not exercise the GPU simulation.

Run the steps in order. Only one of them may use the GPU at a time.

| Step | What it proves | Time | Needed? |
|---|---|---|---|
| [1. Health check](#1-health-check) | The GPU passthrough is up | 1 min | Yes, before every step |
| [2. Driver self-test](#2-driver-self-test) | `run_stages.sh` works with a live GPU | 1 min | Yes, once |
| [3. Smoke pipeline](#3-smoke-pipeline) | All seven stages run and all 8 runs are scored and filmed | 1.5 to 4 h (estimate; it has never finished) | Yes |
| [4. Learning checks](#4-learning-checks) | AMP and DeepMimic returns rise | 40 min each | Yes |
| [5. Recordings of the real runs](#5-recordings-of-the-real-runs) | Videos of the three long PPO runs | about 10 min each | Optional |

[What to keep](#what-to-keep) lists the files that serve as evidence afterwards.

## Setup for every terminal

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mujoco
export MUJOCO_GL=glfw
cd ~/code/quadruped-locomotion
mkdir -p outputs/debug/p5
```

The activation matters: it puts the WSL driver libraries first, and torch crashes on CUDA
start-up without it.

## What goes wrong on this machine

These are machine faults, not code faults. Do not debug them; rerun.

- **Hung start.** About one fresh Python process in five hangs during import: 100% CPU,
  no output at all. A healthy `train.py --list` or `pytest -q` prints within 30 s. If a
  command has printed nothing for two minutes, press Ctrl-C and run it again.
  `run_stages.sh` does this by itself for pipeline stages.
- **Random one-off crashes.** `wildcardElement INTERNAL ASSERT FAILED`,
  `c10::Error ... operator schema`, `CUDA error: unknown error`, Warp error 999,
  `MALFORMED INPUT: Unhandled node kind ... aten::mul`. Rerun. The same traceback twice
  in a row from the same command is a real fault.
- **Lost GPU.** `nvidia-smi` says it cannot communicate with the driver and
  `ls /usr/lib/wsl/drivers` gives an input/output error. Nothing in WSL brings it back.
  Run `wsl --shutdown` in a Windows terminal, reopen WSL and continue from the step that
  was running. It has happened three times in one day, about one to two hours after boot.

## 1. Health check

```bash
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
ls /usr/lib/wsl/drivers > /dev/null && echo "driver mount ok"
dmesg | grep -ciE "bad page|bad rss|EXT4-fs error"
```

Good: the RTX 4080 with a few MiB used, `driver mount ok`, and a count of `0`. A count
above 0 means the kernel is corrupted: `wsl --shutdown` before doing anything else.
Lines such as `dxgk: dxgkio_escape: Ioctl failed: -22` in `dmesg` are harmless; every
`nvidia-smi` call produces them.

## 2. Driver self-test

`run_stages.sh` was changed on 2026-11-16 to stop when the GPU is lost, and that change
has only been run with the GPU down. This runs it with dummy stages:

```bash
STAGE_CMD='echo ran $STAGE' STAGES="check ppo" PIPELINE_ID=driver_test \
  scripts/dev/run_stages.sh outputs/debug/p5/driver_test
```

Good: both stages `passed` in the table and the last line is `ALL STAGES PASSED`.
If it prints `GPU LOST`, the GPU is down (step 1).

## 3. Smoke pipeline

```bash
PIPELINE_ID=smoke_p5 ATTEMPT_LIMIT_S=7200 \
  scripts/dev/run_stages.sh outputs/debug/p5/run_stages 2>&1 | tee outputs/debug/p5/run_stages_driver.log
```

It runs the stages `check ppo mpc motions mimic supermimic evaluate` one at a time in
smoke mode (tiny runs of a few iterations). Each stage gets up to 5 attempts; an attempt
that hangs before using the GPU is killed and retried. The terminal shows one line per
attempt and a table at the end. The full output of each attempt is in
`outputs/debug/p5/run_stages/stage_<stage>_<attempt>.log`. Video rendering in the
`evaluate` stage is slow and prints nothing for minutes at a time; that is normal.

Exit codes: 0 all stages passed, 1 a stage failed 5 times, 3 the GPU was lost.

**Good** means all three of these:

1. The last line is `ALL STAGES PASSED`.
2. `evaluation.csv` has a row for each of the 8 runs:

   ```bash
   cut -d, -f1-3,6,8,10 outputs/pipelines/smoke_p5/report/evaluation.csv
   ```

   (run, algorithm, task, robots, falls, velocity error)

3. Every run has a non-empty video:

   ```bash
   for d in outputs/pipelines/smoke_p5/report/*/; do
     printf '%-45s %s\n' "$(basename "$d")" "$(du -h "$d/video.mp4" 2>/dev/null | cut -f1 || true)"
   done
   ```

   Expected folders: `ppo_flat_velocity_go1_s0`, `ppo_rough_velocity_go1_s0`,
   `ppo_rslrl_flat_velocity_go1_s0`, `ppo_rslrl_rough_velocity_go1_s0`,
   `mpc_flat_velocity_go1_s0`, `amp_flat_velocity_go1_s0`,
   `deepmimic_motion_tracking_go1_s0`, `supermimic_motion_tracking_go1_s0`. A folder
   with no size next to it has no video.

The policies are trained for a few iterations only, so robots that stand still or fall
over in the videos are expected. The point is that every stage runs and every run is
scored and filmed.

**If it stops early**

- A stage failed 5 times: look at the last lines of its attempt logs.

  ```bash
  for f in outputs/debug/p5/run_stages/stage_<stage>_*.log; do echo "== $f"; grep -nE "Error|Traceback|assert|HANG" "$f" | tail -5; done
  ```

  Different errors from the list above on different attempts: bad luck, continue from
  that stage (below). The same traceback on every attempt: a code fault; keep the logs
  and stop here.
- The GPU was lost: `wsl --shutdown`, then continue from the stage that was running.
- A video is missing although all stages passed (a recording died from a random crash
  and the stage carried on): rerun the `evaluate` stage alone.

To continue from a stage, name it and every later one:

```bash
STAGES="mimic supermimic evaluate" PIPELINE_ID=smoke_p5 ATTEMPT_LIMIT_S=7200 \
  scripts/dev/run_stages.sh outputs/debug/p5/run_stages_2 2>&1 | tee outputs/debug/p5/run_stages_driver_2.log
```

Use a new log folder each time, as here, so the earlier attempt logs are kept.

## 4. Learning checks

The smoke pipeline shows that the code runs, not that it learns. These two runs show
whether the return rises. Run them one after the other; each stops after 300 iterations
or 39 minutes, whichever comes first.

```bash
python scripts/train.py --algo amp --task flat_velocity --num-envs 1024 \
  --max-iterations 300 --max-hours 0.65 --run-dir outputs/debug/p5/amp_learn \
  2>&1 | tee outputs/debug/p5/amp_learn.log
```

```bash
python scripts/train.py --algo deepmimic --num-envs 1024 \
  --max-iterations 300 --max-hours 0.65 --run-dir outputs/debug/p5/deepmimic_learn \
  2>&1 | tee outputs/debug/p5/deepmimic_learn.log
```

If a run is interrupted, continue it with
`python scripts/train.py --continue-run <run dir>` and the same limits.

Look at the result in the run folder: `training_curves.png`, or in the terminal

```bash
head -1 outputs/debug/p5/amp_learn/progress.csv
awk -F, 'NR > 1 && (NR % 30 == 2)' outputs/debug/p5/amp_learn/progress.csv | cut -d, -f1-4
```

which prints the iteration, the simulated steps, the mean return and the mean episode
length every 30 iterations.

**AMP, good:** the mean return and the episode length rise over the run. For scale, plain
PPO on the same task reached a return of 22.8 in 300 iterations at 4096 robots; AMP at
1024 robots will be lower. In `training_curves.png` the four AMP panels have data, and
the discriminator's output on clips stays above its output on the policy without the
two flying apart to +1 and -1 from the first iterations (that would mean the
discriminator wins outright and the style reward is zero).

**DeepMimic, good:** the mean return rises and the episode length grows. Known risk
(`docs/specs/deepmimic.md`, Deviations): in a 5-iteration CPU run the action-rate
penalty was about twice the tracking reward, so the return starts negative and the
learning rate fell to its floor of 1e-5. If the return is flat or falling after 300
iterations, that is a finding, not a machine fault. The first thing to try is a lower
starting action noise, by adding `--algo-set init_noise_std=0.5` to the command (the
default of 1.0 is set in `configs/algorithms/ppo.yaml`). That override has not been tried.

## 5. Recordings of the real runs

Optional. The three long PPO runs of 2026-11-14 have never been filmed to the end.
Rendering takes about 9 minutes for 1600 frames.

```bash
for r in ppo/flat_velocity/20261114_220614_go1_s0 \
         ppo/rough_velocity/20261114_225312_go1_s0 \
         ppo_rslrl/flat_velocity/20261114_215548_go1_s0; do
  python scripts/record.py "outputs/$r" --out "outputs/debug/p5/rec_$(echo "$r" | tr / _)" \
    2>&1 | tee "outputs/debug/p5/rec_$(echo "$r" | tr / _).log"
done
```

Good: each output folder holds `video.mp4`, `plots.png` and `summary.txt`. The rough PPO
run may have no checkpoint files (never verified); if `record.py` says so, skip it.
For reference, an earlier recording of the flat PPO policy had 2 of 16 robots fall
walking backward at 0.5 m/s and 2 fall stopping from 1.5 m/s.

## What to keep

These files are the evidence. With them, the open boxes in `docs/orchestration.md` can
be ticked and the "what was run" part of `hpc/README.md` rewritten.

| File | Settles |
|---|---|
| `outputs/debug/p5/run_stages_driver*.log` (the final tables) | P0.4, P0.5, P5.4 |
| `outputs/pipelines/smoke_p5/report/evaluation.csv` and the 8 `video.mp4` | P0.4, P0.10, P5.1 to P5.3 (runs, scores, films) |
| `outputs/debug/p5/amp_learn/progress.csv` and `training_curves.png` | P5.2 (learns) |
| `outputs/debug/p5/deepmimic_learn/progress.csv` and `training_curves.png` | P5.1 (learns) |
| `outputs/debug/p5/rec_*/` | P0.10 (real runs) |

Still open after all of this: someone has to look at the `smoke_p5` videos and
dashboards (P0.9, P0.10), and `hpc/README.md` has to be rewritten from the evidence
(P0.7). `outputs/` is not in git, so the evidence lives only on this laptop.
