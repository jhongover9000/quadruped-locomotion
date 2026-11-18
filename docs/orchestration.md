# Orchestration: what is next and how to do it

State as of 2026-11-16, about 15:45 EST, at commit `c0abbe8`. This file is the live task
list for the project. It was rebuilt from the session transcripts of 2026-11-14 to
2026-11-16 and from a survey of the repo, after two sessions ended on usage limits and
API errors with their status unreported.

## How to use this file

- Pick the first unchecked task whose blockers are done. Each task states what "done"
  means; how you get there is your call unless the task says otherwise.
- When you finish a task, tick it, add the evidence (command run and the last lines of
  its output, or the output path) on the line below it, and commit that with the work.
- If the repo contradicts this file, trust the repo, fix the file, and say so in the
  [Log](#log). Several items below are marked "unverified" for that reason.
- If you have to deviate from a task as written, take the conservative option, record
  it in the Log, and keep going.
- Division of labour between the main session and subagents is in `~/.claude/CLAUDE.md`.
  Subagents start cold, so a brief needs the [environment facts](#environment-and-gotchas)
  pasted in.
- What the real robot needs before it moves (team inputs, hardware, the interface
  numbers, moteus settings, code still to write) is in `docs/robot_requirements.md`.
- Detail lives in the guides, not here: `docs/robot_import.md`, `docs/measurement_protocol.md`,
  `docs/supermimic.md`, `docs/deployment_checklist.md`, `docs/setup_wsl2.md`, `hpc/README.md`.

## Goal

Transfer the team's own quadruped into simulation from its URDF and measured properties
(real2sim), check trained policies in a second simulator (sim2sim), then run them on the
hardware (sim2real). The Unitree Go1 is a stand-in until the team's URDF is ready.
Policies are trained with PPO (in-house and rsl_rl) and with SuperMimic, which runs
through an adapter over the separate repo `~/code/supermimic-quadruped`. Long runs go to
the Slurm cluster through `hpc/`. No deadline has been stated.

## Decisions

Answered by Joseph on 2026-11-16:

1. WSL was restarted and `autoMemoryReclaim=disabled` is set in `C:\Users\mainj\.wslconfig`.
   If the faults recur anyway, the next suspect is host RAM (Windows Memory Diagnostic).
2. The cluster settings in `hpc/config.sh` are correct as they stand.
3. The team robot's URDF, gains and measurements are coming; no date. Phase 2 waits.
4. DeepMimic, AMP and MPC are in scope and should be implemented (phase 5).
5. `prompting_guide/` is ignored by git.
7. Hardware for phase 4 (answered 11-16, late evening): the policy runs on a Raspberry
   Pi 4 and the motor controllers are moteus r4.11. What follows from that is under P4.1.

Still open:

6. **Go1 hip stance.** `configs/robots/go1.yaml` stands with hips at ±0.1 rad and the
   SuperMimic Go1 spec uses 0; training only warns. Fixing it means recalibrating and
   retraining. Working assumption: leave it, since the Go1 is a stand-in, and make sure
   the team robot's two files agree (P2.4).
8. **Rest of the hardware interface for phase 4:** how the Pi reaches the controllers
   (mjbots pi3hat, fdcanusb, or another CAN-FD adapter), which IMU and where it is
   mounted, the motors and gear ratios, and the battery voltage. Not needed before P4.1.
   The full list of open questions is in `docs/robot_requirements.md`, section 1.

## Work in flight

### Astra import, from 2026-11-18 10:35

Joseph delivered the team robot's URDF and meshes on 11-18 (`quadruped-model/`, robot
"Astra": 12 joints, CAD mass about 17 kg, 30 N m and 20 rad/s on every joint). Gains,
measurements, motors and gear ratios are still to come. To find importer and model problems
early, an implementer is importing it with provisional gains (kp 25/25/40, kd 1.0/1.0/1.6)
on branch `astra-import`, without the SuperMimic side. It also moves the files to
`assets/robots/custom/`, works out a knee range from the collision geometry (the URDF's
knee is `continuous`) and a foot radius from the foot mesh. Logs in `outputs/debug/astra/`.
Anything trained on this import is throwaway: the real gains and masses change the model.

First pass done 11:00: the files moved and are staged in Git LFS; `--joint-range` was added
to the importer (diff reviewed, accepted); the joint map and leg labels are right; `pytest -q`
110 passed. Found:
- (a) The foot meshes became bounding boxes 14 mm below the foot spheres, so the robot
  stood on boxes that count as undesired contacts.
- (b) kp 25/25/40 is too soft: the rear legs folded to the floor. Each foot sits 0.151 m
  outboard of its hip abduction axis, so standing loads the hips with about 6 N m.
- (c) The geometry never blocks the knee, so its range has to come from the team.

Fix round done 12:10, CPU only, not committed:
- Importer change: a collision mesh on the calf whose vertices all lie within 2 foot radii
  of the foot is dropped, and the foot sphere is used instead. The report lists each one, and
  a test covers it.
- Re-imported with `--kp 50 50 80 --kd 2.0 2.0 3.2 --stance 0.7854 -1.5708 --foot-radius 0.035
  --joint-range calf 0.3927 2.9671 --mesh-dir assets/robots/custom/meshes --no-supermimic`.
- Standing: trunk 0.389 -> 0.351 m, tilt 2.1 deg, sag 0.129 rad, holding torque up to 34% of
  the limit. The foot spheres are the lowest geoms and nothing touches at spawn.
- `record.py --scenario stand` (`outputs/debug/astra/stand2`): 0 falls. The video was checked
  and the pose holds.
- The 5-iteration PPO smoke runs on the CPU.
- `pytest -q` 111 passed; `-m sim` 7 passed.
- The duplicate `torso` geom name comes from MuJoCo's URDF parser. It is harmless, because
  the importer renames every geom.
- Adversarial review: nothing medium or above. It was a Claude reviewer, because no Codex or
  Gemini CLI is installed.

Still open:
- A GPU PPO smoke, after a WSL restart.
- Each recording's `model.mjb.gz` is 77 MB because the visual meshes have 893k faces.
  Consider decimating them at import.

At 11:10 a stray PPO test process was stuck in D state, about 70 threads, load average 72.
`ps` and `pgrep` hang on it, and nothing but `wsl --shutdown` clears it. Restart WSL now
that the fix round has reported.

### Session `ac9d5b2f`, from 2026-11-16 22:56 EST

Joseph restarted WSL at about 22:55. At 22:56 (uptime 1 minute): `nvidia-smi` reports the
RTX 4080 with 0 MiB used, `ls /usr/lib/wsl/drivers` works, and
`dmesg | grep -ciE "bad page|bad rss|EXT4-fs error"` gives 0. The `dxgk ... Ioctl failed: -22`
lines appear at every `nvidia-smi` call on a healthy GPU too, so they are not a signal by
themselves.

| Work | Agent | State |
|---|---|---|
| Step 1 of "Next after the restart" | runner | Done on the landed tree, each on its second try (`outputs/debug/p5/step1_*.log`): `train.py --list` shows all six as implemented (22:58; the first try aborted at import with `c10::Error ... aten::_fused_adam.out`); `pytest -q` "108 passed, 7 deselected, 1 warning in 23.71s" (23:23; the first try hung 25 minutes with no output); `pytest -q -m sim` "7 passed, 108 deselected, 5 warnings in 19.91s" (23:54; the first try hung 30 minutes). These tests do not exercise the GPU kernels, and the last one passed with the GPU already gone. |
| Step 2, `smoke_p5` pipeline | runner | **Blocked, nothing validated.** The GPU passthrough was found dead at 23:54 (`nvidia-smi` cannot reach the driver, `ls /usr/lib/wsl/drivers` gives an input/output error, the bad-page grep still 0). It was alive at about 23:23. The pipeline was launched by accident in the same command as the failed check and was stopped by the main session at 23:56 inside its first `check` attempt; `outputs/pipelines/smoke_p5/` is empty. Needs `wsl --shutdown` from Windows (Joseph). |
| `run_stages.sh` stops on a lost GPU | main session | Done 23:57. Before this, a lost GPU meant "no hang verdict", so each attempt would have run to the hard limit (7200 s here, five attempts a stage). Now the driver mount is checked before each attempt and at every poll, and the run stops with exit code 3 and the time of the loss. Checked with the GPU down: `STAGE_CMD='echo ran $STAGE' STAGES="check ppo" scripts/dev/run_stages.sh <dir>` prints "GPU LOST before stage check attempt 1; not starting it" and exits 3. Not yet run with a working GPU, and the loss-during-an-attempt branch has not been exercised. |
| Review of the uncommitted phase 5 diff (working tree against `4c39b41`) before the commit | adversarial-reviewer | Done. Claude only (no Codex or Gemini CLI), by reading, nothing run. No serious defect: AMP losses, reset handling and feature frames, the checkpoint hook, the MPC dynamics, QP and torque mapping, and the Slurm dependencies all read as correct. Four low-severity notes, none blocking the commit; they are in the Backlog. Not covered: the three new test files, `evaluate.py`'s handling of the new algorithm names, `hpc/00_setup.sh`. |

Joseph will run the GPU steps himself (decided 11-17). The runbook is
`docs/gpu_validation.md`: health check, driver self-test, smoke pipeline, the two
learning runs, optional recordings, and which files count as evidence. The landed tree
was committed on branch `phase5-landed` (not pushed, not merged into `main`) as
"CPU-tested, GPU validation pending". When the evidence exists: tick P0.4, P0.5, P0.10,
P5.1 to P5.4 from it, do P0.7, merge the branch, remove the worktrees.

Then, after the next restart: step 2 again, step 3 (the two learning runs), step 4
(`visual-check` on the `smoke_p5` report), step 5 (P0.7 README, P0.12 commit). Step 1 need
not be repeated. For step 2, first confirm the driver with a working GPU:
`STAGE_CMD='echo ran $STAGE' STAGES="check ppo" PIPELINE_ID=driver_test scripts/dev/run_stages.sh <scratch dir>`
should print `ALL STAGES PASSED`.

Third GPU loss of the day, and what it adds (P0.13): this time nothing was force-killed
with `kill -9` and no process had used the GPU. Between the last good `nvidia-smi` (about
23:23) and the failed one (23:54) the only activity was one `pytest -q -m sim` process
hung in import for 30 minutes at 100% CPU, ended by `timeout` with SIGTERM at 23:54. So
the loss does not need a forced kill of a GPU-bound process; whether it happens during a
long hang or at the moment the hung process is ended is still unknown. To find out, do
not wait out hangs: brief runners to use short timeouts on quick commands (120 s for
`train.py --list` and for either pytest set, which take 20 to 30 s when healthy) and to
check `ls /usr/lib/wsl/drivers` before and after each one, logging the time.

### Earlier sessions

Started 2026-11-16 20:45 EST by session `2b7b9bd3`, one minute after another WSL restart
(nothing from `22c07b4a` survived it; its agents left uncommitted work in two worktrees).
One runner owns the GPU; do not start other GPU work until it reports. The implementers
work in worktrees based on `c0abbe8`, CPU tests only, no commits; their diffs still have
to be reviewed by the main session, merged into the main checkout on top of its
uncommitted changes, and given their GPU acceptance checks.

| Work | Agent | State at 20:45 |
|---|---|---|
| P0.1b Warp check, then tests, P0.2 stall test, P0.4 evaluation stage, P0.5 whole pipeline (`smoke_local3`), P0.8 `play.py` | runner | Running. Stops and diagnoses at the first step that needs a code or environment change. Logs in `outputs/debug/p0/`. |
| P5.3 MPC baseline | implementer | Code done and diff reviewed at 21:55 (accepted as a baseline). CPU: MPC tests 10 passed; tracks 0.5 and 1.0 m/s forward, 0.3 m/s sideways, 0.5 rad/s yaw with 0 to 3 falls per 16 robots in 10 s; cannot keep up at 1.5 m/s (43 falls in `evaluate.py`). GPU checks not run. In `.claude/worktrees/agent-a2fa4d9835e8da89e`. |
| P5.4 pipeline and README | implementer | Running since 22:15 in `integration-p5`, from `docs/specs/hpc_phase5.md`: two new stages (`mpc`, `mimic`), giving `check ppo mpc motions mimic supermimic evaluate`. |
| P5.2 AMP | implementer | Code done and diff reviewed by the main session at 21:20 (accepted). CPU: fast suite 70 passed, 4-iteration training, resume and `evaluate.py` work. GPU checks listed in the spec's Deviations are not run. In `.claude/worktrees/agent-aa519201bbcd5026d`. |
| P5.1 DeepMimic | implementer | Code done and diff reviewed at 21:30 (accepted). CPU: fast suite 69 passed, sim 5 passed, train, resume, `record.py`, `evaluate.py` work. GPU checks not run; learning risk noted in the spec. In `.claude/worktrees/agent-ab26d07be6548768b`. |
| Merge of DeepMimic, AMP and MPC onto `4c39b41` | implementer | Done 22:10 in `.claude/worktrees/integration-p5` (branch `integration-p5`, staged and untracked, uncommitted). CPU on the merged tree: `pytest -q` 108 passed, `pytest -q -m sim` 7 passed, `train.py --list` shows all six implemented. One hook set on `PPOAgent` (`checkpoint_kind`, `_extra_state`, `_load_extra_state`); AMP defaults to `flat_velocity`. Next: land it in the main checkout once the runner has finished with it, then the GPU checks. Three test runs hung at 100% CPU in `import warp` with a clean `dmesg`; reruns passed. |

| Landing | main session | Done 22:45: the whole `integration-p5` tree (three algorithms, seven-stage pipeline, READMEs, hang watchdog in `scripts/dev/run_stages.sh`) is applied to the main checkout as uncommitted changes on `4c39b41`; the two trees are identical under `src configs tests scripts hpc README.md`. The diffs of all four pieces were reviewed by the main session. |
| GPU validation of the landed tree | runner | **Blocked at 22:47, nothing run.** The GPU passthrough died at about 22:45 (uptime 7714 s): `nvidia-smi` cannot reach the driver, `ls /usr/lib/wsl/drivers` fails, `dmesg` shows `dxgk: dxgkio_escape: Ioctl failed: -22`. Same fault as 15:45. Needs `wsl --shutdown` from Windows (Joseph). Then run the steps under "Next after the restart" below. |
| P0.9, P0.10 first visual pass | visual-check | Done 22:50 on `outputs/pipelines/smoke_local/report/`; see P0.9 and P0.10. Recording, plots and dashboards look sound. |

### Next after the restart

Nothing below has been run on the landed tree with a GPU. In the main checkout, with the
conda env active and `MUJOCO_GL=glfw`; check `nvidia-smi` and the `dmesg` grep first.

1. `python scripts/train.py --list` (all six implemented), `pytest -q` (about 108
   passed on the CPU before landing), `pytest -q -m sim` (7 passed on the CPU).
2. `PIPELINE_ID=smoke_p5 ATTEMPT_LIMIT_S=7200 scripts/dev/run_stages.sh outputs/debug/p5/run_stages`,
   detached. Done when it prints `ALL STAGES PASSED` and
   `outputs/pipelines/smoke_p5/report/` has an `evaluation.csv` row and a non-empty
   `video.mp4` for each of the 8 runs (ppo and ppo_rslrl on flat and rough, mpc, amp,
   deepmimic, supermimic). This settles P0.4, P0.5 (stage by stage), P5.1 to P5.4.
3. Learning checks, one after the other:
   `python scripts/train.py --algo amp --task flat_velocity --num-envs 1024 --max-iterations 300 --max-hours 0.65 --run-dir outputs/debug/p5/amp_learn`
   and the same with `--algo deepmimic` (no `--task`) into `outputs/debug/p5/deepmimic_learn`.
   Look at whether the return rises; DeepMimic's risk is in `docs/specs/deepmimic.md`.
4. `visual-check` on the `smoke_p5` report videos and dashboards (P0.9, P0.10).
5. Rewrite the "what was run" part of `hpc/README.md` from that evidence (P0.7), then
   commit (P0.12).

Both GPU losses today (about 15:45 and 22:45) came within minutes of a session
force-killing hung GPU-bound processes. Unproven, but prefer letting the watchdog in
`run_stages.sh` do the killing, and check `nvidia-smi` after any forced kill.

The earlier rows of this table are finished work; the worktrees under `.claude/worktrees/`
can be removed once the landed tree is committed.

At 20:55 Joseph committed and pushed the base work as `4c39b41` (checkpoint change, dev
scripts, docs, `hpc/config.sh` fix). The AMP and DeepMimic code is not in that commit; it
is still in the worktrees above.

## Checklist

### Phase 0: stabilise the machine and land what is half done

- [x] **P0.1 GPU and kernel health.** Done when `nvidia-smi` works and
  `dmesg | grep -iE "bad page|bad rss|EXT4-fs error"` is empty after a fresh boot.
  Evidence, 11-16 about 16:50 after the restart: `nvidia-smi` reports the RTX 4080 with
  0 MiB used and the dmesg grep counts 0 lines. The fault at 15:45 was a lost driver
  mount (`ls /usr/lib/wsl/drivers` gave an input/output error), the same as that
  morning. Anything timed or trained on 11-16 between about 15:30 and the restart does
  not count.
- [x] **P0.1b GPU simulation loads again.** Evidence, 11-16 21:00 after another restart:
  the check command below exits 0 on a retry in 38 s (`outputs/debug/p0/warp_check_b3r.log`:
  Warp banner "CUDA Toolkit 13.4, Driver 13.5", kernels compiled, "Finished at iteration
  3/3"). The ptxas error did not recur. The first try hung in an import (see P0.13).
  History: since the restart of 11-16 about 16:45, every
  script that simulates through `mujoco_warp` on `cuda:0` dies at startup with
  `Warp CUDA error 222 ... ptxas ... Unsupported .version 9.4; current version is '9.3'`
  and `Failed to load CUDA module 'mujoco_warp._src.smooth'`. Torch CUDA works and the
  test suite passes (78 passed, 3 sim passed), so the tests do not catch it. Suspected: the
  PTX compiler library being loaded does not belong to the current Windows driver.
  Diagnosis in progress. Done when
  `python scripts/train.py --algo ppo --num-envs 64 --max-iterations 3 --run-dir outputs/debug/p0/warp_check`
  exits 0. Blocks P0.2, P0.4, P0.5, P0.8, P0.10 and every acceptance check in phase 5.
- [x] **P0.2 Land the checkpoint change.** Evidence, 11-16 21:09: `pytest -q` "78 passed,
  3 deselected", `pytest -q -m sim` "3 passed"; `scripts/dev/stall_test.sh` printed
  "supervisor exit code 75 after 42 s frozen", "[supermimic] 454 clips, 256 envs,
  iterations 200 -> 3000" and "PASS: exit code 75, ckpt_latest.pt present, resumed from
  iteration 200" (`outputs/debug/p0/step3b.log`). Committed by Joseph in `4c39b41`.
  History: before it, a run killed by the stall watchdog
  resumed from iteration 0, because `save_interval` is 2000. Done when the main session
  has read the diff, `pytest -q` passes, and a frozen-then-restarted run logs a resume
  from iteration 100 or later. Diff reviewed by the main session on 11-16 and found
  correct against the reference trainer (`supermimic/training/loop.py`: `save` at 148,
  `act` once per iteration at 216). Tests and stall test in progress.
- [x] **P0.3 Move the two scratchpad scripts into the repo.** Now `scripts/dev/stall_test.sh`
  and `scripts/dev/run_stages.sh`, with a README. Both find the repo from their own
  location and take an optional working folder. `stall_test.sh` exits non-zero unless the
  supervisor returned 75, `ckpt_latest.pt` exists and the run resumed from iteration 100
  or later. Checked with `bash -n` only; neither has been run in its new form (P0.1b).
  Also added `tests/test_supermimic_resume.py`: `17 passed in 0.96s`. Uncommitted.
- [ ] **P0.4 Evaluation stage passes on its own.** Partly, 11-16 21:50: the command exits
  0 in 2423 s and the report holds `evaluation.csv` (5 rows) and recordings for 4 of 5
  runs (`outputs/debug/p0/step4.log`). The rsl_rl rough run has no recording: its
  `record.py` died with the random TorchScript fault `wildcardElement INTERNAL ASSERT
  FAILED ... memory_dag.cpp:184` (P0.13), and the stage carried on. Inside the whole
  pipeline the stage has not passed yet (import hang). Still to do: a run with all
  recordings. History: it had never finished: about 9 minutes
  per 1600-frame video, then three hangs killed at the 50-minute limit. Cause unconfirmed
  (kernel fault or a `torch.compile` hang). Done when
  `SMOKE=1 hpc/submit_pipeline.sh --local --id smoke_local --only evaluate` exits 0 and
  `outputs/pipelines/smoke_local/report/` holds `evaluation.csv` and a recording per run.
  If it hangs, get thread stacks (`kill -USR1 <pid>`, or wait for the 5-minute stall dump)
  before changing code. Blocked by P0.1.
- [ ] **P0.5 Whole local smoke pipeline in one command.** Not passed, 11-16 21:50 to
  22:45: eight attempts (`outputs/debug/p0/step5*.log`), none through all five stages.
  check, ppo, motions and supermimic each passed in at least one attempt; evaluate never
  did inside the pipeline. Every stop but one was a hung Python import (P0.13); one was
  `torch.AcceleratorError: CUDA error: unknown error` in a SuperMimic segment. The
  `aten::mul` flake did not appear. Plan: land the seven-stage tree (P5.4) and validate
  stage by stage with `scripts/dev/run_stages.sh`, which is getting a hang watchdog. On
  this machine "one command, no retry" may not be reachable; the cluster run (P1.3) is
  the real test of that. Originally: done when
  `SMOKE=1 PIPELINE_ID=smoke_local2 hpc/submit_pipeline.sh --local` exits 0. The only
  single-command attempt failed in the check stage with
  `RuntimeError: MALFORMED INPUT: Unhandled node kind (in computeValue): aten::mul` from
  `torch.compile`; the same step passed on a retry. Decide whether the check stage should
  retry like the training stages do. Blocked by P0.4.
- [x] **P0.6 Fix `hpc/config.sh:9`.** Commit `c0abbe8` had `$$PROJECT_DIR/hpc/logs`, where
  `$$` is the shell's process ID. Now `$PROJECT_DIR/hpc/logs`. `PROJECT_DIR` is set on
  line 7, `submit_pipeline.sh` creates the folder and passes it to `sbatch --output`, and
  `logs/` is already ignored by git. Uncommitted.
- [ ] **P0.7 Make `hpc/README.md` true.** It says every stage ran end to end locally.
  Rewrite that part from the evidence of P0.4 and P0.5. Blocked by P0.5.
- [x] **P0.8 `play.py` on a SuperMimic run.** Evidence, 11-16 22:45:
  `python scripts/play.py outputs/supermimic/motion_tracking/smoke_local3_go1_s0` ran its
  60 s under `timeout 60` with no traceback (`outputs/debug/p0/step6b.log`); it has no
  duration flag. The first try died with the random TorchScript fault (P0.13).
  Originally: never tried. Done when
  `python scripts/play.py outputs/supermimic/motion_tracking/smoke_local_go1_s0` plays
  without error (run folder name unverified; see `outputs/pipelines/smoke_local/runs.txt`).
- [x] **P0.9 Look at the SuperMimic training dashboard.** Evidence, 11-16 22:50,
  `visual-check`: every panel has data, the "Imagined reward per term" legend sits below
  its panel and the other legends are in clear corners. The other dashboards and
  `compare_flat_velocity.png` it opened are also fine (curves of 2 or 3 points, as
  expected for 3 iterations). Originally: its legend was fixed and never
  viewed again. `outputs/pipelines/smoke_local/report/training_supermimic_motion_tracking_go1_s0.png`.
  A first pass by `visual-check` is enough.
- [ ] **P0.10 Finish and check the recordings.** Partly, 11-16 22:50: `visual-check`
  looked at the 4 smoke recordings in `outputs/pipelines/smoke_local/report/` (PPO flat,
  PPO rough, rsl_rl flat, SuperMimic): right format, readable overlay, no blank or frozen
  video, plots and `summary.txt` present. The main session opened the PPO rough frames it
  flagged: the robot stands still on the white flat spawn tile with the terrain patches
  in the distance, which is what a 3-iteration policy does; not a tooling fault. Still to
  do: the rsl_rl rough recording (crashed on the random TorchScript fault), the three new
  methods' recordings (`smoke_p5`), and the 3 real runs. Originally: 3 real runs (PPO flat, PPO rough, rsl_rl
  flat) and 5 smoke runs. Done when each has a recording folder and `visual-check` has
  looked at every video. Known from an earlier recording of the flat PPO policy: 2 of 16
  robots fell walking backward at 0.5 m/s, 2 fell stopping from 1.5 m/s, top speed 1.31
  m/s against 1.5 commanded. The rough PPO run has no checkpoint files (unverified).
- [x] **P0.11 Finish and check `docs/transfer_guide.md`.** Fact-checked on 11-16: about 45
  commands and 95 paths, keys and messages agree with the code, every link resolves, and
  no stub is presented as usable. Corrected: the install command now includes the `video`
  and `supermimic` extras (also in `README.md`), the licence sentence (16 CyberDog2 clips
  are Apache-2.0), and the Git LFS wording. `docs/supermimic.md` was corrected to say
  three data sources and that `import_robot.py` writes the robot spec. Not verifiable
  without a robot or cluster: that an imported robot stands, and the Slurm stages.
  Uncommitted.
- [ ] **P0.12 Commit and tidy.** Commit P0.2 to P0.11, the two file-mode changes and the
  deleted `Zone.Identifier` file. Delete the two aborted run folders
  `outputs/ppo*/flat_velocity/20261114_1142*`. `main` matched `origin/main` at 15:45.

- [ ] **P0.13 Machine faults on the laptop (Joseph's call).** Measured 11-16 21:00 to
  22:45 by the runner, with `dmesg` clean and `autoMemoryReclaim=disabled`: about 9 of 45
  fresh Python processes hung during import (100% CPU, R state, no output, before CUDA
  init; innermost frames in pure Python such as `inspect.signature` in triton and
  `Function.__init__` in warp; once `conda activate` itself). The same imports normally
  take 1 to 4 s, and the hangs also happened at load average 1 to 2, so it is not CPU
  contention. Separately, random `wildcardElement INTERNAL ASSERT FAILED` from TorchScript
  (twice) and one `CUDA error: unknown error` / Warp error 999. Not a code fault; a code
  change will not fix it. Next suspects: host RAM (Windows Memory Diagnostic), the WSL
  kernel build. Mitigation in the repo: the hang watchdog in `scripts/dev/run_stages.sh`.

### Phase 1: first cluster run, with the Go1

Nothing has ever been submitted to Slurm. `hpc/README.md` lists what is untested there:
submission, module and conda paths, and EGL rendering on compute nodes.

- [x] **P1.1 Site settings.** Joseph confirmed on 11-16 that `hpc/config.sh` is right for
  the cluster as it stands (scratch root, conda path, env `mujoco`, partition `nvidia`,
  `gpu:a100:1`, no account flag). Overrides, if ever needed, go in `hpc/config.local.sh`.
- [ ] **P1.2 Copy both repos to scratch and run `bash hpc/00_setup.sh`** on the login
  node. `~/code/supermimic-quadruped` is not a git repo, so it has to be copied as files,
  including `src/supermimic/robots/go1.py` and the built library `data/npz/go1/`.
  Watch the versions: the SuperMimic repo pins MuJoCo 3.16 and an mjlab commit, the local
  env has MuJoCo 3.11 and mjlab 1.6.0, and the evaluator carries a fallback for that.
- [ ] **P1.3 Cluster smoke run.** Done when `SMOKE=1 hpc/submit_pipeline.sh` finishes all
  five stages (check, ppo, motions, supermimic, evaluate) and the report has videos.
  Blocked by P0.5 and P1.2.
- [ ] **P1.4 Measure A100 speed and set `SM_SEGMENTS`.** Segments are 19 hours; the
  default is 3. `hpc/README.md:106` says the speed has not been measured.
- [ ] **P1.5 Full run.** `hpc/submit_pipeline.sh`. This gives the first PPO runs at the
  full 1500 iterations and the first SuperMimic run long enough for a meaningful video.
  Local reference points: PPO flat reached return 22.8 in 300 iterations; a SuperMimic
  smoke run of 3000 iterations at 1024 envs failed 3 of 135 training clips and 1 of 33
  held-out clips.
- [ ] **P1.6 Bring the report back and review it.** `visual-check` first, then the main
  session on anything it flags.

### Phase 2: the team's robot in simulation (real2sim)

The team robot is **Astra**. URDF and meshes arrived 11-18 (`quadruped-model/`, moving to
`assets/robots/custom/`). What the URDF says, from the main session's reading:

- 12 joints `<leg>_j1` (hip abduction), `_j2` (hip pitch), `_j3` (knee); leg order in the
  file FR, FL, RL, RR. Team convention (`assets/robots/custom/urdf/README.md`): all zero is
  lying down; standing is j1 0, j2 45 deg, j3 90 deg, which puts each foot about 0.354 m
  straight below its hip pitch axis, knees behind the hips (Go1-like).
- Limits: j1 ±26 deg, j2 0 to 90 deg (±45 deg around the stance), j3 `continuous` with no
  range. 30 N m and 20 rad/s on every joint.
- CAD mass 17.0 kg: trunk 6.6, per leg 2.6 (hip 1.0, thigh 1.31, calf 0.29). Standing
  needs about 7.4 N m at each knee and almost nothing at the hip pitch.
- Hip pitch axes are 0.525 m apart front to back and 0.383 m side to side; each hip link
  reaches 0.167 m forward (rear: backward) from the abduction motor.
- Feet: a convex collision mesh on each calf; the foot links have no geometry.
- Joint dry friction 0.2 N m on j1 and j2, 2.0 N m on j3; no damping, no armature.

Questions for the team (the first three block a real import):

1. The gains the moteus controllers will run (kp, kd per joint).
2. Motors, gear ratios and rotor inertia (armature), and whether there is an output-side
   encoder (zeroing at power-on, `docs/robot_requirements.md` section 4).
3. The knee's physical range (the URDF has none). The provisional import derives one from
   the collision shapes.
4. Is the 30 N m a peak or a continuous rating, and at what current limit?
5. Is the knee's 2.0 N m dry friction measured or estimated? It is ten times the others
   and matters for the swing.
6. Does the trunk's 6.6 kg include battery, Pi, controllers and wiring? A scale reading of
   the assembled robot settles it.
7. Can j2 go past 90 deg or below 0 if needed? ±45 deg around the stance is enough for
   walking but tight for recovering from falls.

- [ ] **P2.1 Import the URDF.** Provisional import done on the CPU (see Work in flight). It
  uses placeholder gains of kp 50/50/80, a placeholder knee range of 0.3927 to 2.9671 rad,
  and has no SuperMimic side. The real import needs questions 1 to 3 answered. Rerun the
  provisional command with the real `--kp`, `--kd` and `--joint-range`, then read
  `assets/robots/astra/import_report.md`.
  Guide: `docs/robot_import.md`.
- [ ] **P2.2 Take the measurements** in `docs/measurement_protocol.md` (mass, centre of
  mass, tilt test, pendulum swings) and store them under `data/sysid/`. Bench work for
  the team; `data/` is empty today.
- [ ] **P2.3 Write the measurement scripts.** `scripts/compute_mass_properties.py` is a
  stub, although its formulas exist and are tested in `src/quadruped_rl/sysid/mass_properties.py`.
  `scripts/fit_sysid.py`, `scripts/fit_actuator_model.py` and the modules behind them
  (`sysid/swing_test.py`, `fit_dynamics.py`, `actuator_data.py`, `actuators/analytic.py`,
  `actuator_net.py`) are stubs. Joint friction and damping are `null` in every robot config.
- [ ] **P2.4 Sync and check.** Edit `configs/robots/<name>.yaml`, run
  `python scripts/import_robot.py --sync <name>`, then `pytest -q` and `pytest -m sim`.
  Confirm the robot config and the generated SuperMimic spec use the same standing pose.
- [ ] **P2.5 Train on it.** A 300-iteration PPO run, then the SuperMimic chain: retarget
  and build the motion library in the SuperMimic repo, screen the clips, calibrate, train.
  On `testdog` 153 of 602 windows failed screening, so expect to tune this.
- [ ] **P2.6 Rerun phase 1 for the new robot.**

### Phase 3: sim2sim

- [ ] **P3.1 Write `scripts/sim2sim.py` and `src/quadruped_rl/envs/backends/mujoco_cpu.py`.**
  Both are stubs. The point is an independent check: plain MuJoCo on CPU, with the
  observation built by separate code from the training environment. Done when a trained
  policy's tracking error and fall rate there are within a stated tolerance of the
  training simulator. This is the first box in `docs/deployment_checklist.md`.

### Phase 4: sim2real

- [ ] **P4.1 Write the export and runtime.** `scripts/export_policy.py` and
  `src/quadruped_rl/deploy/{export,obs_builder,policy_runner,safety}.py` are stubs.
  Observation normalisation has to be baked into the exported file.
  Target hardware is a Raspberry Pi 4 with moteus r4.11 controllers (decision 7), which
  sets these constraints; check each against the moteus reference before relying on it:
  - The Pi has no GPU, so the export has to run on an ARM CPU (ONNX Runtime or
    TorchScript) and the runtime has to hold the control rate there. Measure the
    inference time and the loop jitter on the Pi itself.
  - moteus speaks CAN-FD and the Pi has no CAN-FD port, so an adapter is needed (open
    decision 8). Twelve controllers have to be commanded and read back within one
    control period; measure the round trip.
  - moteus runs its own position loop. The policy's joint targets go out as position
    commands with the simulator's kp and kd, and the controller's units are revolutions,
    not radians, with the gear ratio configured on the controller. The gains in
    `configs/robots/<name>.yaml` have to be converted, and that conversion belongs in
    the joint order, sign and zero-offset box of `docs/deployment_checklist.md`.
  - The controller's torque, velocity and position limits and its watchdog timeout are
    a second safety layer under `deploy/safety.py`; set them, and make the simulated
    torque limits match what the controller will enforce.
  - The actuator model for P2.3 should be fitted from moteus telemetry (position,
    velocity, torque, at the control rate), which the controller reports in each reply.
- [ ] **P4.2 Work through `docs/deployment_checklist.md`.** All 8 boxes are open: sim2sim,
  joint order, sign and zero offsets, IMU frame, pose, gains and rate, normalisation,
  log comparison, safety layer.
- [ ] **P4.3 Compare hardware logs with simulation** and feed the differences back into
  P2.3.

### Phase 5: DeepMimic, AMP and MPC

In scope by Joseph's decision of 11-16. All three are stubs today and `train.py` refuses
them; `tests/test_compatibility.py` asserts that refusal and has to change with each one.
They do not block phases 1 to 4 and can proceed alongside them, but code changes must not
land in the checkout while a pipeline validation run is using it. Order: DeepMimic first,
because AMP reuses its motion sampling and reference-state resets; MPC is independent.
Each gets a spec from the main session, then an implementer, then a diff review.

- [x] **P5.0 Specs.** `docs/specs/deepmimic.md`, `docs/specs/amp.md`, `docs/specs/mpc.md`,
  written 11-16 from an interface survey. Each leads with the design decisions taken.
  Implementers record deviations at the end of the spec they worked from.
- [ ] **P5.1 DeepMimic.** Done when `python scripts/train.py --algo deepmimic` trains on
  the Go1 motion library, the tests pass, and `record.py` and `evaluate.py` work on the run.
- [ ] **P5.2 AMP.** Same bar, with the discriminator's loss and style reward in the
  training curves.
- [ ] **P5.3 MPC baseline.** Done when it walks the Go1 at commanded velocities in
  simulation without learning, and `evaluate.py` can score it next to the learned policies.
- [ ] **P5.4 Add all three to the `hpc/` pipeline and the README table.**

## Environment and gotchas

- Python: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate mujoco`. The
  activation matters, because its hook puts the WSL driver libraries first; without it
  torch crashes on CUDA init. Never create a venv (a stray `.venv/` exists; ignore it).
- Tests: `pytest -q` for the fast set, `pytest -m sim` for the simulator set.
- Training: `python scripts/train.py --algo ppo|ppo_rslrl|supermimic --robot <name>`,
  with `--continue-run`, `--max-hours` and `--stall-timeout`. `--list` prints which
  algorithm and task pairings are allowed. A stalled run exits with code 75, which means
  "restart me".
- Pipeline: stages are `check ppo motions supermimic evaluate`. Flags: `--local`,
  `--dry-run`, `--only`, `--skip`, `--id`. State is kept in `outputs/pipelines/<id>/`.
- Random hangs of fresh Python processes, `ps` blocking, or processes in D state are the
  WSL kernel fault, not the code. Check `dmesg` before debugging anything.
- A cold-cache `torch.compile` of SuperMimic hung in 3 of 6 trials and in 0 of 8 warm
  starts. The watchdog and restart are the current answer; compiling off costs about 5x.
- Rendering: `MUJOCO_GL=glfw` works under WSLg and EGL does not. The cluster should use EGL.
- Video rendering is slow locally, about 9 minutes for 1600 frames. `RECORD_DURATION` in
  `hpc/config.sh` controls it and smoke mode uses 4 seconds.
- `scripts/benchmark.py` projects 20 minutes for 1500 PPO iterations; the real figure
  locally is about 75.
- `outputs/` is ignored by git. Nothing trained so far exists anywhere but this laptop.
- Review: no Codex or Gemini CLI is installed, so `adversarial-reviewer` falls back to a
  Claude-only review.

## Known unknowns

- Why the evaluation stage hangs (P0.4).
- Whether the hung `torch.compile` depends on the Python hash seed.
- One bare-import failure in 65 and one import-test failure in 25, never investigated.
  Probably the kernel fault.
- A100 training speed, and whether EGL rendering works on the compute nodes.
- The team robot's hardware interface beyond the Pi 4 and the moteus r4.11 controllers
  (open decision 8).
- Whether the leftover CUDA 12.0 apt packages from 11-14 were purged.

## Backlog, off the critical path

- `scripts/sweep.py` and `src/quadruped_rl/evaluation/{evaluator,pareto,plots}.py` are
  stubs. Two sweeps were proposed for the cluster: `base_height` (12 runs) and
  `smoothness_pareto` (135 runs).
- `scripts/generate_trajopt_gait.py` and `scripts/retarget_motion.py` are stubs.
- Reference clips recorded from a policy need a smooth, fully trained policy first.
- The motion data is licensed for research only (AI4Animation is CC BY-NC, the Tencent
  set is research-only). Check before any commercial use.
- `configs/robots/testdog.yaml` records a `/tmp` scratchpad path as its source URDF.
- Report polish, from the visual check of 11-16: the fall counts in `evaluation.csv`
  (from `evaluate.py`, 32 robots) and in each run's `summary.txt` (from `record.py`, 16
  robots) differ and nothing says they are separate passes; label them. The spawn tile of
  the rough terrain renders plain white, which makes motion hard to judge in the video; a
  grid or checker texture would help.
- `hpc/22_mpc.slurm` requests a GPU for a job that takes seconds on a CPU; it will wait in
  the GPU queue. Harmless, but a CPU partition would start sooner if the cluster has one.
- From the review of the phase 5 diff, 11-16 (read only, not run-tested):
  - `hpc/35_train_mimic.slurm:88-92`: a DeepMimic or AMP run that reaches `MIMIC_MAX_HOURS`
    before its last iteration only prints a warning and exits 0, so the report scores a
    partly trained policy with nothing to mark it. Mark such rows in the report, or check
    the iteration count after P1.4 has measured the A100 speed.
  - `mpc_baseline/core.py:164-173`: when the command drops to zero in mid-swing, all four
    legs count as stance at once, so a force is planned for a foot that has just lifted.
    Plausible, not reproduced.
  - `amp/agent.py:118-128`: the step on which an environment ends gets no style reward, a
    small extra termination cost. `amp/agent.py:140-141`: the replay buffer is not saved,
    so the first discriminator updates after `--continue-run` see only new windows. Both
    are as designed; listed so they are not rediscovered.
- With AMP and DeepMimic added, the `check` stage may come close to its 45-minute limit
  on the cluster; look at its wall time in the first cluster smoke run (P1.3).

## Log

- 2026-11-16: file created from transcripts `9bb2888a`, `725cf83d`, `eab039e1`,
  `cd72618b`, `178fed15` and a repo survey. Not independently verified: everything in
  "Work in flight", the run folder name in P0.8, and the checkpoint note in P0.10.
- 2026-11-16 16:00 (`cd72618b`): GPU passthrough failed (`nvidia-smi` errors, dxg ioctl
  failures in dmesg). Stopped the checkpoint implementer and the evaluation rerun ahead of
  a WSL restart. While stopping the evaluation run, `cd72618b` also killed `178fed15`'s
  `record.py` on `outputs/ppo/flat_velocity/20261114_220614_go1_s0` by mistake, so that
  recording needs redoing.
- 2026-11-16 about 17:00 (`22c07b4a`): Joseph restarted WSL with `autoMemoryReclaim=disabled`
  and answered the open decisions (see Decisions). GPU healthy. Fixed `hpc/config.sh:9`
  and added `prompting_guide/` to `.gitignore`. `docs/transfer_guide.md` exists (written
  15:37 by `178fed15`, unchecked). The `/tmp` scratchpads survived the restart. Relaunched
  phase 0 validation; added phase 5.
- 2026-11-16 17:40 (`22c07b4a`): the validation queue found that GPU simulation no longer
  loads after the restart (P0.1b). Earlier in the day the same scripts ran, so something
  changed with the restarts. `pytest -m sim` passes regardless, which means the sim tests
  do not exercise the GPU kernels; worth fixing once the cause is known. The old
  `stall_test.sh` wrote into the repo root when run by relative path; the copy in
  `scripts/dev/` does not.
- 2026-11-16 22:45 (`2b7b9bd3`): GPU passthrough lost a second time, right after the
  landing of `integration-p5` in the main checkout; nothing was validated on a GPU.
- 2026-11-16 22:56 (`ac9d5b2f`): WSL restarted by Joseph, GPU healthy. Started the GPU
  validation of the landed tree and an independent review of its diff.
- 2026-11-16 23:57 (`ac9d5b2f`): GPU passthrough lost a third time, between 23:23 and
  23:54, one hour after boot, with only CPU tests run. The review found no serious defect.
  The CPU tests pass on the landed tree. `run_stages.sh` now stops on a lost GPU. Joseph
  gave the phase 4 hardware: Raspberry Pi 4 and moteus r4.11 (decision 7).
