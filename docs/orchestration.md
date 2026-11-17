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

Still open:

6. **Go1 hip stance.** `configs/robots/go1.yaml` stands with hips at ±0.1 rad and the
   SuperMimic Go1 spec uses 0; training only warns. Fixing it means recalibrating and
   retraining. Working assumption: leave it, since the Go1 is a stand-in, and make sure
   the team robot's two files agree (P2.4).
7. **Hardware interface for phase 4:** which motor controllers and which onboard computer
   the policy will run on.

## Work in flight

Started 2026-11-16 20:45 EST by session `2b7b9bd3`, one minute after another WSL restart
(nothing from `22c07b4a` survived it; its agents left uncommitted work in two worktrees).
One runner owns the GPU; do not start other GPU work until it reports. The implementers
work in worktrees based on `c0abbe8`, CPU tests only, no commits; their diffs still have
to be reviewed by the main session, merged into the main checkout on top of its
uncommitted changes, and given their GPU acceptance checks.

| Work | Agent | State at 20:45 |
|---|---|---|
| P0.1b Warp check, then tests, P0.2 stall test, P0.4 evaluation stage, P0.5 whole pipeline (`smoke_local3`), P0.8 `play.py` | runner | Running. Stops and diagnoses at the first step that needs a code or environment change. Logs in `outputs/debug/p0/`. |
| P5.3 MPC baseline | implementer | Running. Continues the half-done work in `.claude/worktrees/agent-a2fa4d9835e8da89e`. |
| P5.2 AMP | implementer | Running. Continues the half-done work in `.claude/worktrees/agent-aa519201bbcd5026d`. |
| P5.1 DeepMimic | implementer | Running. Fresh worktree under `.claude/worktrees/`; nothing existed before. |

## Checklist

### Phase 0: stabilise the machine and land what is half done

- [x] **P0.1 GPU and kernel health.** Done when `nvidia-smi` works and
  `dmesg | grep -iE "bad page|bad rss|EXT4-fs error"` is empty after a fresh boot.
  Evidence, 11-16 about 16:50 after the restart: `nvidia-smi` reports the RTX 4080 with
  0 MiB used and the dmesg grep counts 0 lines. The fault at 15:45 was a lost driver
  mount (`ls /usr/lib/wsl/drivers` gave an input/output error), the same as that
  morning. Anything timed or trained on 11-16 between about 15:30 and the restart does
  not count.
- [ ] **P0.1b GPU simulation loads again.** Since the restart of 11-16 about 16:45, every
  script that simulates through `mujoco_warp` on `cuda:0` dies at startup with
  `Warp CUDA error 222 ... ptxas ... Unsupported .version 9.4; current version is '9.3'`
  and `Failed to load CUDA module 'mujoco_warp._src.smooth'`. Torch CUDA works and the
  test suite passes (78 passed, 3 sim passed), so the tests do not catch it. Suspected: the
  PTX compiler library being loaded does not belong to the current Windows driver.
  Diagnosis in progress. Done when
  `python scripts/train.py --algo ppo --num-envs 64 --max-iterations 3 --run-dir outputs/debug/p0/warp_check`
  exits 0. Blocks P0.2, P0.4, P0.5, P0.8, P0.10 and every acceptance check in phase 5.
- [ ] **P0.2 Land the checkpoint change.** Before it, a run killed by the stall watchdog
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
- [ ] **P0.4 Evaluation stage passes on its own.** It has never finished: about 9 minutes
  per 1600-frame video, then three hangs killed at the 50-minute limit. Cause unconfirmed
  (kernel fault or a `torch.compile` hang). Done when
  `SMOKE=1 hpc/submit_pipeline.sh --local --id smoke_local --only evaluate` exits 0 and
  `outputs/pipelines/smoke_local/report/` holds `evaluation.csv` and a recording per run.
  If it hangs, get thread stacks (`kill -USR1 <pid>`, or wait for the 5-minute stall dump)
  before changing code. Blocked by P0.1.
- [ ] **P0.5 Whole local smoke pipeline in one command.** Done when
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
- [ ] **P0.8 `play.py` on a SuperMimic run.** Never tried. Done when
  `python scripts/play.py outputs/supermimic/motion_tracking/smoke_local_go1_s0` plays
  without error (run folder name unverified; see `outputs/pipelines/smoke_local/runs.txt`).
- [ ] **P0.9 Look at the SuperMimic training dashboard.** Its legend was fixed and never
  viewed again. `outputs/pipelines/smoke_local/report/training_supermimic_motion_tracking_go1_s0.png`.
  A first pass by `visual-check` is enough.
- [ ] **P0.10 Finish and check the recordings.** 3 real runs (PPO flat, PPO rough, rsl_rl
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

The importer is finished and was exercised on a deliberately awkward synthetic robot
(`testdog`). It has not seen the real URDF. `assets/robots/custom/` is empty.

- [ ] **P2.1 Import the URDF.** Blocked by decision 5.
  `python scripts/import_robot.py robot.urdf --name <name> --kp ... --kd ...`, then read
  `assets/robots/<name>/import_report.md`. Guide: `docs/robot_import.md`.
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
- Everything about the team robot's hardware interface.
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
