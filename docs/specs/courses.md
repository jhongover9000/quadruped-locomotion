# Spec: evaluation courses (gauntlet and beacon)

Written 2026-11-23 by the main session. Implementers record deviations at the end.

## Goal

Two fixed, reproducible test environments that every velocity-command policy runs
unchanged, so policies can be compared on the same ground:

1. **Gauntlet**: a long straight track of back-to-back sections, each one a single
   challenge: friction, steps, inclines, declines and rough ground. It answers which
   section each policy fails on, how fast it gets through each section, and at what
   energy cost.
2. **Beacon**: a wide procedural field of rocky, hilly ground with friction patches, never
   seen in training. The robot has to reach a target point quickly and efficiently without
   falling, which simulates homing on an emergency beacon.

Both are driven by a simple high-level controller that turns the robot's pose and the
goal into (vx, vy, wz) commands. The policy under test only does locomotion. The same
controller is used for every policy, so differences come from the policy alone.

Applies to policies that take a velocity command: `ppo`, `ppo_rslrl`, `amp` and `mpc`. For
motion-tracking runs (`deepmimic`, `supermimic`), refuse with a clear message.

## Design decisions

- **Geometry is data.** Each course is a YAML file in `configs/courses/`. The gauntlet is
  an ordered list of sections. The beacon field is a set of generator parameters and a seed.
  Editing the YAML changes the course, with no code change.
- **Friction per region.** MuJoCo combines the friction of the two geoms in contact, by
  default taking the larger, so a low-friction floor under high-friction feet would have no
  effect. Every course geom therefore gets a `priority` above the robot's geoms, so the
  course's friction is the one used. A test must prove the friction in effect for contacts
  on a 0.35 patch is 0.35, for example from `mj_contactForce` / `contact.friction`, or by
  a box that slides on a slope where it should.
- **Evaluation conditions are fixed.** No domain randomization (friction, mass, pushes,
  motor strength), no observation noise, deterministic actions, and fixed seeds. Recorded
  in the output.
- **Falls are final.** A termination (base contact) ends that robot's attempt. Record where
  and when it happened. The env may reset the robot internally, but everything after the
  first termination is masked out of the metrics.
- **Runs on the same backend as `record.py`/`play.py`** (mjlab, GPU or CPU), reusing
  `LoadedRun` and the rollout and metrics code in `src/quadruped_rl/evaluation/`. Integrate
  the course geometry into the mjlab scene in whatever way fits best: a custom terrain
  entity, an MjSpec edit hook, or a custom TerrainGenerator. Never edit the installed
  mjlab package.

## Course 1: gauntlet (`configs/courses/gauntlet.yaml`)

- Built along +x. There is one lane per robot, 2.0 m wide, side by side in y, all with the
  same sections. This gives N independent attempts at once. A 0.3 m high wall between lanes
  is optional; skip it if it complicates contacts. Rely on the lane-keeping controller.
- Section types, each with its own parameters: `flat`, `ramp` (angle_deg, positive up),
  `stairs` (n, rise_m, tread_m, up or down), `curb` (height_m, up or down), `rough`
  (uniform noise amplitude_m, cell_m), `blocks` (random block heights 0..h_max, block size),
  and `wave` (amplitude_m, wavelength_m). Every section also has `length_m`, `friction`
  and `label`.
- Sections join without gaps or vertical jumps, unless the section is meant to be a jump
  (stairs, curb). The track's height profile must be continuous otherwise. The test checks
  this.
- The default course is about 65 m. Labels in brackets:
  1. flat 3 m, friction 1.0 [start]
  2. flat 4 m, friction 0.35 [low friction]
  3. ramp +10 deg 4 m, flat 1.5 m, ramp -10 deg 4 m [incline 10 / decline 10]
  4. rough 5 m, amplitude 0.03 m [rough]
  5. stairs up 5 x (0.08 rise, 0.30 tread), flat 1.5 m, stairs down 5 x the same [stairs]
  6. ramp +20 deg 3 m, flat 1.5 m, ramp -20 deg 3 m [incline 20 / decline 20]
  7. blocks 4 m, 0.4 m blocks, h_max 0.06 [blocks]
  8. curb up 0.10, flat 2 m, curb down 0.10 [curb]
  9. ramp +10 deg 3 m at friction 1.0, then ramp -10 deg 4 m at friction 0.4 [slippery decline]
  10. wave 4 m, amplitude 0.05, wavelength 1.0 [wave]
  11. flat 3 m [finish]
- Start: each robot spawns at the start of its lane, facing +x.
- Controller (lane following):
  - vx = the target speed (`--speed`, default 0.8 m/s), clipped to the policy's trained
    command range from its config;
  - vy = clip(-k_y * lateral_error);
  - wz = clip(k_yaw * heading_error).
  Gains and limits go in the course YAML.
- Time limit: 3 x course length / speed.
- Per robot:
  - finished (bool), total time, and the x reached;
  - fall section and position.
- Per robot and section:
  - entered and exited (bool);
  - time in section, mean forward speed;
  - lateral deviation RMS;
  - mechanical energy integral of |tau . qdot|, and cost of transport;
  - joint jerk, RMS torque per joint group against `continuous_torque`;
  - foot slip.
  Reuse `evaluation/metrics.py` where it already computes these.
- Aggregate per section over robots: completion rate (of those that entered), median time,
  mean speed, cost of transport, and falls.

## Course 2: beacon field (`configs/courses/beacon.yaml`)

- Field 60 x 60 m by default, built from square tiles, 6 m by default. Each tile is its own
  heightfield geom, so each tile can have its own friction. All tiles sample ONE global
  height function, so the surface is continuous across tile edges. Use shared edge
  vertices; the test checks the maximum height mismatch at edges, which should be under
  1 mm.
- Global height function, seeded:
  - smooth hills: low-frequency value or Perlin noise, scaled so the steepest slope is
    about 15 to 20 deg (parameter `max_slope_deg`);
  - plus rocky texture: mid and high-frequency noise, amplitude 0.02 to 0.04 m;
  - plus scattered rocks: smooth bumps or boxes merged into the heightfield, 0.05 to
    0.12 m high, at a set density per 100 m^2.
- Per tile, friction is drawn from [0.4, 1.0] (seeded). Optionally set `low_friction_share`
  so a fraction of tiles is 0.3 to 0.4.
- Episodes: for each seed, place K robots (K = num_envs) with distinct start and goal pairs.
  Goals are `goal_distance_m` from the start (default 20 m), at a random bearing, inside the
  field with a 3 m margin. Starts are at least 5 m apart and paths do not cross where that
  can be avoided. Seeds run one after another (`--seeds 0 1 2 ...`). The default seed list
  must be disjoint from anything used in training; note which seeds the defaults are.
- Controller (beacon homing). The robot knows its own world pose and the goal position, as
  GPS, odometry or a beacon bearing would give it, but not the terrain.
  - bearing error e = angle to the goal minus yaw;
  - wz = clip(k_yaw * e, ±wz_max);
  - vx = v_max * max(0, cos e), slowing linearly within `slow_radius_m`;
  - vy = 0.
  Parameters go in the YAML.
- Outcomes:
  - success: within `goal_radius_m` (default 0.5) of the goal before the time limit, which
    defaults to 3 x distance / v_max + 10 s;
  - failure modes: `fell`, `timeout`, and `stuck` (less than 0.5 m progress in 10 s).
- Per attempt: outcome, time to goal, path length, path efficiency (straight-line distance /
  path length), mean speed, energy and cost of transport, joint jerk, RMS torque, minimum
  distance to the goal, and the friction and slope encountered (mean and max along the path).
- Aggregate per policy: success rate, median time to goal of the successes, mean path
  efficiency, mean cost of transport, and counts of each failure mode.

## Entry point: `scripts/course.py`

```
python scripts/course.py <run> [<run> ...] --course gauntlet|beacon [--num-envs 16]
       [--seeds 0 1 2] [--speed 0.8] [--checkpoint ...] [--device ...] [--video] [--robots 0 3]
       [--out DIR]
python scripts/course.py --robot astra_stance2 --course gauntlet --preview    # no policy
```

- Writes `<out>/` with:
  - `results.csv`: one row per robot and section (gauntlet), or per attempt (beacon), with
    run name and seed columns;
  - `summary.txt`: aggregate tables, one block per run, then a side-by-side comparison
    across runs;
  - `meta.json`: the course YAML resolved, the seeds, the git hash, the checkpoint per run,
    and the evaluation conditions.
- Default `<out>`: `outputs/courses/<time>_<course>/`.
- `--video`: a follow-camera video per listed robot, reusing the record.py rendering. For
  the beacon course, also draw the goal marker and the path. Also save `rollout.npz` and
  `model.mjb.gz` in the recording format, so `scripts/replay.py` works on it. If the replay
  HUD assumes the tour, it is fine if it shows less.
- `--preview`: write PNGs of the course without a policy: a top view and a side elevation
  for the gauntlet; a top view coloured by height with friction tiles outlined, plus the
  start and goal pairs, for the beacon. This is how a person checks a course before using it.
- Policy compatibility: velocity-command runs only (see Goal).

## Also in this change

- Fix: the "Spawn tiles" block in record.py's summary lists only robots 0 to 11 of 16
  (seen on `outputs/ppo/rough_velocity/20261122_001801_astra_stance2_s0/recordings/20261122_221050_tour`).
  Find out why (`spawn_tiles` or `format_summary`) and list every robot.
- README: a short "Evaluation courses" subsection under "Looking at results", with the
  commands.

## Acceptance checks

- `pytest -q` and `pytest -m sim -q` both pass. On the current tree, `pytest -q` gave
  144 passed.
- New tests:
  - gauntlet profile continuity and section lengths;
  - beacon tile-edge continuity, under 1 mm;
  - friction actually in effect on a low-friction region (see Design decisions);
  - controller unit tests: heading and lateral errors give correctly signed, clipped
    commands, and the goal check;
  - fall masking;
  - a CPU sim smoke of each course with zero actions or a tiny random policy, 2 robots and
    a few seconds, that writes results.csv and summary.txt.
- Run `python scripts/course.py --robot astra_stance2 --course gauntlet --preview` and
  the same with `--course beacon`. Report the PNG paths for visual checking.
- On CPU with few envs and a short time limit: `python scripts/course.py outputs/ppo/rough_velocity/20261122_001801_astra_stance2_s0 --course gauntlet`
  and `--course beacon`. Report the summary tables. If CPU is too slow for the full
  course, use a `--time-limit` override and say so.

## Out of scope

- A learned or terrain-aware navigation planner. The beacon controller is deliberately
  simple and shared by all policies.
- Training on these courses. They are held out for evaluation only.
- GPU runs (the user runs those), and commits.

## Deviations

Recorded by the implementer, 2026-11-23.

- **Injection.** The course is a custom `SubTerrainCfg` (`envs/courses.py`,
  `CourseSubTerrainCfg`) inside an ordinary mjlab `TerrainGeneratorCfg`, selected by
  `terrain: {type: course}` in `envs/terrain.make_terrain`. Gauntlet: a 1 x N grid whose
  tiles are the N identical lanes, so mjlab's own origin logic puts robot i at the start of
  lane i. Beacon: a 1 x 1 grid whose single tile holds all 100 heightfield tiles; start
  poses are written into `terrain.env_origins` per seed. The generator's border (10 m, top
  at z = 0) surrounds both. mjlab is not edited.
- **Course length is 54.5 m, not about 65 m.** That is the sum of the sections as listed;
  the list was kept exactly. Edit the YAML to lengthen it.
- **Section details the spec left open.** Stairs and curbs jump at the start of their
  section (each tread starts with its riser). Curb sections are 0.5 m long. The flats
  between ramps and stairs and the first ramp of item 9 have their own labels (`top 10`,
  `stairs landing`, `top 20`, `curb top`, `approach 10`), because metrics are per YAML
  section. Rough noise is uniform in +-amplitude on a `cell_m` grid (default 0.1 m), zero
  on its first and last rows so it meets its neighbours. Blocks and the vertical faces of
  stairs and curbs are boxes; ramps, rough and wave are heightfields. No lane walls.
- **Finish line** is 1 m before the end of the last section (`finish_margin_m`): the
  course ends at z = -0.18 m and the mjlab border (z = 0) makes a small wall there. Robots
  spawn 1 m into the first section (`spawn_x_m`).
- **Contact parameters.** With `priority`, MuJoCo (and mujoco_warp) also takes condim,
  solref and solimp from the course geom. Course geoms copy the feet's condim 6, solref
  (0.01, 1) and torsional/rolling friction, so foot contacts differ from training only in
  sliding friction. Side effect: non-foot robot geoms (condim 1, frictionless in training)
  get condim 6 frictional contacts with the course.
- **Beacon seeds.** The field is fixed by the YAML `seed` (3001). `--seeds` choose the start
  and goal pairs; the env is built once with the first seed as its sim seed (actions are
  deterministic and nothing else is random). Default episode seeds: 1000, 1001, 1002;
  gauntlet default: 1000 (training runs use seeds 0..99). Robots start facing their goal
  (`start_facing_goal`). Pair placement prefers paths that cross no earlier path; when 4000
  samples find none, the one with the fewest crossings is used.
- **Beacon metrics.** Path efficiency is displacement (start to final position) / path
  length, so it never exceeds 1. Aggregate time to goal, path efficiency and cost of
  transport are over successes. Friction and slope along the path are reported as mean
  and max, as specified (no minimum).
- **Extra outcome `out_of_bounds`.** A reset that is a time-out (leaving the generated
  terrain) also ends an attempt; it is reported separately from falls.
- **Evaluation conditions.** All robot domain randomization is removed (`eval_robot`);
  the action delay is fixed at the middle of the training range rounded down (0 steps for
  astra_stance2's [0, 1]). Resets place the robot exactly at its origin in the default
  joint pose (`fixed_reset` in the task, read by `make_events`).
- **CLI additions.** `--time-limit` (override), `--task` (zero-action runs only, default
  flat_velocity), `--checkpoint` takes one name for all runs or one per run.
- **Spawn tiles.** Not reproducible: `summary.txt` and `meta.json` of both recordings of
  `20261122_001801_astra_stance2_s0` (14:05 and 22:10 tours) list all 16 robots, and
  neither `spawn_tiles` nor `format_summary` truncates. No code change; added a regression
  test (`tests/test_recording.py::test_summary_lists_spawn_tiles_of_every_robot`).
- **CPU acceptance runs** used `--num-envs 2 --device cpu` and the default time limits (no
  `--time-limit` override): the gauntlet stops early once every attempt has ended (126 s
  wall), the beacon ran 3 seeds in 149 s. A first gauntlet attempt stalled after 60 s of
  simulated time with no error and was killed; the identical rerun did not stall (a stuck
  D-state python from before this session was present on the WSL kernel).
