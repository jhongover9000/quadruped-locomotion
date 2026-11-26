# Spec: courses v2 (ladders, graded gauntlet, directional gauntlet)

Written 2026-11-25 by the main session. It builds on `docs/specs/courses.md` and the code
it produced: `src/quadruped_rl/envs/courses.py`, `src/quadruped_rl/evaluation/courses.py`,
`scripts/course.py` and `configs/courses/`. Implementers record deviations at the end.

## Why

The first policy we ran finishes the standard gauntlet 16/16 with no falls, so the course
no longer tells policies apart. We want courses where the point of failure measures a
capability:

1. **Ladders**: one track per challenge type. Each track steps through rising difficulty
   levels that go past the training range. A policy's score on a ladder is the highest
   level it completes. Each capability is measured on its own, so failing on stairs does
   not hide how the policy handles slopes. Across all ladders this gives a capability
   profile: what the policy is good at and where it falls short.
2. **Graded gauntlet**: one mixed track whose levels get harder (L1 to L5, every challenge
   type in each level). It gives one overall "how far does it get" number. This is YAML
   only.
3. **Directional gauntlet**: the robot must cross sections while facing different ways
   relative to its travel: sideways, backwards, diagonally, plus turning on the spot. It
   tests the whole command space (vy, negative vx, combined commands) on terrain, not just
   walking forward.

## 0. Fix first: falls attributed to the wrong section

In `outputs/courses/20261124_170945_gauntlet/summary.txt`, run
`20261123_133324_astra_stance2_s0`:
- "blocks" shows entered 14, done 14, falls 4;
- "curb up" shows entered 14, done 10, falls 0.

The four robots that failed at the curb are counted as falls in blocks. Candidate causes:
- base x lags the front feet at a step-type section;
- the position is read after the reset (`record_rollout` notes that qpos at `fell[t]`
  already shows the reset robot);
- an off-by-one in the section lookup.

Fix it so that each section satisfies: entered = done + falls in that section + attempts
that ended there by timeout. A fall belongs to the furthest section the robot had entered
when its attempt ended. Write that rule in the summary legend. Add a regression test on a
synthetic rollout. If the code can recompute the summary from the saved results without
the sim, show the corrected rows for that run.

## 1. Ladders (`kind: ladders`, `configs/courses/ladders.yaml`)

- A ladder is a straight gauntlet track:
  - a 3 m flat start;
  - then levels 1..L, each one feature (using the existing section types) followed by a
    1.5 m flat rest, except where the feature is itself flat, as in friction;
  - then a 3 m flat finish.
- Section labels are `L<k> <feature part>`, for example `L3 stairs up`, `L3 stairs down`,
  `L3 rest`. The failure section then says both the level and which part failed.
- The YAML defines a ladder by a template and a list of level values, and the code expands
  it into sections. Do not hand-write every section. For example:
  ```yaml
  ladders:
    stairs:
      template:
        - {type: stairs, n: 4, rise_m: $v, tread_m: 0.30, direction: up, label: stairs up}
        - {type: flat, length_m: 1.5, label: landing}
        - {type: stairs, n: 4, rise_m: $v, tread_m: 0.30, direction: down, label: stairs down}
      levels: [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16]
  ```
  `$v` is the level value. Templates may also take derived values; keep the mechanism
  simple.
- Default ladders, with level values:

  | Ladder | Feature per level | Levels |
  |---|---|---|
  | slope | ramp up 3 m, flat 1.5 m, ramp down 3 m at ±v deg | 5, 10, 15, 20, 25, 30 |
  | stairs | 4 steps up of rise v (tread 0.30), landing 1.5 m, 4 down | 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16 m |
  | curb | curb up v, flat 2 m, curb down v | 0.04, 0.07, 0.10, 0.13, 0.16, 0.19 m |
  | friction | flat 4 m at friction v | 0.8, 0.6, 0.45, 0.35, 0.25, 0.18, 0.12 |
  | slippery decline | ramp up 10 deg 3 m at friction 1.0, ramp down 10 deg 3 m at friction v | 0.6, 0.45, 0.35, 0.25, 0.18 |
  | rough | rough 4 m, amplitude v | 0.02, 0.04, 0.06, 0.08, 0.10, 0.12 m |
  | blocks | blocks 4 m, h_max v | 0.04, 0.06, 0.08, 0.10, 0.12, 0.15 m |
  | wave | wave 4 m, wavelength 1.0, amplitude v | 0.03, 0.05, 0.07, 0.09, 0.11 m |

- Running: `scripts/course.py <runs> --course ladders [--tracks stairs curb ...]` runs each
  ladder as its own pass with all `--num-envs` robots on identical lanes. This reuses the
  gauntlet machinery: one ladder at a time, one lane per robot. The time limit is per
  ladder, from its length.
- Per robot and ladder, the score is the highest level k whose sections it fully exited,
  with 0 if L1 failed. Also record where the attempt ended (section label) and how
  (fell, timeout, stuck, finished).
- Aggregate per run and ladder:
  - the median level and the interquartile range;
  - the fraction of robots completing each level;
  - the most common failure section.
- Capability profile:
  - in `summary.txt`, a table with runs as rows and ladders as columns. Each cell holds
    the median level and its value in physical units, for example `4 (0.10 m)`;
  - `profile.png`: one horizontal bar group per ladder, one bar per run, giving the
    median level as a fraction of the ladder's top level, with the IQR as a whisker and
    the physical value as a label. Plain matplotlib, readable at a glance, colourblind-safe
    colours, legend outside the bars.
- `results.csv`: one row per robot, ladder and section, with a ladder column, plus a
  separate `ladder_scores.csv` with one row per robot and ladder.
- `--preview` draws each ladder's side elevation, with one subplot per ladder in one PNG.

## 2. Graded gauntlet (`configs/courses/gauntlet_graded.yaml`, YAML only)

- `kind: gauntlet`, existing section types only.
- Levels L1 to L5. Each level holds, in this order: low-friction flat, ramp up/down,
  rough, stairs up/down, blocks, curb up/down, wave. Each level is harder than the one
  before.
- The difficulty runs from about the training range at L1-L2 to beyond it at L4-L5. Take
  the values from the ladder table, roughly its levels 1, 2, 3, 4 and 6.
- Labels are `L<k> <part>`.
- No new code is needed beyond one summary line per run: the furthest level reached
  (median over robots). Add it only if it fits cleanly into `format_gauntlet`.

## 3. Directional gauntlet (`configs/courses/gauntlet_directions.yaml`)

The track stays straight along +x and the robot still travels +x. What changes per
section is the body heading relative to the travel direction, so the same lane geometry
tests sideways, backward and diagonal walking.

- New optional section fields:
  - `mode`: forward (body yaw 0), backward (180), left (-90 deg: the robot's left side
    leads), right (+90), diag_fl (-45), diag_fr (+45), diag_bl (-135), diag_br (+135),
    or spin. Default forward, so existing YAMLs are unchanged.
  - `speed_m_s`: travel speed along +x for that section.
- Controller: a generalization of the lane-following controller.
  - desired world velocity v_w = (speed, clip(-k_y * lateral_error, ±vy_max));
  - desired yaw = the mode's yaw;
  - wz = clip(k_yaw * wrap(desired yaw - yaw), ±wz_max);
  - body command (vx, vy) = R(-yaw) * v_w.
  Clip the body command to the policy's trained ranges (for Astra: vx ±1.0, vy ±0.5,
  wz ±1.0). Clip by scaling the planar vector, not per axis, so the direction is kept.
- Heading transitions:
  - while the yaw error exceeds 15 deg, travel speed is scaled down by
    `transition_speed_factor` (default 0.3), so the robot mostly turns first;
  - time spent with yaw error above 15 deg is reported per section as `settle_s`.
- `spin` section (flat, `spin_deg`, for example 360): on entry, travel speed is 0 until the
  robot has turned `spin_deg` in total, measured as accumulated unwrapped yaw, at the
  commanded wz (`wz_max`, sign from `spin_deg`). Then it continues forward through the rest
  of the section. Record the spin time and the drift: the distance from the spin start
  point to the point where the spin finished.
- Default speeds by mode (overridable per section): forward 0.8, backward 0.5, left/right
  0.35, diagonals 0.5 m/s.
- Extra per-section metrics, for all gauntlet kinds:
  - heading error RMS (deg);
  - body-frame command tracking RMSE (planar and yaw);
  - `settle_s`.
  Show heading error and tracking RMSE in the directional summary table.
- Default course, about 55 m; each line is label: mode, feature:
  1. start: forward, flat 3 m
  2. left: left, flat 4 m
  3. right: right, flat 4 m
  4. backward: backward, flat 4 m
  5. diagonal: diag_fl, flat 4 m
  6. spin: spin 360, flat 2 m
  7. side up 10: left, ramp +10 deg 4 m; side top: left, flat 1.5 m; side down 10: right, ramp -10 deg 4 m
  8. fwd up 10: forward, ramp +10 deg 4 m; back down 10: backward, ramp -10 deg 4 m
  9. diag rough: diag_fr, rough 5 m, amplitude 0.03
  10. side slippery: left, flat 4 m, friction 0.4
  11. back curb: backward, curb up 0.05, flat 2 m, curb down 0.05
  12. fwd stairs: forward, stairs up 3 x 0.06 (tread 0.30), landing 1.5 m, stairs down 3 x 0.06
  13. finish: forward, flat 3 m
- The previews show each section's mode, as an arrow for the body heading in the top
  view, or as a label.
- The video/replay HUD shows the current section and mode if that is straightforward.

## Acceptance checks

- `timeout 600 pytest -q` and `timeout 900 pytest -m sim -q` both pass. The current
  counts are 170 and 14.
- New tests:
  - the falls attribution regression (part 0);
  - ladder template expansion: the right number of sections, labels and values, and a
    profile that returns to ground level;
  - ladder scoring on synthetic data;
  - the directional controller: for each mode, the commanded body velocity points along
    world +x after rotation, the planar clip keeps the direction, a spin accumulates
    `spin_deg` and then resumes, and transition slow-down applies;
  - the graded YAML loads and its profile is continuous except at steps.
- Previews for `ladders`, `gauntlet_graded` and `gauntlet_directions`. Report the PNG paths.
- CPU smokes, 2 robots and short time limits, on
  `outputs/ppo/rough_velocity/20261122_001801_astra_stance2_s0`:
  - `--course ladders --tracks stairs friction`;
  - `--course gauntlet_directions`.
  Paste the summary tables. A full CPU run is not needed.

## Out of scope

Training on any course, GPU runs, commits, and changes to the beacon course.

## Deviations

Recorded by the implementer, 2026-11-25.

- **Falls bug (part 0): cause and fix.** `gauntlet_results` attributed a fall to the section
  under the base at the last counted step (`x[-1]`). It did not read the post-reset pose
  (the attempt's steps end before the reset step) and the lookup had no off-by-one: the
  robots climbed onto the curb (base x reached 37.88-37.99 m, the curb starts at 37.5 m),
  tipped back off it and fell with the base at 35.9-37.0 m, inside "blocks". Now an attempt
  ends in the furthest section it entered (the section holding its maximum x), recorded per
  robot as `end_section` (`fall_section` is the same for falls). `aggregate_gauntlet` no
  longer reads `robot_fall_section`: it takes each attempt's furthest entered section from
  the `entered` flags and its status, so old `results.csv` files can be resummarized. The
  tables get a `t/o` column (timeout or stuck) and the rule is in the legend, so entered =
  done + falls + t/o. Recomputed from the saved CSV of
  `outputs/courses/20261124_170945_gauntlet` for `20261123_133324_astra_stance2_s0`: blocks
  14 entered / 14 done / 0 falls, curb up 14 / 10 / 4 falls (was 0); other counts unchanged.
- **Plain gauntlets keep the old controller.** The directional controller drives only courses
  whose sections set `mode` or `speed_m_s` (`C.is_directional`). `gauntlet`,
  `gauntlet_graded` and the ladders keep `lane_command` (per-axis clip, no rotation and no
  transition slow-down), so their results stay comparable with runs already made. The new
  heading, tracking and `settle_s` metrics are computed for every gauntlet-type course.
- **Ladders.** `kind: ladders` in `configs/courses/ladders.yaml`; `C.ladder_course` expands
  one ladder into a plain gauntlet course, which then runs through the gauntlet path. Template
  values: a string that mentions `$v` is arithmetic on it (`-$v`, `$v * 2`, brackets),
  evaluated with a small whitelisted AST evaluator. A rest is added after each level unless
  every template section is `flat`; a ladder can set its own `rest_m`. Ladder key for the
  slippery decline: `slippery_decline` (no spaces, for `--tracks`). Curb sections are 0.5 m
  long, as on the gauntlet. A level counts as completed when all its sections, rest included,
  are exited.
- **Stuck on ladders.** The spec lists "stuck" as an outcome, so the gauntlet controller
  takes an optional `outcome: {stuck_window_s, stuck_progress_m}`. The ladders set 10 s and
  0.3 m, measured on the furthest x reached so that rocking back and forth does not count as
  progress. The other gauntlets do not set it, so they keep their behaviour.
- **Median level** is the lower median of the scores, so it is always a level that at least
  half the robots completed and has a physical value. The IQR uses numpy's linear
  percentiles. "Most common failure" is the most common `end_section` among attempts that did
  not finish, with how they ended (for example `L3 stairs up (fell 2, stuck 1)`).
- **Summary for ladders:** the capability profile table, then one block per run (median/top,
  value, IQR, finished count, fraction completing each level, most common failure), then the
  usual per-section table for each run and ladder. `profile.png` uses Okabe-Ito colours, one
  per run, with the legend outside the axes.
- **Graded gauntlet.** Flat 3 m, rough 3 m, blocks 3.2 m (8 blocks of 0.4 m), wave 3 m;
  ramps 3 m + 1.5 m top + 3 m; stairs 4 steps (tread 0.30) with a 1.5 m landing; curbs
  0.5 m + 2 m top + 0.5 m. Total 139 m. Values are the ladder levels 1-4 and 6 (wave 1-5,
  which has only 5 levels). The "furthest level reached" line went into `format_gauntlet`;
  it appears for any course with `L<k>` labels (on a ladder too). It is the lower median
  over attempts of the highest level entered.
- **Directional details.** The mode is taken from the section under the base at each step.
  `--speed` sets the forward (and post-spin) speed. Other modes use `controller.mode_speeds`
  (in the YAML) or `MODE_SPEEDS` (code), overridable per section with `speed_m_s`. During a
  spin the whole planar command is 0 (no lateral correction), so the drift is the policy's
  own. The spin rate is `wz_max` (1.0 rad/s in the YAML) clipped to the trained range. The
  spin starts the first time the robot is in the section and is not repeated. Its time is
  counted in command steps, and the drift is the planar distance between the base positions
  at the spin's start and end. The time limit is `time_limit_factor` x (sum of section
  length / section speed + spin time). Heading error leaves out the spin phase. `settle_s`
  is the time with yaw error above `transition_deg` (15). Course length is 56.8 m.
- **Previews:** `ladders_side.png` (one subplot per ladder; with `--tracks`, only those). In
  the directional top view, purple arrows show the body x axis in every lane, spin sections
  are marked "spin", and the mode is printed in the section labels of both views.
- **HUD:** gauntlet recordings store `course_sections` and `course_x0` in their meta, and
  the HUD shows `section <label> (mode <mode>)`. Ladder recordings go to
  `recordings/<run>/<ladder>/seed<k>/`.
- **New tests:** falls attribution, stuck, template values, ladder expansion and continuity,
  ladder scoring, profile.png, graded YAML and its level line, the directional controller
  (every mode, the planar clip, transition, spin), directional metrics, previews, and two
  `sim` smokes (ladders with `--tracks friction curb`, gauntlet_directions).
- **`nconmax` for course terrains (`envs/builder.py`).** The env build on the stairs ladder
  failed with `nconmax overflow (nconmax must be >= 36)`. mujoco_warp's `put_data` runs a
  forward pass on the model's initial pose, which puts the robot at the world origin; on a
  43.8 m ladder centred on the origin that point is inside L4's stairs. Course terrains now
  get `nconmax = 100` instead of 35. This sets buffer capacity only: training terrains and
  the physics of contacts below the old limit are unchanged.
- **CPU runs with `CUDA_VISIBLE_DEVICES=`.** While the user's GPU training held the GPU at
  100%, CPU-only python processes (`pytest`, `course.py --device cpu`) blocked in the WSL GPU
  driver (`dxgvmb_send_sync_msg`) during warp's CUDA init. No bad-page messages appeared in
  dmesg. One `pytest -q` was killed by its 600 s timeout this way. All acceptance runs used
  `CUDA_VISIBLE_DEVICES=` and then ran normally. Smokes: 2 robots, `--device cpu`,
  `--time-limit 30` (ladders) and `60` (directions), so the ladder scores are capped by time,
  not by the policy.
- **Not completed in this session:** the `gauntlet_directions` CPU smoke (killed after 5 s of
  sim time) and `pytest -m sim` were both stopped by the harness because the system was low
  on memory (15 GB total, the GPU training running alongside). Neither was restarted. The
  directional code is covered only by the non-sim tests so far.
