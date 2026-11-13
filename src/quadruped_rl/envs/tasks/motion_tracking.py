"""Reference-motion tracking task.

Steps the reference clip in time, exposes the phase and future frames to the policy,
supports reference state initialization (start at a random phase), and terminates
when the robot drifts too far from the clip.

TODO(phase 4)."""
