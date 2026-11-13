"""Builds the policy observation from hardware sensors.

This must match envs/observations.py exactly: joint order, signs, zero offsets, IMU frame,
scaling, and the previous action. Most failed first deployments are mismatches here.

TODO(phase 6)."""
