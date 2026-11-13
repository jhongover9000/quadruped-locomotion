"""Analytic actuator model with parameters fitted by system identification.

    tau = clip( Kp (q*_{t-d} - q) - Kd qd,  +/- tau_max (1 - |qd| / w_max) )
          - mu_c sign(qd) - mu_v qd

d: delay in control steps; the clip is the torque-speed envelope; mu_c, mu_v are
Coulomb and viscous friction. Parameters come from scripts/fit_actuator_model.py and
live in the robot YAML under actuator.params.

TODO(phase 6)."""
