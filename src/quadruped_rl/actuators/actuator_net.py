"""Actuator network (Hwangbo et al. 2019): an MLP mapping a short history of
position errors and velocities to torque.

    tau_t = f(e_t, e_{t-1}, e_{t-2}, qd_t, qd_{t-1}, qd_{t-2}),   e = q* - q

The history lets it capture latency and internal motor dynamics. Trained by
scripts/fit_actuator_model.py on logged motor data; weights path in actuator.params.

TODO(phase 6)."""
