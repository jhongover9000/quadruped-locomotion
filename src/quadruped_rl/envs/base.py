"""QuadrupedEnv: the batched environment interface every algorithm talks to.

    obs = env.reset()                       # {"policy": [N, d_pi], "critic": [N, d_v]}
    obs, reward, done, info = env.step(actions)

Observation, reward, and termination terms are looked up by the names in the task YAML,
so a new task is a config file, not new environment code. Physics runs through a backend
(envs/backends); the joint targets pass through the robot's actuator model each physics step.

TODO(phase 1): implement on the mujoco_cpu backend first, then the GPU backend."""
