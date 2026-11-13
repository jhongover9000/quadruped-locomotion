"""Physics backends behind one interface.

mujoco_cpu   plain MuJoCo: debugging, the viewer, and sim2sim checks
mjlab        MuJoCo Warp on NVIDIA GPUs (Linux/WSL2): main training backend
mjx          MuJoCo MJX / Playground (JAX): alternative GPU backend

Choose the GPU backend once the training GPU is known."""
