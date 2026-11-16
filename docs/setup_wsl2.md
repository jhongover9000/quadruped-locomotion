# Training setup on Windows (WSL2 + Ubuntu)

Training runs in Ubuntu under WSL2, which gives Linux direct access to the NVIDIA GPU.
The Ubuntu app *is* WSL2: it's the Linux distribution running on the WSL2 subsystem.

## 1. Check WSL2 and the GPU

In **PowerShell**:

    wsl -l -v

`Ubuntu` should show `VERSION 2`. If it shows 1: `wsl --set-version Ubuntu 2`.

Install the latest NVIDIA driver **on Windows** (GeForce/Studio driver). Do not install
an NVIDIA driver inside Ubuntu; WSL2 uses the Windows one. Then, in **Ubuntu**:

    nvidia-smi

It should list the RTX 4080 Laptop GPU and a CUDA version of 12.4 or newer.

Also check that no native Linux NVIDIA driver libraries are installed in Ubuntu. They
are often pulled in by `apt install nvidia-cuda-toolkit` and crash CUDA under WSL2 (see
[Troubleshooting](#troubleshooting)):

    dpkg -l | grep -E "libnvidia-compute|nvidia-driver"      # should print nothing

You don't need a system CUDA toolkit: PyTorch and Warp install their own CUDA libraries
through pip.

## 2. Get the repo into the Linux filesystem

Work from Ubuntu's own disk (`~/...`), not `/mnt/c/...`: file access across the
Windows boundary is many times slower.

    sudo apt update && sudo apt install -y git git-lfs
    git lfs install
    mkdir -p ~/code && cd ~/code

Then either clone from GitHub (once the repo is pushed; the cluster will need that
anyway):

    git clone https://github.com/<user>/quadruped-locomotion.git

or clone straight from the Windows copy (it becomes this clone's `origin`):

    git clone /mnt/c/Users/mainj/Documents/GitHub/quadruped-locomotion

## 3. Install (Miniconda)

Install [Miniconda](https://docs.anaconda.com/miniconda/) in Ubuntu if you don't have it
(`conda --version`), then create the environment and install the project with pip:

    cd ~/code/quadruped-locomotion
    conda create -n mujoco python=3.12 -y      # needs 3.10 - 3.13
    conda activate mujoco
    pip install --upgrade pip
    pip install -e ".[mjlab,dev]"       # several GB: PyTorch with CUDA, MuJoCo Warp, mjlab

Install the Python packages with pip, not conda, so that PyTorch's CUDA wheels match the
versions mjlab was tested with.

Add an activation hook so that the env loads NVIDIA libraries from the Windows driver
first. This guards against the driver conflict described in
[Troubleshooting](#troubleshooting), and finds the driver directory again each time, so
Windows driver updates don't break it:

    mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
    cat > "$CONDA_PREFIX/etc/conda/activate.d/wsl_cuda.sh" <<'EOF'
    _wsl_cuda_dir=$(ls -d /usr/lib/wsl/drivers/*/ 2>/dev/null | while read d; do [ -e "$d/libcuda.so.1.1" ] && echo "${d%/}"; done | head -n1)
    if [ -n "$_wsl_cuda_dir" ]; then
        export _WSL_CUDA_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-__unset__}"
        export LD_LIBRARY_PATH="$_wsl_cuda_dir:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    unset _wsl_cuda_dir
    EOF
    cat > "$CONDA_PREFIX/etc/conda/deactivate.d/wsl_cuda.sh" <<'EOF'
    if [ -n "${_WSL_CUDA_OLD_LD_LIBRARY_PATH+x}" ]; then
        if [ "$_WSL_CUDA_OLD_LD_LIBRARY_PATH" = "__unset__" ]; then unset LD_LIBRARY_PATH
        else export LD_LIBRARY_PATH="$_WSL_CUDA_OLD_LD_LIBRARY_PATH"; fi
        unset _WSL_CUDA_OLD_LD_LIBRARY_PATH
    fi
    EOF
    conda deactivate && conda activate mujoco

Check that PyTorch sees the GPU:

    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

## 4. Verify, from known-good to ours

    pytest                  # configs, metrics, formulas, PPO on a toy task (seconds)
    pytest -m sim           # builds the Go1 environment (first run compiles GPU kernels)

Then three short training runs. Each isolates one layer, so if something fails you know
where to look:

    # a. mjlab's own Go1 task and trainer: checks the install and GPU
    #    (mjlab defaults to 1 env, 10000 iterations and wandb, so set all three)
    train Mjlab-Velocity-Flat-Unitree-Go1 --env.scene.num-envs 4096 \
        --agent.logger tensorboard --agent.max-iterations 300

    # b. our environment + rsl_rl's PPO: checks our environment and rewards
    python scripts/train.py --algo ppo_rslrl --max-iterations 300

    # c. our environment + our PPO: checks our PPO
    python scripts/train.py --algo ppo --max-iterations 300

Check the first lines of each run: (a) prints `Training with: device=cuda:0` and
`Learning iteration 0/300`; (b) and (c) print `Device: cuda:0`. If a run says `cpu`, the
env isn't activated or PyTorch can't see the GPU (section 3). (a) writes to
`logs/rsl_rl/go1_velocity/`, (b) and (c) to `outputs/<algo>/flat_velocity/`.

(b) and (c) should learn at a similar rate. The mean return and episode length should
climb steadily; by a few hundred iterations the robot should walk.

## 5. Measure throughput

    python scripts/benchmark.py

This prints steps/s and GPU memory at 1024/2048/4096 environments, plus how long a full
run and each sweep would take on this machine.

The projections are optimistic. The benchmark times an untrained policy, whose robots
fall within a few dozen steps. Once the robot walks, every episode runs to the time limit
with more contacts, and iterations slow down 3-4x. On the RTX 4080 Laptop, the benchmark
reports ~0.8 s/iteration at 4096 envs, but real training averaged ~2.3 s/iteration and
reached ~3 s/iteration late in the run: about 75 min for a 1500-iteration run, not 20.

## 6. Watch training and results

    tensorboard --logdir outputs        # open http://localhost:6006 in a Windows browser
    python scripts/play.py outputs/ppo/flat_velocity/<run>/

`play.py` opens a MuJoCo window through WSLg. If no window appears, add `--viewer viser`
and open http://localhost:8080 in a Windows browser. Robots follow random velocity
commands, so some walk backwards; `--command 1.0 0 0` sends every robot forward.

To save videos, or to watch a policy later without the GPU:

    python scripts/record.py outputs/ppo/flat_velocity/<run>/       # ~2-6 min
    python scripts/replay.py outputs/ppo/flat_velocity/<run>/recordings/<time>_tour/

`record.py` drives 16 robots through a tour of commands (stand, forward, backward,
sideways, turns) and writes a recording folder with an MP4 per filmed robot, a
diagnostics plot (commanded vs actual velocity, joints, torques against their limits, a
foot-contact gait diagram, falls in red), and `summary.txt` with falls and tracking error
per command. `replay.py` opens the recording in a viewer you can pause, step and switch
robots in, or re-renders clips (`--video fall.mp4 --robot 3 --start 9 --end 13 --speed
0.5`). Replaying needs only `mujoco`, so a recording folder can be zipped and shared. The
scripts' `--help` lists the options (other scenarios, fixed commands, other terrain, GIFs,
camera angles).

Every command in sections 4–6 assumes `conda activate mujoco`.

## Troubleshooting

**`munmap_chunk(): invalid pointer` (core dump) on `torch.cuda.is_available()`.** The
WSL `libcuda` from the Windows driver is loading a native Ubuntu NVIDIA library of a
different version (usually `libnvidia-ptxjitcompiler.so.1` from `libnvidia-compute-*`).
Check with:

    LD_DEBUG=libs python -c "import torch; torch.cuda.is_available()" 2>&1 | grep "calling init" | tail -3

If a library under `/lib/x86_64-linux-gnu/libnvidia-*` appears, remove the apt NVIDIA
packages (removing only `libnvidia-compute-*` makes apt pull in another version):

    dpkg -l | grep -iE "nvidia|libcuda"                 # see what's installed
    sudo apt purge nvidia-cuda-toolkit nvidia-cuda-dev 'libnvidia-compute-*' \
        libcudart12 libcuinj64-12.0 libnvidia-ml-dev 'nsight-*' nvidia-profiler \
        nvidia-visual-profiler nvidia-cuda-gdb nvidia-opencl-dev
    sudo apt autoremove

The activation hook from section 3 also prevents this crash on its own.

## Laptop tips

- Stay plugged in and use the "Best performance" power mode; laptop GPUs throttle on battery.
- Set the lid-close action to "Do nothing" for long runs.
- WSL2 uses up to half the machine's RAM by default. To change it, create
  `C:\Users\mainj\.wslconfig` with `[wsl2]` and `memory=24GB` (for example), then run
  `wsl --shutdown` in PowerShell.
