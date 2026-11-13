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

## 2. Get the repo into the Linux filesystem

Work from Ubuntu's own disk (`~/...`), not `/mnt/c/...`: file access across the
Windows boundary is many times slower.

    sudo apt update && sudo apt install -y git git-lfs python3-venv python3-pip
    git lfs install
    mkdir -p ~/code && cd ~/code

Then either clone from GitHub (once the repo is pushed; the cluster will need that
anyway):

    git clone https://github.com/<user>/quadruped-locomotion.git

or clone straight from the Windows copy (it becomes this clone's `origin`):

    git clone /mnt/c/Users/mainj/Documents/GitHub/quadruped-locomotion

## 3. Install

    cd ~/code/quadruped-locomotion
    python3 --version                   # needs 3.10 - 3.13
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e ".[mjlab,dev]"       # several GB: PyTorch with CUDA, MuJoCo Warp, mjlab

Check that PyTorch sees the GPU:

    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

## 4. Verify, from known-good to ours

    pytest                  # configs, metrics, formulas, PPO on a toy task (seconds)
    pytest -m sim           # builds the Go1 environment (first run compiles GPU kernels)

Then three short training runs. Each isolates one layer, so if something fails you know
where to look:

    # a. mjlab's own Go1 task and trainer: checks the install and GPU
    train Mjlab-Velocity-Flat-Unitree-Go1 --agent.logger tensorboard --agent.max-iterations 300

    # b. our environment + rsl_rl's PPO: checks our environment and rewards
    python scripts/train.py --algo ppo_rslrl --max-iterations 300

    # c. our environment + our PPO: checks our PPO
    python scripts/train.py --algo ppo --max-iterations 300

(b) and (c) should learn at a similar rate. The mean return and episode length should
climb steadily; by a few hundred iterations the robot should walk.

## 5. Measure throughput

    python scripts/benchmark.py

This prints steps/s and GPU memory at 1024/2048/4096 environments, plus how long a full
run and each sweep would take on this machine.

## 6. Watch training and results

    tensorboard --logdir outputs        # open http://localhost:6006 in a Windows browser
    python scripts/play.py outputs/ppo/flat_velocity/<run>/

`play.py` opens a MuJoCo window through WSLg. If no window appears, add `--viewer viser`
and open http://localhost:8080 in a Windows browser.

## Laptop tips

- Stay plugged in and use the "Best performance" power mode; laptop GPUs throttle on battery.
- Set the lid-close action to "Do nothing" for long runs.
- WSL2 uses up to half the machine's RAM by default. To change it, create
  `C:\Users\mainj\.wslconfig` with `[wsl2]` and `memory=24GB` (for example), then run
  `wsl --shutdown` in PowerShell.
