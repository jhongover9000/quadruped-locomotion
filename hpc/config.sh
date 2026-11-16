# Cluster settings, sourced by every script in hpc/. Edit here, or override any of them
# in hpc/config.local.sh (not committed) or in the environment before submitting.
# shellcheck shell=bash

# ---- where things live ---------------------------------------------------------------
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER}"
PROJECT_DIR="${PROJECT_DIR:-$SCRATCH_ROOT/quadruped-locomotion}"       # this repository
SUPERMIMIC_ROOT="${SUPERMIMIC_ROOT:-$SCRATCH_ROOT/supermimic-quadruped}" # SuperMimic reference repo
LOG_DIR="${LOG_DIR:-$SCRATCH_ROOT/logs}"                                 # Slurm .out / .err files
export QUADRUPED_OUTPUT_DIR="${QUADRUPED_OUTPUT_DIR:-$PROJECT_DIR/outputs}"
PIPELINE_DIR="${PIPELINE_DIR:-$QUADRUPED_OUTPUT_DIR/pipelines}"          # one state folder per submission

# ---- Python environment --------------------------------------------------------------
CONDA_ACTIVATE="${CONDA_ACTIVATE:-/share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate}"
CONDA_MODULE="${CONDA_MODULE:-miniconda}"
CONDA_ENV="${CONDA_ENV:-mujoco}"

# ---- Slurm resources (the GPU jobs; see each .slurm file for its own time limit) -------
SLURM_PARTITION="${SLURM_PARTITION:-nvidia}"
SLURM_GRES="${SLURM_GRES:-gpu:a100:1}"
# Completion mail goes here; default: your git e-mail on the cluster. Empty = no mail.
MAIL_USER="${MAIL_USER:-$(git config user.email 2>/dev/null || true)}"

# ---- what to train -------------------------------------------------------------------
ROBOT="${ROBOT:-go1}"
SEEDS="${SEEDS:-0 1 2}"                       # PPO seeds; each is one array task

PPO_ALGOS="${PPO_ALGOS:-ppo ppo_rslrl}"       # trained on every task in PPO_TASKS
PPO_TASKS="${PPO_TASKS:-flat_velocity rough_velocity}"
PPO_NUM_ENVS="${PPO_NUM_ENVS:-4096}"
PPO_ITERATIONS_FLAT="${PPO_ITERATIONS_FLAT:-1500}"
PPO_ITERATIONS_ROUGH="${PPO_ITERATIONS_ROUGH:-3000}"

# SuperMimic. One iteration = one control step in every environment.
# GPU memory grows with SM_NUM_ENVS: about 4 GiB at 1024 (measured), 72 GiB at 20480
# (reference run). 8192 leaves headroom on a 40 GB A100; an 80 GB A100 fits 20480.
SM_NUM_ENVS="${SM_NUM_ENVS:-8192}"
SM_ITERATIONS="${SM_ITERATIONS:-50000}"
SM_SEED="${SM_SEED:-0}"
SM_SEGMENTS="${SM_SEGMENTS:-3}"               # chained jobs, each continuing the same run
SM_SEGMENT_HOURS="${SM_SEGMENT_HOURS:-19}"    # training stops cleanly before the 20 h limit
SM_STIFFNESS="${SM_STIFFNESS:-76}"            # kp used to screen which clips are trackable

# A run with no progress for this long is restarted from its last checkpoint (0 = never).
STALL_TIMEOUT_S="${STALL_TIMEOUT_S:-1800}"
STALL_RETRIES="${STALL_RETRIES:-3}"

# ---- evaluation ----------------------------------------------------------------------
EVAL_SEEDS="${EVAL_SEEDS:-5}"
RECORD_ROBOTS="${RECORD_ROBOTS:-0 1}"         # robots filmed per run
RECORD_DURATION="${RECORD_DURATION:-}"        # seconds of video per run (empty: record.py's default tour)

# Tiny settings for a dry run of the whole pipeline (SMOKE=1 hpc/submit_pipeline.sh, or
# SMOKE=1 bash hpc/<stage>.slurm on a workstation).
if [[ "${SMOKE:-0}" == "1" ]]; then
  SEEDS="0"; PPO_NUM_ENVS=64; PPO_ITERATIONS_FLAT=3; PPO_ITERATIONS_ROUGH=3
  SM_NUM_ENVS=256; SM_ITERATIONS=120; SM_SEGMENTS=2; SM_SEGMENT_HOURS=1
  EVAL_SEEDS=1; RECORD_ROBOTS="0"; RECORD_DURATION=4
fi
