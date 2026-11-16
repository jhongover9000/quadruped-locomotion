#!/bin/bash
#SBATCH --job-name=ic_nn_refine
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joseph.hong@nyu.edu
#SBATCH --output=/scratch/jhh508/logs/%x_%j.out
#SBATCH --error=/scratch/jhh508/logs/%x_%j.err

SCRATCH_ROOT=/scratch/jhh508
PROJECT_DIR=$SCRATCH_ROOT/Neuroinformatics/intracortical_inference
DATA_ROOT=$PROJECT_DIR/data
OUT_DIR=$PROJECT_DIR/outputs
SUB_CSV=submission.csv
mkdir -p "$OUT_DIR"

# ---- choose which script to run ----
# This MUST be the multiprocessing-enabled script that supports --n_jobs
REG_SCRIPT="nn_refinement.py"

# ---- modules / conda ----
module purge || true
module load miniconda || true
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
echo "===================== Activating Conda Env ======================="
conda activate neuro

# ---- avoid CPU oversubscription when using multiprocessing ----
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"

echo "============================================================"
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Project: $PROJECT_DIR"
echo "Data root: $DATA_ROOT"
echo "Script: $REG_SCRIPT"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "============================================================"

TUNE_LOG="$OUT_DIR/tune_${SLURM_JOB_ID}.log"

echo "==== [1] Generating Data ===="
python "$REG_SCRIPT" \
  --data_root "$DATA_ROOT" \
  --mode generate_data \
  --cg_lag 2 --cg_cov_reg 0.01 --blend_alpha 0.3 \
  --global_cov_path ./data/global_cov.npy \
  --temporal_lag 10 --n_seeds 3

echo "==== [2] NN Refinement ===="
python "$REG_SCRIPT" \
  --data_root "$DATA_ROOT" \
  --mode train \
  --device cuda --epochs 80 --batch_size 512 \
  --hidden_dim 512 --n_blocks 2 --dropout 0.2 \
  --patience 50 --temporal_lag 10

echo "==== [3] Evaluation ===="
python nn_refinement.py --data_root ./data/ --mode eval \
    --device cuda --checkpoint refinement_model.pt \
    --cg_lag 2 --cg_cov_reg 0.01 --blend_alpha 0.3 \
    --global_cov_path ./data/global_cov.npy --temporal_lag 10

echo "==== [4] Generate Submission ===="
python nn_refinement.py --data_root ./data/ --mode submit \
    --device cuda --checkpoint refinement_model.pt \
    --cg_lag 2 --cg_cov_reg 0.01 --blend_alpha 0.3 \
    --global_cov_path ./data/global_cov.npy --out_csv "$SUB_CSV" --temporal_lag 10

echo "DONE. Submission written to: $SUB_CSV"