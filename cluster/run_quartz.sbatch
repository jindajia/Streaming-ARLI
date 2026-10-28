#!/bin/bash
# Quartz (H100) variant of run.sbatch; submit from BR200 via: ssh quartz.uits.iu.edu "cd <repo> && CMD=... sbatch ..."
#   export CMD="python -m arli.train --config configs/r2_arli_seed0.yaml"; sbatch -J r2_arli --time=1-12:00:00 cluster/run.sbatch
#SBATCH --account=r01156
#SBATCH --partition=h100-debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --time=01:00:00
#SBATCH --output=/N/scratch/jindjia/arli/logs/%x_%j.out
set -eo pipefail
export ARLI_SCRATCH=/N/scratch/jindjia/arli
export OPENPI_DATA_HOME=$ARLI_SCRATCH/openpi_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
# EGL must render on the GPU Slurm allocated (hard-coding 0 aborts when the job gets another device)
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
export HF_HOME=/N/scratch/jindjia/hf TORCH_HOME=/N/scratch/jindjia/torch TRITON_CACHE_DIR=/N/scratch/jindjia/triton
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB_MODE=disabled
source /N/slate/jindjia/Btools/miniforge3/etc/profile.d/conda.sh
conda activate /N/scratch/jindjia/envs/arli
cd /N/slate/jindjia/SD/VLA/streaming_arli
echo "job=$SLURM_JOB_ID name=$SLURM_JOB_NAME node=$(hostname) start=$(date) git=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
echo "CMD: $CMD"
eval "$CMD"
echo "end=$(date) exit=$?"
