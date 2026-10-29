#!/bin/bash
# R-sensitivity probe launcher (gpu-debug, one GPU).  STAGE=sources|matrix
#   STAGE=sources sbatch -J rsens_sources -p gpu-debug --time=00:30:00 r_sensitivity_probe/run.sbatch
#   STAGE=matrix  sbatch -J rsens_matrix  -p gpu-debug --time=00:59:00 r_sensitivity_probe/run.sbatch
#SBATCH --account=r01156
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --output=/N/scratch/jindjia/arli/runs_scheduling/r_sensitivity_probe/logs/%x_%j.out
set -o pipefail
WT=/N/slate/jindjia/SD/VLA/streaming_arli_rsens
export ARLI_SCRATCH=/N/scratch/jindjia/arli OPENPI_DATA_HOME=/N/scratch/jindjia/arli/openpi_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_MODE=disabled
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
export HF_HOME=/N/scratch/jindjia/hf TORCH_HOME=/N/scratch/jindjia/torch TRITON_CACHE_DIR=/N/scratch/jindjia/triton
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"
source /N/slate/jindjia/Btools/miniforge3/etc/profile.d/conda.sh
conda activate /N/scratch/jindjia/envs/arli
cd $WT/experiments/scheduling
export PYTHONPATH=$WT/src:$WT/experiments/scheduling/src:$WT/experiments/scheduling/fixed_delay10_v2/src
echo "job=$SLURM_JOB_ID stage=$STAGE node=$(hostname) start=$(date) git=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python r_sensitivity_probe/rsens.py --stage $STAGE --deadline_min 54; rc=$?
echo "end=$(date) exit=$rc"
exit $rc
