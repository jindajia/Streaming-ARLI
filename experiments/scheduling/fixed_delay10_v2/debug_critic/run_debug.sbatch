#!/bin/bash
# Time-bounded critic diagnostics (one GPU, hard 20-minute wall limit including initialization).
#   sbatch -J sch2_critic_debug -p gpu-debug --time=00:20:00 debug_critic/run_debug.sbatch
#SBATCH --account=r01156
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --output=/N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/logs/%x_%j.out
set -o pipefail
WT=/N/slate/jindjia/SD/VLA/streaming_arli_debug
export ARLI_SCRATCH=/N/scratch/jindjia/arli OPENPI_DATA_HOME=/N/scratch/jindjia/arli/openpi_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_MODE=disabled
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
export HF_HOME=/N/scratch/jindjia/hf TORCH_HOME=/N/scratch/jindjia/torch TRITON_CACHE_DIR=/N/scratch/jindjia/triton
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"
source /N/slate/jindjia/Btools/miniforge3/etc/profile.d/conda.sh
conda activate /N/scratch/jindjia/envs/arli
cd $WT/experiments/scheduling/fixed_delay10_v2
export PYTHONPATH=$WT/src:$WT/experiments/scheduling/src:$WT/experiments/scheduling/fixed_delay10_v2/src
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date) git=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python debug_critic/critic_debug.py --out /N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/debug_critic/job_$SLURM_JOB_ID --deadline_min 17.0; rc=$?
echo "end=$(date) exit=$rc"
exit $rc
