#!/bin/bash
# SCH-01 launcher (worktree streaming_arli_sched, branch experiment/queue-scheduler). Export CMD before sbatch.
#   export CMD="python scripts/gpu_checks.py"; sbatch -J sch01_checks -p gpu-debug --time=00:59:00 cluster/run_sch.sbatch
#SBATCH --account=r01156
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g
#SBATCH --output=/N/scratch/jindjia/arli/runs_scheduling/logs/%x_%j.out
set -o pipefail
export ARLI_SCRATCH=/N/scratch/jindjia/arli OPENPI_DATA_HOME=/N/scratch/jindjia/arli/openpi_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_MODE=disabled
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}
export HF_HOME=/N/scratch/jindjia/hf TORCH_HOME=/N/scratch/jindjia/torch TRITON_CACHE_DIR=/N/scratch/jindjia/triton
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
source /N/slate/jindjia/Btools/miniforge3/etc/profile.d/conda.sh
conda activate /N/scratch/jindjia/envs/arli
cd /N/slate/jindjia/SD/VLA/streaming_arli_sched/experiments/scheduling
export PYTHONPATH=/N/slate/jindjia/SD/VLA/streaming_arli_sched/src:/N/slate/jindjia/SD/VLA/streaming_arli_sched/experiments/scheduling/src
echo "job=$SLURM_JOB_ID name=$SLURM_JOB_NAME partition=$SLURM_JOB_PARTITION node=$(hostname) start=$(date) git=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "CMD: $CMD"
eval "$CMD"; rc=$?
echo "end=$(date) exit=$rc"
exit $rc
