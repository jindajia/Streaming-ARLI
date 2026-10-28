#!/bin/bash
# Run several *independent* single-GPU training processes in one job (one per GPU).  Not a multi-GPU algorithm:
# each process is a separate run with its own run dir.  CMDS = ';'-separated list, exported before sbatch:
#   export CMDS="python -m arli.train_continue --config configs/s1_A_seed0.yaml --max_wall_hours 0.85;python -m ..."
#   sbatch -J s1_AC -p gpu-debug --gpus-per-node=2 --cpus-per-task=16 --mem=120g --time=00:59:00 cluster/multi.sbatch
#SBATCH --account=r01156
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --time=00:59:00
#SBATCH --output=/N/scratch/jindjia/arli/logs/%x_%j.out
set -o pipefail
export ARLI_SCRATCH=/N/scratch/jindjia/arli OPENPI_DATA_HOME=/N/scratch/jindjia/arli/openpi_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_MODE=disabled
export HF_HOME=/N/scratch/jindjia/hf TORCH_HOME=/N/scratch/jindjia/torch TRITON_CACHE_DIR=/N/scratch/jindjia/triton
source /N/slate/jindjia/Btools/miniforge3/etc/profile.d/conda.sh
conda activate /N/scratch/jindjia/envs/arli
cd /N/slate/jindjia/SD/VLA/streaming_arli
echo "job=$SLURM_JOB_ID name=$SLURM_JOB_NAME node=$(hostname) start=$(date) git=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
IFS=';' read -ra C <<< "$CMDS"
NG=$(nvidia-smi -L | wc -l); NC=$(( SLURM_CPUS_PER_TASK / ${#C[@]} ))
i=0; pids=()
for cmd in "${C[@]}"; do
  g=$(( i % NG ))
  echo "[$i] GPU $g: $cmd"
  ( export CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC ARLI_OWNER="${SLURM_JOB_ID}_$i"
    eval "$cmd" ) > /N/scratch/jindjia/arli/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_p$i.out 2>&1 &
  pids+=($!); i=$((i+1))
done
for p in "${pids[@]}"; do wait $p; echo "pid $p exit $?"; done
echo "end=$(date)"
