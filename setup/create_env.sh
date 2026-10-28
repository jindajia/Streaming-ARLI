#!/bin/bash
# Build the `arli` conda env (python 3.11, JAX 0.5.0 + openpi@a6d2400 from the pinned DSRL submodule).
# Env lives on scratch (slate is near quota). Mirrors dsrl_pi0 README install, minus LIBERO/robosuite/tensorflow.
set -eo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENV=${ARLI_ENV:-/N/scratch/jindjia/envs/arli}
source /N/slate/jindjia/Btools/miniforge3/etc/profile.d/conda.sh
export PIP_CACHE_DIR=/N/scratch/jindjia/pipcache
[ -d "$ENV" ] || conda create -y --offline -p "$ENV" python=3.11.11
PY="$ENV/bin/python"  # call the env interpreter explicitly (conda activate is unreliable in batch shells)
$PY -m ensurepip --upgrade && $PY -m pip install -U pip
# torch CPU only (openpi imports torch in a few utilities; DSRL installs the CPU wheel too)
$PY -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
$PY -m pip install "jax[cuda12]==0.5.0" flax==0.10.2 optax==0.2.4 distrax==0.1.5 orbax-checkpoint==0.11.1 \
    ml_collections==1.0.0 ml_dtypes==0.5.1 optree==0.15.0 chex \
    gym==0.26.2 gym-notices==0.0.8 gymnasium==1.0.0 gym-aloha==0.1.1 dm-control==1.0.14 mujoco==2.3.7 \
    numpy==1.26.4 pillow==10.4.0 imageio==2.37.0 imageio-ffmpeg matplotlib==3.9.3 pandas==2.2.3 scipy==1.15.2 \
    easydict==1.13 einops==0.8.1 tqdm wandb==0.19.9 h5py==3.13.0 opencv-python-headless==4.11.0.86
$PY -m pip install -e "$REPO/third_party/dsrl_pi0"                      # jaxrl2
$PY -m pip install -e "$REPO/third_party/dsrl_pi0/openpi/packages/openpi-client"
$PY -m pip install -e "$REPO/third_party/dsrl_pi0/openpi"
$PY -m pip install -e "$REPO"                                            # streaming_arli
$PY -m pip install numpy==1.26.4   # later installs pull numpy 2.x; DSRL-era stack expects 1.26
$PY -m pip freeze > "$REPO/setup/pip_freeze_arli.txt"
$PY -c "import jax, flax, openpi, jaxrl2, gym_aloha, mujoco; print('ok', jax.__version__, flax.__version__, mujoco.__version__)"
