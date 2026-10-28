#!/bin/bash
# Run ONCE on the Adroit login node (compute nodes have no internet):
#   bash setup_adroit.sh
set -e

module purge
module load anaconda3/2026.7

conda create --name ball-env python=3.12 -y
conda activate ball-env

# CPU-only torch: smaller download, and the networks are too small to gain from a GPU
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -c "import mujoco, torch, gymnasium, tensorboard; print('mujoco', mujoco.__version__, '| torch', torch.__version__, '| setup OK')"
