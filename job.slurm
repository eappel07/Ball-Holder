#!/bin/bash
# Full training run
#   sbatch job.slurm
#SBATCH --job-name=ball-td3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --mail-type=end,fail
# To try a GPU instead, remove one # from the next line:
##SBATCH --gres=gpu:1

module purge
module load anaconda3/2026.7
conda activate ball-env

cd /scratch/network/$USER/ball_holder

# -u = print immediately, so the slurm-*.out log updates live
python -u main.py
