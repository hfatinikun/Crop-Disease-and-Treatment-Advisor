#!/bin/bash
#SBATCH --job-name=retrain_dice
#SBATCH --partition=pgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G 
#SBATCH --time=12:00:00
#SBATCH --output=/data/users/hfatinikun/deep_learning/Crop-Disease-and-Treatment-Advisor/log_files/6_run_retrain_dice/output_%J.o
#SBATCH --error=/data/users/hfatinikun/deep_learning/Crop-Disease-and-Treatment-Advisor/log_files/6_run_retrain_dice/error_%J.e
#SBATCH --mail-user=heritage.fatinikun@students.unibe.ch
#SBATCH --mail-type=begin,end,fail

#dir and paths
WORKDIR="/data/users/hfatinikun/deep_learning/Crop-Disease-and-Treatment-Advisor"
SCRIPT_PY="${WORKDIR}/src/retrain_dice_loss.py"
ENVIRONMENT="${WORKDIR}/tools/envs/dl_project/bin/activate"

cd ${WORKDIR}

#load pytorch
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1

#activate environment 
source ${ENVIRONMENT}

#prevent mixing ~/.local packages with venv
export PYTHONNOUSERSITE=1 

#run code
python ${SCRIPT_PY}