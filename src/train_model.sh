#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --partition=pgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G 
#SBATCH --time=4:00:00
#SBATCH --output=/data/users/aikiror/deepLearning/project/logFiles/train_model/output_%J.o
#SBATCH --error=/data/users/aikiror/deepLearning/project/logFiles/train_model/error_%J.e
#SBATCH --mail-user=amo.ikiror@students.unibe.ch
#SBATCH --mail-type=begin,end,fail


#utilizing pgpu partition as it has gpu and pibu-el8 doesnt.
#also changed cpus-per-gpu

WORKDIR="/data/users/aikiror/deepLearning/project"
TRAIN_PY=""
ENVIRONMENT=""

#load pytorch
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1

#activate environment
source activate ${ENVIRONMENT}

#run training code
python train.py
