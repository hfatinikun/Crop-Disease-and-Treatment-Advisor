#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --partition=pgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G 
#SBATCH --time=8:00:00
#SBATCH --output=/data/users/aikiror/deepLearning/project/logFiles/3_train_model/output_%J.o
#SBATCH --error=/data/users/aikiror/deepLearning/project/logFiles/3_train_model/error_%J.e
#SBATCH --mail-user=amo.ikiror@students.unibe.ch
#SBATCH --mail-type=begin,end,fail


#utilizing pgpu partition as it has gpu and pibu-el8 doesnt.
#also changed cpus-per-gpu

WORKDIR="/data/users/aikiror/deepLearning/project"
TRAIN_PY="${WORKDIR}/src/train.py"
ENVIRONMENT="${WORKDIR}/tools/envs/dl_project/bin/activate"

cd ${WORKDIR}

#load pytorch
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1

#activate environment
source ${ENVIRONMENT}

#run training code
python ${TRAIN_PY}
