#!/bin/bash
#SBATCH --job-name=pseudo_mask_gen
#SBATCH --partition=pibu_el8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G 
#SBATCH --time=6:00:00
#SBATCH --output=/data/users/aikiror/deepLearning/project/logFiles/4_run_pseudo_mask_gen/output_%J.o
#SBATCH --error=/data/users/aikiror/deepLearning/project/logFiles/4_run_pseudo_mask_gen/error_%J.e
#SBATCH --mail-user=amo.ikiror@students.unibe.ch
#SBATCH --mail-type=begin,end,fail

#4_run_pseudo_mask_gen

#dir and paths
WORKDIR="/data/users/aikiror/deepLearning/project"
SCRIPT_PY="${WORKDIR}/src/pseudo_mask_gen_4.py"
ENVIRONMENT="${WORKDIR}/tools/envs/dl_project/bin/activate"

cd ${WORKDIR}

#load pytorch
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1

#activate environment 
source ${ENVIRONMENT}

#run code
python ${SCRIPT_PY}