#!/bin/bash
#SBATCH --job-name=filter
#SBATCH --partition=pibu_el8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G 
#SBATCH --time=4:00:00
#SBATCH --output=/data/users/aikiror/deepLearning/project/logFiles/1_run_filtering_datasets/output_%J.o
#SBATCH --error=/data/users/aikiror/deepLearning/project/logFiles/1_run_filtering_datasets/error_%J.e
#SBATCH --mail-user=amo.ikiror@students.unibe.ch
#SBATCH --mail-type=begin,end,fail

#dir and paths
WORKDIR="/data/users/aikiror/deepLearning/project"
SCRIPT_PY="${WORKDIR}/src/1_filtering_datasets.py"
ENVIRONMENT="${WORKDIR}/tools/envs/dl_project/bin/activate"

cd ${WORKDIR}

#load pytorch
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1

#activate environment 
source ${ENVIRONMENT}

#run code
python ${SCRIPT_PY}