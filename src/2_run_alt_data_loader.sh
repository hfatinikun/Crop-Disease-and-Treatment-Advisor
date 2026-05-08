#!/bin/bash
#SBATCH --job-name=load
#SBATCH --partition=pibu_el8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G 
#SBATCH --time=4:00:00
#SBATCH --output=/data/users/aikiror/deepLearning/project/logFiles/2_run_alt_data_loader/output_%J.o
#SBATCH --error=/data/users/aikiror/deepLearning/project/logFiles/2_run_alt_data_loader/error_%J.e
#SBATCH --mail-user=amo.ikiror@students.unibe.ch
#SBATCH --mail-type=begin,end,fail

#dir and paths
WORKDIR="/data/users/aikiror/deepLearning/project"
SCRIPT_PY="${WORKDIR}/src/alt_data_loader_2.py"
ENVIRONMENT="${WORKDIR}/tools/envs/dl_project/bin/activate"

cd ${WORKDIR}

#load pytorch
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1

#activate environment 
source ${ENVIRONMENT}

#run code
python ${SCRIPT_PY}
