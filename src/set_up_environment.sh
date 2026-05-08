#!/bin/bash
#SBATCH --job-name=filter
#SBATCH --partition=pibu_el8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G 
#SBATCH --time=4:00:00
#SBATCH --output=/data/users/aikiror/deepLearning/project/logFiles/set_up_environment/output_%J.o
#SBATCH --error=/data/users/aikiror/deepLearning/project/logFiles/set_up_environment/error_%J.e
#SBATCH --mail-user=amo.ikiror@students.unibe.ch
#SBATCH --mail-type=begin,end,fail

#sets up environment

WORKDIR="/data/users/aikiror/deepLearning/project"
TOOLS="$WORKDIR/tools"
mkdir -p ${TOOLS}

#load pytorch
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1

#create environment; inherit packages from pytorch
python -m venv ${TOOLS}/envs/dl_project --system-site-packages

#activate environment 
source ${TOOLS}/envs/dl_project/bin/activate

#only install missing packages
pip install --no-cache-dir opencv-python

pip install --no-cache-dir torchvision==0.11.0 --no-deps

pip install --no-cache-dir scikit-learn

pip install --no-cache-dir Pillow

pip install --no-cache-dir "albumentations==1.3.1"

pip install --no-cache-dir pandas

pip install --no-cache-dir tqdm

pip install --no-cache-dir "numpy==1.26.4"
