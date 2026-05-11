#!/bin/bash
#SBATCH --job-name=evaluate_vis
#SBATCH --partition=pgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/data/users/hfatinikun/deep_learning/Crop-Disease-and-Treatment-Advisor/log_files/7_evaluate_visualize/output_%J.o
#SBATCH --error=/data/users/hfatinikun/deep_learning/Crop-Disease-and-Treatment-Advisor/log_files/7_evaluate_visualize/error_%J.e

WORKDIR="/data/users/hfatinikun/deep_learning/Crop-Disease-and-Treatment-Advisor"
ENVIRONMENT="${WORKDIR}/tools/envs/dl_project/bin/activate"

cd ${WORKDIR}

module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1

source ${ENVIRONMENT}

#prevent mixing ~/.local packages with venv
export PYTHONNOUSERSITE=1 

pip uninstall -y matplotlib
pip install matplotlib==3.5.3
pip install "pillow>=8.3.2,<10"

python src/evaluate_visualise.py \
  --stage1 outputs/checkpoints/train/20260508_120557/best_model.pth \
  --stage3 outputs/checkpoints/retrain_dice/stage3_20260509_205728/best_model.pth \
  --output-dir outputs/evaluation_dice \
  --n-vis 20