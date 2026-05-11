#!/bin/bash
#SBATCH --job-name=report_figures
#SBATCH --partition=pgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/data/users/hfatinikun/deep_learning/Crop-Disease-and-Treatment-Advisor/log_files/8_make_report_figures/output_%J.o
#SBATCH --error=/data/users/hfatinikun/deep_learning/Crop-Disease-and-Treatment-Advisor/log_files/8_make_report_figures/error_%J.e

WORKDIR="/data/users/hfatinikun/deep_learning/Crop-Disease-and-Treatment-Advisor"
ENVIRONMENT="${WORKDIR}/tools/envs/dl_project/bin/activate"

cd ${WORKDIR}

module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1

source ${ENVIRONMENT}

#prevent mixing ~/.local packages with venv
export PYTHONNOUSERSITE=1 

python src/make_report_figures.py \
  --stage1-log outputs/checkpoints/train/20260508_120557/training_log.csv \
  --normal-log outputs/checkpoints/retrain/stage3_20260509_071022/training_log.csv \
  --dice-log outputs/checkpoints/retrain_dice/stage3_20260509_205728/training_log.csv \
  --output-dir outputs/report_figures