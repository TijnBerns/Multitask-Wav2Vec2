#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --output=./logs/slurm/%J.out
#SBATCH --error=./logs/slurm/%J.err

project_dir=.

# Train on train-clean-100 with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
python "$project_dir"/src/train.py \
    --train_trans=/scratch/tberns/asr/data/LibriSpeech/train-clean-100.trans.csv \
    --val_trans=/scratch/tberns/asr/data/LibriSpeech/val-clean.trans.csv \