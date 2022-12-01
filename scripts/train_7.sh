#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00
#SBATCH --output=./logs/slurm/%J.out
#SBATCH --error=./logs/slurm/%J.err

project_dir=.

# Train on train-full with speaker change symbols
source "$project_dir"/venv/bin/activate
python "$project_dir"/src/train.py \
    --vocab_path=src/models/vocab_spch.json \
    --train_trans=/scratch/tberns/asr/data/train-clean-100-no-rep/trans-st.csv \
    --train_trans=/scratch/tberns/asr/data/train-clean-100-rep/trans-st.csv \
    --val_trans=/scratch/tberns/asr/data/val-clean-no-rep/trans-st.csv \
    --val_trans=/scratch/tberns/asr/data/val-clean-rep/trans-st.csv
