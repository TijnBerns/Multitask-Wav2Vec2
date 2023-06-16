#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/%J.out
#SBATCH --error=./logs/%J.out

project_dir=.

# Train on train-clean-100 with speaker change at start
source "$project_dir"/venv/bin/activate
srun python "$project_dir"/src/train.py \
    --vocab_path=src/models/vocab_spch.json \
    --train_trans=/scratch/tberns/asr/data/LibriSpeech/train-clean-100.trans-st.csv \
    --val_trans=/scratch/tberns/asr/data/LibriSpeech/val-clean.trans-st.csv \