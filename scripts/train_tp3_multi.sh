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

# Train on train-clean-full with speaker ids instead of speaker change symbols
source "$project_dir"/venv/bin/activate
srun python "$project_dir"/src/train.py \
    --vocab_path=src/models/vocab_spid.json \
    --train_trans=/scratch/tberns/asr/data/train-clean-100-no-rep/trans-id.csv \
    --train_trans=/scratch/tberns/asr/data/train-clean-100-rep/trans-id.csv \
    --val_trans=/scratch/tberns/asr/data/val-clean-no-rep/trans-st.csv \
    --val_trans=/scratch/tberns/asr/data/val-clean-rep/trans-st.csv
