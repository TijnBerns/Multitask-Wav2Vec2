#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=./logs/%J.out
#SBATCH --error=./logs/%J.out

project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
python "$project_dir"/src/train.py \
    --vocab_path=/home/tberns/Speaker_Change_Recognition/src/models/vocab_base.json \
    --train_trans=/scratch/tberns/asr/data/train-clean-100-no-rep/trans.csv \
    --train_trans=/scratch/tberns/asr/data/train-clean-100-rep/trans.csv \
    --val_trans=/scratch/tberns/asr/data/val-clean-no-rep/trans.csv \
    --val_trans=/scratch/tberns/asr/data/val-clean-rep/trans.csv