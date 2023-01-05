#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=3
#SBATCH --time=72:00:00
#SBATCH --output=./logs/slurm/%J.out
#SBATCH --error=./logs/slurm/%J.err

project_dir=.

# Train on train-full with no speaker change symbols or ids
source "$project_dir"/venv/bin/activate
python "$project_dir"/src/train.py \
    --vocab_path=/home/tberns/Speaker_Change_Recognition/src/models/vocab_base.json \
    --train_trans=/scratch/tberns/asr/data/train-clean-100-no-rep/trans.csv \
    --train_trans=/scratch/tberns/asr/data/train-clean-100-repB/trans.csv \
    --val_trans=/scratch/tberns/asr/data/val-clean-no-rep/trans.csv \
    --val_trans=/scratch/tberns/asr/data/val-clean-repB/trans.csv