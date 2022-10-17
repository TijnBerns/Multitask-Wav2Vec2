#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=./logs/slurm/%J.out
#SBATCH --error=./logs/slurm/%J.err


project_dir=.

if [ $# -eq 0 ]
  then
    echo "No checkpoint path provided."
    exit 1

# Evaluate on clean dev set with 
source "$project_dir"/venv/bin/activate
python "$project_dir"/src/eval.py --checkpoint_path=$1 \
    --dataset=/scratch/tberns/asr/data/LibriSpeech/dev-clean \
    --trans_path=/scratch/tberns/asr/data/LibriSpeech/dev-clean/trans-st.csv \
    --vocab_path=/home/tberns/Speaker_Change_Recognition/src/models/vocab_spid.json \
    --asr_only=True
