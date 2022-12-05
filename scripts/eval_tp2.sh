#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --output=./logs/slurm/%J.out
#SBATCH --error=./logs/slurm/%J.err


project_dir=.

# execute train CLI
source "$project_dir"/venv/bin/activate
python "$project_dir"/src/eval.py \
    --version_number=$1 \
    --trans_file=trans-st.csv \
    --vocab_file=vocab_spch.json
