#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --output=./logs/%J.out
#SBATCH --error=./logs/%J.out


project_dir=.

# execute train CLI
source "$project_dir"/venv/bin/activate
python "$project_dir"/src/eval.py \
    --version_number=$1 \
    --trans_file=trans-st.csv \
    --vocab_file=vocab_spid.json
