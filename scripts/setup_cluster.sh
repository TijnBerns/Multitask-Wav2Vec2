#! /usr/bin/env bash

# set variable to path where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# make a directory on the ceph file system to store logs and checkpoints
# and make a symlink to access it directly from the root of the project
mkdir -p /ceph/csedu-scratch/other/"$USER"/lightning_logs
ln -sfn /ceph/csedu-scratch/other/"$USER"/lightning_logs "$SCRIPT_DIR"/../lightning_logs

mkdir -p /ceph/csedu-scratch/other/"$USER"/logs
mkdir -p /ceph/csedu-scratch/other/"$USER"/logs/measures
mkdir -p /ceph/csedu-scratch/other/"$USER"/logs/preds
ln -sfn /ceph/csedu-scratch/other/"$USER"/logs "$SCRIPT_DIR"/../logs

mkdir -p /ceph/csedu-scratch/other/"$USER"/embeddings
ln -sfn /ceph/csedu-scratch/other/"$USER"/embeddings "$SCRIPT_DIR"/../embeddings

mkdir -p /scratch/"$USER"/asr/data/trials
ln -sfn /scratch/"$USER"/asr/data/trials "$SCRIPT_DIR"/../trials

echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN84 ###"
./setup_venv.sh

echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN77 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn77 rsync -a cn84:/scratch/"$USER"/ /scratch/"$USER"/

echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN47 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn47 rsync -a cn84:/scratch/"$USER"/ /scratch/"$USER"/

echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN48 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn48 rsync -a cn84:/scratch/"$USER"/ /scratch/"$USER"/