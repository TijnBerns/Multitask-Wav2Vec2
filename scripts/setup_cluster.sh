#! /usr/bin/env bash

# set variable to path where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# make a directory on the ceph file system to store logs and checkpoints
# and make a symlink to access it directly from the root of the project
mkdir -p /ceph/csedu-scratch/course/IMC030_MLIP/users/"$USER"/checkpoints
chmod 700 /ceph/csedu-scratch/course/IMC030_MLIP/users/"$USER" # only you can access
ln -sfn /ceph/csedu-scratch/course/IMC030_MLIP/users/"$USER"/checkpoints "$SCRIPT_DIR"/../checkpoints


# # make sure that there's also a virtual environment
# # on the GPU nodes
# echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN47 ###"
# ssh cn47 "
#   source .profile
#   cd $PWD;
#   ./setup_venv.sh;
# "

# echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN48 ###"
# ssh cn48 "
#   source .profile
#   cd $PWD;
#   ./setup_venv.sh;
# "