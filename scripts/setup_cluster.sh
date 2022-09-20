#! /usr/bin/env bash
set -e




# make sure that there's also a virtual environment
# on the GPU nodes
echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN47 ###"
ssh cn47 "
  source .profile
  cd $PWD;
  ./setup_venv.sh;
"

echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN48 ###"
ssh cn48 "
  source .profile
  cd $PWD;
  ./setup_venv.sh;
"