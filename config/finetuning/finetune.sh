#!/bin/bash

CONFIG_FILE=$1  # path to the config file

# Modify to match your python environment activation command(s)
source /etc/profile.d/modules.sh
if [ "$USER" == "am84" ]; then
    module load cellgen/conda
    conda activate /software/cellgen/team361/am84/envs/nemo
else
    echo "User $USER not supported"
    exit 1
fi

python source-code/src/app/finetune.py \
    --fname ${CONFIG_FILE}