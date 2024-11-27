#!/bin/bash

CONFIG_FILE=$1  # path to the config file
OVERWRITE=$2  # overwrite the token dictionary files

# Modify to match your python environment activation command(s)
module load cellgen/conda
conda activate nichejepa_new

python analysis/tokenization/create_cell_tokenized_dataset.py \
    --config_file ${CONFIG_FILE} \
    --override \
    overwrite=${OVERWRITE}