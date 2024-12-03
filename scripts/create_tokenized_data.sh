#!/bin/bash

CONFIG_FILE=$1  # path to the config file
OVERWRITE=$2  # overwrite the token dictionary files

# Add settings for other users

if [ "$USER" == "am84" ]; then
    module load cellgen/conda
    conda activate nichejepa-reproducibility
elif [ "$USER" == "sb75" ]; then
    module load cellgen/conda
    conda activate nichejepa_new
fi

python analysis/tokenization/create_cell_tokenized_dataset.py \
    --config_file ${CONFIG_FILE} \
    --override \
    overwrite=${OVERWRITE}