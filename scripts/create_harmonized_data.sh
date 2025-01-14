#!/bin/bash

CONFIG_FILE=$1  # path to the config file

# # Add settings for different users
# if [ "$USER" == "am84" ]; then
#     module load cellgen/conda
#     conda activate nichejepa-reproducibility
# elif [ "$USER" == "sb75" ]; then
#     module load cellgen/conda
#     conda activate nichejepa_new
# fi

python analysis/data_preparation/harmonize_bronze_to_silver.py \
    --config_file ${CONFIG_FILE}