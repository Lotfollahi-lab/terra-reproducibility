#!/bin/bash

CONFIG_FILE=$1  # path to the config file
CORES=$2    # number of cores

# Modify to match your python environment activation command(s)
module load cellgen/conda
conda activate nichejepa-reproducibility

python analysis/data_preparation/harmonize_bronze_to_silver.py \
    --config_file ${CONFIG_FILE}