#!/bin/bash

CONFIG_FILE=$1  # path to the config file
CORES=$2    # number of cores

# Modify to match your python environment activation command(s)
module load cellgen/conda
conda activate nichejepa-reproducibility

python analysis/tokenization/create_tokenized_dataset.py \
    --config_file ${CONFIG_FILE} \
    --override \
    nproc=${CORES} \
    overwrite=True