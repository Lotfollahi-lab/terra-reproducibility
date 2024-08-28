#!/bin/bash

DATASET_NAME=$1    # dataset name
CONFIG_FILE=$2  # path to the config file
CORES=$3    # number of cores

module load cellgen/conda
conda activate nichejepa-reproducibility

echo "Creating tokenized data for ${DATASET_NAME} dataset with ${CORES} cores"

python reproducibility/analysis/tokenization/create_cell_neighborhood_tokenized_dataset.py \
    --config_file ${CONFIG_FILE} \
    --override \
    nproc=${CORES}