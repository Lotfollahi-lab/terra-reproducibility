#!/bin/bash

DATASET_NAME=$1    # dataset name
TOKENIZER_NAME=$2  # tokenizer name
CONFIG_FILE=$3  # path to the config file
CORES=$4    # number of cores

module load cellgen/conda
conda activate nichejepa-reproducibility

echo "Creating ${TOKENIZER_NAME} tokenized data for ${DATASET_NAME} dataset with ${CORES} cores"

case $TOKENIZER_NAME in
    neighborhood)
        python reproducibility/analysis/tokenization/create_cell_neighborhood_tokenized_dataset.py \
            --config_file ${CONFIG_FILE} \
            --override \
            nproc=${CORES}
        ;;
    graph)
        python reproducibility/analysis/tokenization/create_cell_graph_tokenized_dataset.py \
            --config_file ${CONFIG_FILE} \
            --override \
            nproc=${CORES}
        ;;
    *)
        echo "Invalid tokenizer name"
        exit 1
        ;;
esac