#!/bin/bash

EXPERIMENT_NAME=$1    # dataset name
TOKENIZER_NAME=$2  # tokenizer name
CONFIG_FILE=$3  # path to the config file
CORES=$4    # number of cores

# Modify to match your python environment activation command(s)
module load cellgen/conda
conda activate nichejepa-reproducibility

case $TOKENIZER_NAME in
    cell-neighborhood-tokenizer)
        python analysis/tokenization/create_cell_neighborhood_tokenized_dataset.py \
            --config_file ${CONFIG_FILE} \
            --override \
            nproc=${CORES} \
            overwrite=True
        ;;
    graph)
        python analysis/tokenization/create_cell_graph_tokenized_dataset.py \
            --config_file ${CONFIG_FILE} \
            --override \
            nproc=${CORES} \
            overwrite=True
        ;;
    *)
        echo "Invalid tokenizer name"
        exit 1
        ;;
esac