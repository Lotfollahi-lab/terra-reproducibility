#!/bin/bash

GROUP="team361"
SCRIPT="reproducibility/scripts/create_tokenized_data.sh"

DATASET_NAME=$1
TOKENIZER_NAME=$2 # choose between "neighborhood" and "graph"
CORES=$3

case $DATASET_NAME in
    sim1_1105fts_10000locs)
        QUEUE="normal"
        RAM="2G"
        TIME="0:05"
        CONFIG_FILE="reproducibility/config/cell-${TOKENIZER_NAME}-tokenization_sim1_1105fts_10000locs_read-depth_shifted-log.yaml"
        ;;
    sim1_1105genes_100klocs_strongincrements)
        QUEUE="normal"
        RAM="25G"
        TIME="0:10"
        CONFIG_FILE="reproducibility/config/cell-${TOKENIZER_NAME}-tokenization_sim1_1105genes_100klocs_strongincrements_read-depth_shifted-log.yaml"
        ;;    
    merfish_mouse_brain_8-batch)
        QUEUE="normal"
        RAM="25G"
        TIME="0:10"
        CONFIG_FILE="reproducibility/config/cell-${TOKENIZER_NAME}-tokenization_merfish_mouse_brain_8-batch_read-depth_shifted-log.yaml"
        ;;
    merfish_mouse_brain_16-batch)
        QUEUE="normal"
        RAM="45G"
        TIME="0:20"
        CONFIG_FILE="reproducibility/config/cell-${TOKENIZER_NAME}-tokenization_merfish_mouse_brain_16-batch_read-depth_shifted-log.yaml"
        ;;
    merfish_mouse_brain_64-batch)
        QUEUE="normal"
        RAM="60G"
        TIME="0:30"
        CONFIG_FILE="reproducibility/config/cell-${TOKENIZER_NAME}-tokenization_merfish_mouse_brain_64-batch_read-depth_shifted-log.yaml"
        ;;
    *)
        echo "Invalid dataset name"
        exit 1
        ;;
esac

CWD="/lustre/scratch126/cellgen/team361/am84/NicheJEPA/"
mkdir -p "logs/${DATASET_NAME}/profile-nprocs"

OUTPUT_FILE="logs/${DATASET_NAME}/profile-nprocs/cell_${TOKENIZER_NAME}_tokenization_${CORES}_%J.out"
ERROR_FILE="logs/${DATASET_NAME}/profile-nprocs/cell_${TOKENIZER_NAME}_tokenization_${CORES}_%J.err"
JOB_NAME="${DATASET_NAME}_cell_${TOKENIZER_NAME}_tokenization_${CORES}"

echo "Tokenizing ${DATASET_NAME} with Cell ${TOKENIZER_NAME} Rank using ${CORES} cores..."

bsub \
    -G "${GROUP}" -q "${QUEUE}" -n "${CORES}" \
    -M "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}]" \
    -W "${TIME}" \
    -cwd "${CWD}" \
    -o "${OUTPUT_FILE}" \
    -e "${ERROR_FILE}" \
    -J "${JOB_NAME}" \
    "${SCRIPT}" "${DATASET_NAME}" "${TOKENIZER_NAME}" "${CONFIG_FILE}" "${CORES}"