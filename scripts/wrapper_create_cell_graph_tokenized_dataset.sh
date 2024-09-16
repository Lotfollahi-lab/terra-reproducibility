#!/bin/bash

GROUP="team361"
SCRIPT="reproducibility/scripts/create_tokenized_data.sh"

DATASET_NAME=$1
CORES="${2:-4}" # If not provided, use the default value, i.e. 4 cores
QUEUE="${3:-normal}" # If not provided, use the default value, i.e. normal queue

TOKENIZER_NAME="graph"

case $DATASET_NAME in
    sim1_1105fts_10000locs)
        RAM="5G"
        TIME="0:05"
        ;;
    sim1_1105genes_100klocs_strongincrements)
        RAM="30G"
        TIME="0:20"
        ;;    
    merfish_mouse_brain_8-batch)
        RAM="25G"
        TIME="0:30"
        ;;
    merfish_mouse_brain_16-batch)
        RAM="45G"
        TIME="0:20"
        ;;
    merfish_mouse_brain_64-batch)
        RAM="60G"
        TIME="0:30"
        ;;
    merfish_mouse_brain_239-batch)
        RAM="120G"
        TIME="12:00"
        ;;
    merfish_mouse_brain_100k)
        RAM="30G"
        TIME="0:15"
        ;;
    merfish_mouse_brain_300k)
        RAM="30G"
        TIME="0:30"
        ;;
    *)
        echo "Invalid dataset name"
        exit 1
        ;;
esac

CWD="/lustre/scratch126/cellgen/team361/am84/NicheJEPA/"
mkdir -p "logs/${DATASET_NAME}/profile-nprocs"

CONFIG_FILE="reproducibility/config/cell-${TOKENIZER_NAME}-tokenization_${DATASET_NAME}_read-depth_shifted-log.yaml"
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