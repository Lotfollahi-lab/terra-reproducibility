#!/bin/bash

GROUP="team361"
SCRIPT="reproducibility/scripts/create_tokenized_data.sh"

DATASET_NAME="${1:-sim1_1105fts_10000locs}"
TOKENIZER_NAME="neighborhood"

case $DATASET_NAME in
    sim1_1105fts_10000locs)
        RAM="2G"
        TIME="0:05"
        CORES=1
        QUEUE="normal"
        ;;
    sim1_1105genes_100klocs_strongincrements)
        RAM="25G"
        TIME="0:10"
        CORES=4
        QUEUE="normal"
        ;;
    merfish_mouse_brain_2-batch)
        RAM="5G"
        TIME="0:05"
        CORES=2
        QUEUE="normal"
        ;;
    merfish_mouse_brain_8-batch)
        RAM="25G"
        TIME="0:20"
        CORES=4
        QUEUE="parallel"
        ;;
    merfish_mouse_brain_16-batch)
        RAM="30G"
        TIME="0:30"
        CORES=4
        QUEUE="parallel"
        ;;
    merfish_mouse_brain_64-batch)
        RAM="60G"
        TIME="1:00"
        CORES=8
        QUEUE="parallel"
        ;;
    merfish_mouse_brain_239-batch)
        RAM="120G"
        TIME="12:00"
        CORES=16
        QUEUE="long"
        ;;
    merfish_mouse_brain_100k)
        RAM="25G"
        TIME="0:20"
        CORES=4
        QUEUE="parallel"
        ;;
    merfish_mouse_brain_300k)
        RAM="45G"
        TIME="0:45"
        CORES=4
        QUEUE="parallel"
        ;;
    merscope_human_liver_cancer_1-batch)
        RAM="25G"
        TIME="0:45"
        CORES=2
        QUEUE="parallel"
        ;;
    *)
        echo "Invalid dataset name"
        exit 1
        ;;
esac

# If not provided, use the default value
CORES="${2:-$CORES}"
QUEUE="${3:-$QUEUE}"

CWD="/lustre/scratch126/cellgen/team361/am84/NicheJEPA/"
mkdir -p "logs/${DATASET_NAME}/profile-nprocs"

CONFIG_FILE="reproducibility/config/cell-${TOKENIZER_NAME}-tokenization_${DATASET_NAME}_read-depth_shifted-log.yaml"
OUTPUT_FILE="logs/${DATASET_NAME}/profile-nprocs/cell_${TOKENIZER_NAME}_tokenization_${CORES}_%J.out"
ERROR_FILE="logs/${DATASET_NAME}/profile-nprocs/cell_${TOKENIZER_NAME}_tokenization_${CORES}_%J.err"
JOB_NAME="${DATASET_NAME}_cell_${TOKENIZER_NAME}_tokenization_${CORES}"

echo "Tokenizing ${DATASET_NAME} with Cell ${TOKENIZER_NAME} Rank using ${CORES} cores..."

bsub \
    -G "${GROUP}" \
    -n "${CORES}" \
    -q "${QUEUE}" \
    -M "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}]" \
    -W "${TIME}" \
    -cwd "${CWD}" \
    -o "${OUTPUT_FILE}" \
    -e "${ERROR_FILE}" \
    -J "${JOB_NAME}" \
    "${SCRIPT}" "${DATASET_NAME}" "${TOKENIZER_NAME}" "${CONFIG_FILE}" "${CORES}"