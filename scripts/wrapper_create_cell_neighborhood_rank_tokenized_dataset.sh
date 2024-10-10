#!/bin/bash

GROUP="team361"

# Set the experiment parameters
EXPERIMENT_NAME="${1:-sss2-1b_1p}"
TOKENIZER_NAME="cell-neighborhood-rank-tokenizer"
LOG_DIR="logs/${EXPERIMENT_NAME}/farm-job-outputs/${TOKENIZER_NAME}"

SCRIPT="scripts/create_tokenized_data.sh"

case $EXPERIMENT_NAME in
    sss2-1b_1p)
        RAM="2G"
        TIME="0:10"
        CORES=1
        QUEUE="parallel"
        ;;
    sss3-1b_1p)
        RAM="25G"
        TIME="0:10"
        CORES=4
        QUEUE="parallel"
        ;;
    mmb0-8b_1p)
        RAM="25G"
        TIME="0:20"
        CORES=4
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
    merfish_mouse_brain_300k_2-gene-panels)
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
    mmb0-1b_smb1-1b_1p)
        RAM="25G"
        TIME="0:10"
        CORES=4
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

# Set the configuration file
CONFIG_FILE="config/${TOKENIZER_NAME}/${TOKENIZER_NAME}_${EXPERIMENT_NAME}_read-depth_shifted-log.yaml"

# Set the output and error files
mkdir -p "${LOG_DIR}"
OUTPUT_FILE="${LOG_DIR}/${TOKENIZER_NAME}_${CORES}_%J.out"
ERROR_FILE="${LOG_DIR}/${TOKENIZER_NAME}_${CORES}_%J.err"

JOB_NAME="${EXPERIMENT_NAME}_${TOKENIZER_NAME}_${CORES}"

echo "Log directory: ${LOG_DIR}"
echo "Running ${TOKENIZER_NAME} on ${EXPERIMENT_NAME} using ${CORES} cores..."

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
    "${SCRIPT}" "${EXPERIMENT_NAME}" "${TOKENIZER_NAME}" "${CONFIG_FILE}" "${CORES}"