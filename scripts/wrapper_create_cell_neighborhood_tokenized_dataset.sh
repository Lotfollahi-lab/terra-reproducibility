#!/bin/bash

GROUP="team361"

# Select dataset (default = ss2-1b_1p)
EXPERIMENT_NAME="${1:-sss2-1b_1p}"
# Overwrite token dictionary files (default = False)
OVERWRITE="${2:-False}"

TOKENIZER_NAME="cell-neighborhood-tokenizer"
LOG_DIR="logs/${EXPERIMENT_NAME}/farm-job-outputs/${TOKENIZER_NAME}"

SCRIPT="scripts/create_tokenized_data.sh"

case $EXPERIMENT_NAME in
    mmb0-4b_1p)
        RAM="25G"
        TIME="0:10"
        CORES=4
        QUEUE="normal"
        ;;
    mmb0-100k-1b_1p)
        RAM="25G"
        TIME="0:20"
        CORES=4
        QUEUE="parallel"
        ;;
    mmb0-300k-2b_1p)
        RAM="45G"
        TIME="0:45"
        CORES=4
        QUEUE="parallel"
        ;;
    mmb0-300k-2b_2p)
        RAM="45G"
        TIME="0:45"
        CORES=8
        QUEUE="parallel"
        ;;
    mmb0-239b_1p)
        RAM="120G"
        TIME="18:00"
        CORES=8
        QUEUE="long"
        ;;
    mmb0-1b_smb1-1b_1p)
        RAM="25G"
        TIME="0:20"
        CORES=4
        QUEUE="parallel"
        ;;
    mmb0-1b_smb1-1b_2p)
        RAM="25G"
        TIME="0:20"
        CORES=4
        QUEUE="parallel"
        ;;
    smb1-20b_1p)
        RAM="50G"
        TIME="2:00"
        CORES=4
        QUEUE="parallel"
        ;;    
    sss2-1b_1p)
        RAM="2G"
        TIME="0:10"
        CORES=1
        QUEUE="parallel"
        ;;
    sss3-1b_1p)
        RAM="25G"
        TIME="0:15"
        CORES=8
        QUEUE="parallel"
        ;;
    xhs1000-39b_1p)
        RAM="45G"
        TIME="4:00"
        CORES=8
        QUEUE="normal"
        ;;
    human_cohort1_2m)
        RAM="120G"
        TIME="6:00"
        CORES=16
        QUEUE="normal"
        ;;
    human_cohort1_10m)
        RAM="512G"
        TIME="24:00"
        CORES=16
        QUEUE="long"
        ;;
    *)
        echo "Invalid dataset name"
        exit 1
        ;;
esac

# Set the configuration file
CONFIG_FILE="config/${TOKENIZER_NAME}/${TOKENIZER_NAME}_${EXPERIMENT_NAME}.yaml"

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
    "${SCRIPT}" "${CONFIG_FILE}" "${OVERWRITE}"