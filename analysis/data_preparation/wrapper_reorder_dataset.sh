#!/bin/bash
GROUP="team361"
EXPERIMENT_NAME="reorder_dataset"
LOG_DIR="logs/${EXPERIMENT_NAME}/farm-job-outputs/analysis"
SCRIPT="analysis/data_preparation/reorder_dataset.sh"

# Resource specifications
RAM="1000G"
TIME="240:00"
CORES=128
QUEUE="hugemem"

# Set the output and error files
mkdir -p "${LOG_DIR}"
OUTPUT_FILE="${LOG_DIR}/${CORES}_%J.out"
ERROR_FILE="${LOG_DIR}/${CORES}_%J.err"

JOB_NAME="${EXPERIMENT_NAME}_${CORES}"

echo "Log directory: ${LOG_DIR}"
echo "Running script ${SCRIPT}..."

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
    "${SCRIPT}"