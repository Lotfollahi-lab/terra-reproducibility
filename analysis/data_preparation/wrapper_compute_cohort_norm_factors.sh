#!/bin/bash
GROUP="team361"
EXPERIMENT_NAME="compute_cohort_norm_factors"
LOG_DIR="logs/data_preparation/${EXPERIMENT_NAME}"
SCRIPT="analysis/data_preparation/compute_cohort_norm_factors.sh"

# Resource specifications
RAM="128G"
TIME="24:00"
CORES=6
QUEUE="long"

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