#!/bin/bash
GROUP="team361"
EXPERIMENT_NAME="hst_corpus_70m"
LOG_DIR="logs/${EXPERIMENT_NAME}/farm-job-outputs/analysis"
SCRIPT="analysis/data_preparation/compute_umap.sh"

# Resource specifications
RAM="1000G"
TIME="96:00"
CORES=4
QUEUE="teramem"

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