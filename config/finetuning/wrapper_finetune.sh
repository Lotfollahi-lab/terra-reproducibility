#!/bin/bash
#
# Array job wrapper for Fine-tuning with nested cross-validation
#
# Usage:
#   ./wrapper_finetune.sh MODEL_NAME DATASET_NAME [NUM_OUTER_FOLDS] [CONFIG_FILE]
#
# Arguments:
#   MODEL_NAME      - Required: terra-finetune
#   DATASET_NAME    - Required: xhs_niche-transfer or xhs_cell-transfer
#   NUM_OUTER_FOLDS - Optional: Number of outer folds (default: 4)
#   CONFIG_FILE     - Optional: Path to config YAML (default: reproducibility/config/finetuning/${MODEL_NAME}_${DATASET_NAME}.yaml)
#
# Example:
#   ./wrapper_finetune.sh terra-finetune xhs_niche-transfer 4
#   ./wrapper_finetune.sh terra-finetune xhs_cell-transfer 4 "my_config.yaml"
#

GROUP="s10396"
CWD="/lustre/scratch126/cellgen/lotfollahi/am84/NEMO/reproducibility"

# Validate required arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: MODEL_NAME and DATASET_NAME are required."
    echo ""
    echo "Usage: ./wrapper_finetune.sh MODEL_NAME DATASET_NAME [NUM_OUTER_FOLDS] [CONFIG_FILE]"
    echo ""
    echo "Supported models:"
    echo "  - terra-finetune"
    echo ""
    echo "Supported datasets:"
    echo "  - xhs_niche-transfer"
    echo "  - xhs_cell-transfer"
    exit 1
fi

# Model name (required)
MODEL_NAME="$1"

# Validate model name
case "${MODEL_NAME}" in
    terra-finetune)
        ;;
    *)
        echo "Error: Invalid MODEL_NAME '${MODEL_NAME}'"
        echo "Supported models: terra-finetune"
        exit 1
        ;;
esac

# Dataset name (required)
DATASET_NAME="$2"

# Validate dataset name and set resource requirements
case "${DATASET_NAME}" in
    xhs_niche-transfer)
        RAM="128G"
        NUM_GPUS=1
        CORES=4
        QUEUE="training-normal"
        ;;
    xhs_cell-transfer)
        RAM="128G"
        NUM_GPUS=1
        CORES=4
        QUEUE="training-normal"
        ;;
    *)
        echo "Error: Invalid DATASET_NAME '${DATASET_NAME}'"
        echo "Supported datasets: xhs_niche-transfer, xhs_cell-transfer"
        exit 1
        ;;
esac

# Set user group based on queue
case "${QUEUE}" in
    training-normal)   USER_GROUP="lotfollahi-training-normal" ;;
    training-parallel) USER_GROUP="lotfollahi-training-parallel" ;;
    *) USER_GROUP="" ;;
esac

# Number of outer folds (default: 4)
NUM_OUTER_FOLDS="${3:-4}"

# Config file path (default: finetune config)
CONFIG_FILE="${4:-config/finetuning/${MODEL_NAME}_${DATASET_NAME}.yaml}"

# Set experiment name based on model
EXPERIMENT_NAME="${MODEL_NAME}"

SCRIPT="config/finetuning/finetune.sh"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/lustre/scratch126/cellgen/lotfollahi/am84/NEMO/data/${DATASET_NAME}/logs/${EXPERIMENT_NAME}/${TIMESTAMP}"
JOB_NAME="${EXPERIMENT_NAME}"

echo "=============================================="
echo "Fine-tuning Array Job Submission"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Script: ${SCRIPT}"
echo "Config file: ${CONFIG_FILE}"
echo "Log directory: ${LOG_DIR}"
echo "Job name: ${JOB_NAME}"
echo "Number of outer folds: ${NUM_OUTER_FOLDS}"
echo "=============================================="

# Create log directory
mkdir -p "${LOG_DIR}"

# Set the output and error log files (use %J for job ID, %I for array index)
OUTPUT_FILE="${LOG_DIR}/outer_fold_%I_%J.out"
ERROR_FILE="${LOG_DIR}/outer_fold_%I_%J.err"

echo "Submitting array job with ${NUM_OUTER_FOLDS} tasks (outer folds 1-${NUM_OUTER_FOLDS})..."
echo "Running on ${CORES} cores with ${NUM_GPUS} GPU(s) on queue ${QUEUE}..."
echo "RAM: ${RAM}"

# Submit array job: -J "name[1-N]" creates N array tasks with LSB_JOBINDEX from 1 to N
bsub \
    -G  "${GROUP}" \
    -J  "${JOB_NAME}[1-${NUM_OUTER_FOLDS}]" \
    -n  "${CORES}" \
    -q  "${QUEUE}" \
    ${USER_GROUP:+-U "${USER_GROUP}"} \
    -M  "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}]" \
    -gpu "num=1:mode=exclusive_process:block=yes" \
    -cwd "${CWD}" \
    -o  "${OUTPUT_FILE}" \
    -e  "${ERROR_FILE}" \
    "${SCRIPT}" "${MODEL_NAME}" "${CONFIG_FILE}"

echo ""