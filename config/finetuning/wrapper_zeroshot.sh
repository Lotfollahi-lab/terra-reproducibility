#!/bin/bash
#
# Array job wrapper for Zero-shot KNN evaluation with nested cross-validation
#
# Usage:
#   ./wrapper_zeroshot.sh MODEL_NAME DATASET_NAME [NUM_OUTER_FOLDS] [CONFIG_FILE]
#
# Arguments:
#   MODEL_NAME    - Required: pantheome-zeroshot, novae-zeroshot, or scgpt-spatial-zeroshot
#   DATASET_NAME  - Required: xhs_bk51_niche-transfer or xhs_niche-transfer
#   NUM_OUTER_FOLDS - Optional: Number of outer folds (default: 4)
#   CONFIG_FILE   - Optional: Path to config YAML (default: reproducibility/analysis/finetune/pantheome-zeroshot_xhs_bk51_niche-transfer.yaml)
#
# Example:
#   ./wrapper_zeroshot.sh pantheome-zeroshot xhs_bk51_niche-transfer 4
#   ./wrapper_zeroshot.sh novae-zeroshot xhs_niche-transfer 4 "my_config.yaml"
#

GROUP="team361"

# Validate required arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: MODEL_NAME and DATASET_NAME are required."
    echo ""
    echo "Usage: ./wrapper_zeroshot.sh MODEL_NAME DATASET_NAME [NUM_OUTER_FOLDS] [CONFIG_FILE]"
    echo ""
    echo "Supported models:"
    echo "  - pantheome-zeroshot"
    echo "  - novae-zeroshot"
    echo "  - scgpt-spatial-zeroshot"
    echo ""
    echo "Supported datasets:"
    echo "  - xhs_bk51_niche-transfer"
    echo "  - xhs_niche-transfer"
    exit 1
fi

# Model name (required)
MODEL_NAME="$1"

# Validate model name
case "${MODEL_NAME}" in
    pantheome-zeroshot|novae-zeroshot|scgpt-spatial-zeroshot)
        ;;
    *)
        echo "Error: Invalid MODEL_NAME '${MODEL_NAME}'"
        echo "Supported models: pantheome-zeroshot, novae-zeroshot, scgpt-spatial-zeroshot"
        exit 1
        ;;
esac

# Dataset name (required)
DATASET_NAME="$2"

# Validate dataset name
case "${DATASET_NAME}" in
    xhs_bk51_niche-transfer)
        TIME="3:00"
        RAM="30G"
        NUM_GPUS=1
        CORES=6
        QUEUE="gpu-lotfollahi"
        PTILE=6        
        ;;
    xhs_niche-transfer)
        TIME="12:00"
        RAM="60G"
        NUM_GPUS=1
        CORES=6
        QUEUE="gpu-lotfollahi"
        PTILE=6        
        ;;
    *)
        echo "Error: Invalid DATASET_NAME '${DATASET_NAME}'"
        echo "Supported datasets: xhs_bk51_niche-transfer, xhs_niche-transfer"
        exit 1
        ;;
esac

# Number of outer folds (default: 4)
NUM_OUTER_FOLDS="${3:-4}"

# Config file path (default: zeroshot_config.yaml)
CONFIG_FILE="${4:-reproducibility/config/finetuning/${MODEL_NAME}_${DATASET_NAME}.yaml}"

# Set experiment name based on model
EXPERIMENT_NAME="${MODEL_NAME}"

SCRIPT="reproducibility/config/finetuning/zeroshot.sh"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/lustre/scratch126/cellgen/lotfollahi/am84/NEMO/data/${DATASET_NAME}/logs/${EXPERIMENT_NAME}/${TIMESTAMP}"
JOB_NAME="${EXPERIMENT_NAME}"

echo "=============================================="
echo "Zero-shot KNN Array Job Submission"
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
echo "Time limit: ${TIME}, RAM: ${RAM}"

# Submit array job: -J "name[1-N]" creates N array tasks with LSB_JOBINDEX from 1 to N
# The inner script reads LSB_JOBINDEX directly from the environment
# Pass MODEL_NAME as first argument, CONFIG_FILE as second
bsub \
    -G "${GROUP}" \
    -n "${CORES}" \
    -q "${QUEUE}" \
    -M "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}]" \
    -W "${TIME}" \
    -gpu "num=${NUM_GPUS}:mode=shared:block=yes" \
    -R "span[ptile=${PTILE}]" \
    -o "${OUTPUT_FILE}" \
    -e "${ERROR_FILE}" \
    -J "${JOB_NAME}[1-${NUM_OUTER_FOLDS}]" \
    "${SCRIPT}" "${MODEL_NAME}" "${CONFIG_FILE}"

echo ""
