#!/bin/bash
#
# Array job wrapper for zero-shot KNN evaluation with nested cross-validation.
#
# Usage:
#   ./wrapper_zeroshot.sh MODEL_NAME DATASET_NAME [NUM_OUTER_FOLDS] [CONFIG_FILE]
#
# Arguments:
#   MODEL_NAME      - Required: see supported models below
#   DATASET_NAME    - Required: see supported datasets below
#   NUM_OUTER_FOLDS - Optional: number of outer folds (default: 4)
#   CONFIG_FILE     - Optional: path to config YAML
#                     (default: config/finetuning/{MODEL_NAME}_{DATASET_NAME}.yaml)
#
# Examples:
#   ./wrapper_zeroshot.sh terra-zeroshot xhs_niche-transfer
#   ./wrapper_zeroshot.sh nicheformer-zeroshot xhs_niche-transfer
#   ./wrapper_zeroshot.sh pantheome-zeroshot xhs_niche-transfer 4
#

GROUP="s10396"
CWD="/lustre/scratch126/cellgen/lotfollahi/am84/NEMO/reproducibility"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: MODEL_NAME and DATASET_NAME are required."
    echo ""
    echo "Usage: ./wrapper_zeroshot.sh MODEL_NAME DATASET_NAME [NUM_OUTER_FOLDS] [CONFIG_FILE]"
    echo ""
    echo "Supported models:"
    echo "  terra-zeroshot, nicheformer-zeroshot, scgpt-spatial-zeroshot,"
    echo "  cellplm-zeroshot, novae-zeroshot"
    echo ""
    echo "Supported datasets:"
    echo "  xhs_niche-transfer, xhs_bk51_niche-transfer,"
    echo "  xhs_disease-classification-patient, xhs_disease-classification-patient-batch"
    exit 1
fi

MODEL_NAME="$1"
DATASET_NAME="$2"

case "${MODEL_NAME}" in
    terra-zeroshot|nicheformer-zeroshot|scgpt-spatial-zeroshot|cellplm-zeroshot|novae-zeroshot)
        ;;
    *)
        echo "Error: Invalid MODEL_NAME '${MODEL_NAME}'"
        exit 1
        ;;
esac

case "${DATASET_NAME}" in
    xhs_bk51_niche-transfer)
        RAM="30G"
        CORES=4
        QUEUE="training-normal"
        ;;
    xhs_niche-transfer|xhs_disease-classification-patient|xhs_disease-classification-patient-batch)
        RAM="60G"
        CORES=4
        QUEUE="training-normal"
        ;;
    *)
        echo "Error: Invalid DATASET_NAME '${DATASET_NAME}'"
        exit 1
        ;;
esac

case "${QUEUE}" in
    training-normal)   USER_GROUP="lotfollahi-training-normal" ;;
    training-parallel) USER_GROUP="lotfollahi-training-parallel" ;;
    *) USER_GROUP="" ;;
esac

NUM_OUTER_FOLDS="${3:-4}"
CONFIG_FILE="${4:-config/finetuning/${MODEL_NAME}_${DATASET_NAME}.yaml}"
SCRIPT="config/finetuning/zeroshot.sh"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/lustre/scratch126/cellgen/lotfollahi/am84/NEMO/data/${DATASET_NAME}/logs/${MODEL_NAME}/${TIMESTAMP}"
JOB_NAME="${MODEL_NAME}"

echo "=============================================="
echo "Zero-shot KNN Array Job Submission"
echo "=============================================="
echo "Model:   ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Config:  ${CONFIG_FILE}"
echo "Log dir: ${LOG_DIR}"
echo "Queue:   ${QUEUE}  RAM: ${RAM}  Cores: ${CORES}  GPU: 1 (exclusive)"
echo "Folds:   1-${NUM_OUTER_FOLDS}"
echo "=============================================="

mkdir -p "${LOG_DIR}"

bsub \
    -G  "${GROUP}" \
    -J  "${JOB_NAME}[1-${NUM_OUTER_FOLDS}]" \
    -n  "${CORES}" \
    -q  "${QUEUE}" \
    ${USER_GROUP:+-U "${USER_GROUP}"} \
    -M  "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}]" \
    -gpu "num=1:mode=exclusive_process:block=yes" \
    -cwd "${CWD}" \
    -o  "${LOG_DIR}/outer_fold_%I_%J.out" \
    -e  "${LOG_DIR}/outer_fold_%I_%J.err" \
    "${SCRIPT}" "${MODEL_NAME}" "${CONFIG_FILE}"

echo ""
