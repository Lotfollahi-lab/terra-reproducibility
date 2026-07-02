#!/bin/bash
#
# Job wrapper for zeroshot embedding computation
#
# Usage:
#   ./wrapper_compute_embeddings.sh MODEL_NAME [CONFIG_FILE]
#
# Arguments:
#   MODEL_NAME  - Required: terra, novae, scgpt_spatial, cellplm, nicheformer
#   CONFIG_FILE - Optional: path to config YAML
#                 (default: config/zeroshot-embeddings/{MODEL_NAME}_zeroshot_embeddings.yaml)
#
# Example:
#   ./wrapper_compute_zeroshot_embeddings.sh terra
#   ./wrapper_compute_zeroshot_embeddings.sh novae
#   ./wrapper_compute_zeroshot_embeddings.sh scgpt_spatial config/zeroshot-embeddings/scgpt_spatial_zeroshot_embeddings.yaml
#

GROUP="s10396"
CWD="/lustre/scratch126/cellgen/lotfollahi/am84/NEMO/reproducibility"

if [ -z "$1" ]; then
    echo "Error: MODEL_NAME is required."
    echo ""
    echo "Usage: ./wrapper_compute_zeroshot_embeddings.sh MODEL_NAME [CONFIG_FILE]"
    echo ""
    echo "Supported models: terra, novae, scgpt_spatial, cellplm, nicheformer"
    exit 1
fi

MODEL_NAME="$1"

case "${MODEL_NAME}" in
    terra)
        RAM="80G"
        NUM_GPUS=1
        CORES=4
        QUEUE="training-normal"
        ;;
    novae)
        RAM="80G"
        NUM_GPUS=1
        CORES=4
        QUEUE="training-normal"
        ;;
    scgpt_spatial)
        RAM="80G"
        NUM_GPUS=1
        CORES=4
        QUEUE="training-normal"
        ;;
    cellplm)
        RAM="120G"
        NUM_GPUS=1
        CORES=4
        QUEUE="training-normal"
        ;;
    nicheformer)
        RAM="120G"
        NUM_GPUS=1
        CORES=4
        QUEUE="training-normal"
        ;;
    *)
        echo "Error: Invalid MODEL_NAME '${MODEL_NAME}'"
        echo "Supported models: terra, novae, scgpt_spatial, cellplm, nicheformer"
        exit 1
        ;;
esac

# Set user group based on queue
case "${QUEUE}" in
    training-normal)   USER_GROUP="lotfollahi-training-normal" ;;
    training-parallel) USER_GROUP="lotfollahi-training-parallel" ;;
    *) USER_GROUP="" ;;
esac

CONFIG_FILE="${2:-config/zeroshot-embeddings/${MODEL_NAME}_zeroshot_embeddings.yaml}"

SCRIPT="analysis/inference/compute_zeroshot_embeddings.sh"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/lustre/scratch126/cellgen/lotfollahi/am84/NEMO/data/logs/zeroshot-embeddings/${MODEL_NAME}/${TIMESTAMP}"
JOB_NAME="zeroshot-emb_${MODEL_NAME}"

echo "=============================================="
echo "Zeroshot Embedding Computation"
echo "=============================================="
echo "Model:   ${MODEL_NAME}"
echo "Script:  ${SCRIPT}"
echo "Config:  ${CONFIG_FILE}"
echo "Log dir: ${LOG_DIR}"
echo "Queue:   ${QUEUE}  RAM: ${RAM}  Cores: ${CORES}  GPUs: ${NUM_GPUS}"
echo "=============================================="

mkdir -p "${LOG_DIR}"

OUTPUT_FILE="${LOG_DIR}/%J.out"
ERROR_FILE="${LOG_DIR}/%J.err"

bsub \
    -G  "${GROUP}" \
    -J  "${JOB_NAME}" \
    -n  "${CORES}" \
    -q  "${QUEUE}" \
    ${USER_GROUP:+-U "${USER_GROUP}"} \
    -M  "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}]" \
    -gpu "num=${NUM_GPUS}:mode=exclusive_process:block=yes" \
    -cwd "${CWD}" \
    -o  "${OUTPUT_FILE}" \
    -e  "${ERROR_FILE}" \
    "${SCRIPT}" "${MODEL_NAME}" "${CONFIG_FILE}"

echo ""
