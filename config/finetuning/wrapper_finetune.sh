GROUP="team361"

# Set the experiment parameters
EXPERIMENT_NAME="niche-classification"

DATASET_NAME="${1:-xhs}"
MODEL_NAME="${2:-finetune}" # finetune or zeroshot
HEAD="${3:-linear}" # linear or mlp

SCRIPT="reproducibility/config/finetuning/finetune.sh"
LOG_DIR="/lustre/scratch126/cellgen/lotfollahi/am84/NEMO/data/${DATASET_NAME}_${EXPERIMENT_NAME}/logs"
CONFIG_FILE="reproducibility/config/finetuning/${DATASET_NAME}_${EXPERIMENT_NAME}_${MODEL_NAME}-${HEAD}.yaml"
JOB_NAME="${EXPERIMENT_NAME}_${DATASET_NAME}_${MODEL_NAME}_${HEAD}"

echo "Script: ${SCRIPT}"
echo "Log directory: ${LOG_DIR}"
echo "Config file: ${CONFIG_FILE}"
echo "Job name: ${JOB_NAME}"

case $DATASET_NAME in
    xhs)
        RAM="128G"
        TIME="24:00"
        NUM_GPUS=1
        CORES=6
        QUEUE="gpu-lotfollahi"
        PTILE=6
        ;;
    *)
        echo "Invalid dataset name"
        exit 1
        ;;
esac

# If not provided, use the default value
CORES="${4:-$CORES}"
QUEUE="${5:-$QUEUE}"

# Set the output and error log files
mkdir -p "${LOG_DIR}"
OUTPUT_FILE="${LOG_DIR}/${CORES}_${QUEUE}_%J.out"
ERROR_FILE="${LOG_DIR}/${CORES}_${QUEUE}_%J.err"

echo "Running ${MODEL_NAME} with ${HEAD} head on ${DATASET_NAME} using ${CORES} cores on ${NUM_GPUS} GPUs..."

bsub \
    -G "${GROUP}" \
    -n "${CORES}" \
    -q "${QUEUE}" \
    -M "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}]" \
    -W "${TIME}" \
    -cwd "${CWD}" \
    -gpu "num=${NUM_GPUS}:mode=exclusive_process:block=yes" \
    -R "span[ptile=${PTILE}]" \
    -o "${OUTPUT_FILE}" \
    -e "${ERROR_FILE}" \
    -J "${JOB_NAME}" \
    "${SCRIPT}" "${CONFIG_FILE}"