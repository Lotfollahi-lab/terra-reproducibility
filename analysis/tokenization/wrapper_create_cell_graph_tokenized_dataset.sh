GROUP="team361"

# Set the experiment parameters
EXPERIMENT_NAME="${1:-sss2-1b_1p}"
TOKENIZER_NAME="cell-graph-tokenizer"
LOG_DIR="logs/${EXPERIMENT_NAME}/farm-job-outputs/${TOKENIZER_NAME}"

SCRIPT="analysis/tokenization/create_tokenized_dataset.sh"

case $EXPERIMENT_NAME in
    mmb0-1b_smb1-1b_1p)
        RAM="200G"
        TIME="12:00"
        CORES=12
        QUEUE="inference"
        ;;
    xhb1002-AT6-1-1b_1p)
        RAM="512G"
        TIME="24:00"
        CORES=12
        QUEUE="long"
        ;;
    xhe1008-8-20b_1p)
        RAM="512G"
        TIME="24:00"
        CORES=12
        QUEUE="long"
        ;;
    xho47-S1-1b_1p)
        RAM="512G"
        TIME="24:00"
        CORES=12
        QUEUE="long"
        ;;
    xhb1002-AT9-6b_1p)
        RAM="512G"
        TIME="24:00"
        CORES=12
        QUEUE="long"
        ;;
    xhp1016-2-18b_1p)
        RAM="512G"
        TIME="24:00"
        CORES=12
        QUEUE="long"
        ;;
    xhb1002-AT5-1b_1p)    
        RAM="512G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhe1008-7-20b_1p)
        RAM="512G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhb1002-AT15-10b_1p)
        RAM="512G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhe1008-6-20b_1p)
        RAM="512G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    *)
        #RAM="2200G" # "600G", "1000G", "1500G"
        #TIME="240:00" # "48:00", "120:00"
        #CORES=12
        #QUEUE="teramem" # "long", "hugemem", "teramem"
        RAM="256G" # "600G", "1000G", "1500G"
        TIME="24:00" # "48:00", "120:00"
        CORES=12
        QUEUE="long" # "long", "hugemem", "teramem", "basement", "parallel", "week"
        ;;
esac

# If not provided, use the default value
CORES="${2:-$CORES}"
QUEUE="${3:-$QUEUE}"

# Set the configuration file
CONFIG_FILE="config/${TOKENIZER_NAME}/${TOKENIZER_NAME}_${EXPERIMENT_NAME}.yaml"

# Set the output and error files
mkdir -p "${LOG_DIR}"
OUTPUT_FILE="${LOG_DIR}/${TOKENIZER_NAME}_${CORES}_%J.out"
ERROR_FILE="${LOG_DIR}/${TOKENIZER_NAME}_${CORES}_%J.err"

JOB_NAME="${EXPERIMENT_NAME}_${TOKENIZER_NAME}_${CORES}"

echo "Log directory: ${LOG_DIR}"
echo "Running ${TOKENIZER_NAME} on ${EXPERIMENT_NAME} using ${CORES} cores..."

# Initialize GPU parameter
GPU_PARAM=""

# Check if the queue is a gpu queue
if [[ "$QUEUE" == *gpu* ]]; then
    GPU_PARAM="-gpu num=1"
    echo "Queue requires GPUs. Requesting 1 GPU(s)."
fi
if [[ "$QUEUE" == inference ]]; then
    GPU_PARAM="-gpu num=1"
    echo "Queue requires GPUs. Requesting 1 GPU(s)."
fi

echo $GPU_PARAM

bsub \
    -G "${GROUP}" \
    -n "${CORES}" \
    -q "${QUEUE}" \
    -M "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}]" \
    ${GPU_PARAM} \
    -W "${TIME}" \
    -cwd "${CWD}" \
    -o "${OUTPUT_FILE}" \
    -e "${ERROR_FILE}" \
    -J "${JOB_NAME}" \
    "${SCRIPT}" "${CONFIG_FILE}"