GROUP="team361"

# Set the experiment parameters
EXPERIMENT_NAME="${1:-sss2-1b_1p}"
TOKENIZER_NAME="cell-graph-tokenizer"
LOG_DIR="logs/${EXPERIMENT_NAME}/farm-job-outputs/${TOKENIZER_NAME}"

SCRIPT="analysis/tokenization/create_tokenized_dataset.sh"

case $EXPERIMENT_NAME in
    sss2-1b_1p)
        RAM="2G"
        TIME="0:10"
        CORES=1
        QUEUE="normal"
        ;;
    sss3-1b_1p)
        RAM="25G"
        TIME="0:30"
        CORES=8
        QUEUE="normal"
        ;;
    xhs1000-39b_1p)
        RAM="128G"
        TIME="12:00"
        CORES=12
        QUEUE="normal"
        ;;
    chl59-8b_1p)
        RAM="300G"
        TIME="12:00"
        CORES=2
        QUEUE="normal"
        ;;
    xhb1002-AT5-1b_1p)
        RAM="400G"
        TIME="36:00"
        CORES=4
        QUEUE="long"
        ;;
    xhb1002-AT6-1-1b_1p)
        RAM="350G"
        TIME="48:00"
        CORES=3
        QUEUE="long"
        ;;
    xhb1002-AT6-4-1b_1p)
        RAM="600G"
        TIME="48:00"
        CORES=3
        QUEUE="long"
        ;;
    xhb1002-AT9-3b_1p)
        RAM="500G"
        TIME="24:00"
        CORES=6
        QUEUE="long"
        ;;
    xhb1002-AT9-6b_1p)
        RAM="500G"
        TIME="48:00"
        CORES=3
        QUEUE="long"
        ;;
    xhb1002-AT10-7b_1p)
        RAM="600G"
        TIME="48:00"
        CORES=4
        QUEUE="long"
        ;;
    xhb1002-AT15-10b_1p)
        RAM="500G"
        TIME="48:00"
        CORES=4
        QUEUE="long"
        ;;
    xhb1002-AT23-9b_1p)
        RAM="300G"
        TIME="48:00"
        CORES=4
        QUEUE="long"
        ;;
    xho47-S1-1b_1p)
        RAM="512G"
        TIME="36:00"
        CORES=1
        QUEUE="long"
        ;;
    xho47-S2-1b_1p)
        RAM="600G"
        TIME="48:00"
        CORES=1
        QUEUE="long"
        ;;
    mhh22-3b_1p)
        RAM="64G"
        TIME="12:00"
        CORES=4
        QUEUE="normal"
        ;;
    chp60-1b_1p)
        RAM="64G"
        TIME="12:00"
        CORES=4
        QUEUE="normal"
        ;;
    shp75-4b_1p)
        RAM="64G"
        TIME="12:00"
        CORES=4
        QUEUE="normal"
        ;;
    xhsc1006-2b_1p)
        RAM="64G"
        TIME="12:00"
        CORES=4
        QUEUE="normal"
        ;;
    xhc1001-1b_1p)
        RAM="64G"
        TIME="12:00"
        CORES=4
        QUEUE="normal"
        ;;
    xhe1008-6-20b_1p)
        RAM="512G"
        TIME="12:00"
        CORES=4
        QUEUE="long"
        ;;
    xhe1008-7-20b_1p)
        RAM="512G"
        TIME="12:00"
        CORES=4
        QUEUE="long"
        ;;
    xhe1008-8-20b_1p)
        RAM="512G"
        TIME="12:00"
        CORES=4
        QUEUE="long"
        ;;
    xhe1008-9-20b_1p)
        RAM="512G"
        TIME="12:00"
        CORES=4
        QUEUE="long"
        ;;
    xhe1008-10-20b_1p)
        RAM="512G"
        TIME="12:00"
        CORES=4
        QUEUE="long"
        ;;
    *)
        RAM="256G"
        TIME="48:00"
        CORES=4
        QUEUE="long"
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

# Check if the queue name contains 'gpu'
if [[ "$QUEUE" == *gpu* ]]; then
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