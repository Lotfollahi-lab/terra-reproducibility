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
        RAM="1000G"
        TIME="12:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhb1002-AT6-4-1b_1p)
        RAM="1500G"
        TIME="48:00"
        CORES=3
        QUEUE="teramem"
        ;;
    xhb1002-AT9-3b_1p)
        RAM="500G"
        TIME="24:00"
        CORES=6
        QUEUE="long"
        ;;
    xhb1002-AT9-6b_1p)
        RAM="1500G"
        TIME="48:00"
        CORES=3
        QUEUE="teramem"
        ;;
    xhb1002-AT10-7b_1p)
        RAM="1500G"
        TIME="48:00"
        CORES=4
        QUEUE="teramem"
        ;;
    xhb1002-AT5-1b_1p)
        RAM="1500G"
        TIME="48:00"
        CORES=4
        QUEUE="teramem"
        ;;
    xhb1002-AT15-10b_1p)
        RAM="1500G"
        TIME="48:00"
        CORES=4
        QUEUE="teramem"
        ;;
    xho47-S1-1b_1p)
        RAM="1200G"
        TIME="36:00"
        CORES=1
        QUEUE="teramem"
        ;;
    xho47-S2-1b_1p)
        RAM="1200G"
        TIME="48:00"
        CORES=1
        QUEUE="teramem"
        ;;
    mhh22-3b_1p)
        RAM="64G"
        TIME="12:00"
        CORES=4
        QUEUE="normal"
        ;;
    chp60-1b_1p)
        RAM="512G"
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
        RAM="256G"
        TIME="12:00"
        CORES=4
        QUEUE="normal"
        ;;
    xhe1008-5-20b_1p)
        RAM="1500G"
        TIME="12:00"
        CORES=4
        QUEUE="teramem"
        ;;
    xhe1008-2-20b_1p)
        RAM="1500G"
        TIME="12:00"
        CORES=4
        QUEUE="teramem"
        ;;
    xhe1008-3-20b_1p)
        RAM="1500G"
        TIME="12:00"
        CORES=4
        QUEUE="teramem"
        ;;
    xhe1008-4-20b_1p)
        RAM="1500G"
        TIME="12:00"
        CORES=4
        QUEUE="teramem"
        ;;
    xhe1008-6-20b_1p)
        RAM="900G"
        TIME="12:00"
        CORES=4
        QUEUE="teramem"
        ;;
    xhe1008-7-20b_1p)
        RAM="1500G"
        TIME="12:00"
        CORES=4
        QUEUE="teramem"
        ;;
    xhe1008-8-20b_1p)
        RAM="2000G"
        TIME="12:00"
        CORES=4
        QUEUE="teramem"
        ;;
    xhe1008-9-20b_1p)
        RAM="900G"
        TIME="12:00"
        CORES=4
        QUEUE="teramem"
        ;;
    xhe1008-10-20b_1p)
        RAM="900G"
        TIME="12:00"
        CORES=4
        QUEUE="teramem"
        ;;
    mmb0-1b_smb1-1b_2p)
        RAM="256G"
        TIME="12:00"
        CORES=4
        QUEUE="normal"
        ;;
    *)
        RAM="1500G"
        TIME="48:00"
        CORES=4
        QUEUE="teramem"
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