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
        QUEUE="normal"
        ;;
    mmb0-1b_smb1-1b_1p_coord_aligned)
        RAM="200G"
        TIME="12:00"
        CORES=12
        QUEUE="normal"
        ;;
    mmb0-1b_smb1-1b_2p)
        RAM="200G"
        TIME="12:00"
        CORES=12
        QUEUE="normal"
        ;;
    mmb0-1b_smb1-1b_2p_coord_aligned)
        RAM="200G"
        TIME="12:00"
        CORES=12
        QUEUE="normal"
        ;;
    xhb1002-AT6-1-1b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhe1008-8-20b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xho47-S1-1b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhb1002-AT9-6b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhp1016-2-18b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhb1002-AT5-1b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhe1008-7-20b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhb1002-AT15-10b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhe1008-6-20b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhb1002-AT6-4-1b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhb1002-AT10-7b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhe1008-9-20b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xho47-S2-1b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1015-R6-28b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1015-R1-28b_1p)    
        RAM="1536G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1015-R3-28b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1015-R4-28b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1015-R5-28b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1015-R7-28b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhk1018-5-23b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhk1018-6-23b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xht43-2b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1020-8-16b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mhp20-2b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1013-4b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mhu21-3b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhb1002-AT16-2-15b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1020-6-16b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1021-1-2b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mhl63-10b_1p)    
        RAM="512G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xho1012-AU16-7b_1p)    
        RAM="512G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1020-3-16b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mhc15-2b_1p)    
        RAM="600G"
        TIME="24:00"
        CORES=12
        QUEUE="long"
        ;;
    xhc38-4b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhp1017-4b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mmb0-P1-239b_1p)    
        RAM="1536G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mmb0-P2-239b_1p)    
        RAM="1536G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mmb0-P3-239b_1p)    
        RAM="1536G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mmb0-P4-239b_1p)    
        RAM="1536G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mmb0-P5-239b_1p)    
        RAM="1536G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    smb1-20b_1p)    
        RAM="1536G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    *)
        #RAM="2200G" # "600G", "1000G", "1500G"
        #TIME="240:00" # "48:00", "120:00"
        #CORES=12
        #QUEUE="teramem" # "long", "hugemem", "teramem"
        RAM="512G" # "600G", "1000G", "1500G"
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