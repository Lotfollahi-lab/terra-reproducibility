GROUP="team361"

# Set the experiment parameters
EXPERIMENT_NAME="${1:-xhl40-2b_2p}"
LOG_DIR="logs/${EXPERIMENT_NAME}/farm-job-outputs/bronze-to-silver"

SCRIPT="scripts/create_harmonized_data.sh"

case $EXPERIMENT_NAME in
    xhb1002-76b_3p)
        RAM="80G"
        TIME="1:00"
        CORES=1
        QUEUE="normal"
        ;;
    xho47-1b_1p)
        RAM="80G"
        TIME="1:00"
        CORES=1
        QUEUE="normal"
        ;;
    *)
        RAM="50G"
        TIME="1:00"
        CORES=1
        QUEUE="normal"
        ;;
esac

# If not provided, use the default value
CORES="${2:-$CORES}"
QUEUE="${3:-$QUEUE}"
TIME="${3:-$TIME}"

# Set the configuration file
CONFIG_FILE="config/bronze_to_silver/${EXPERIMENT_NAME}.yaml"

# Set the output and error files
mkdir -p "${LOG_DIR}"
OUTPUT_FILE="${LOG_DIR}/${CORES}_%J.out"
ERROR_FILE="${LOG_DIR}/${CORES}_%J.err"

JOB_NAME="${EXPERIMENT_NAME}_bronze-to-silver_${CORES}"

echo "Log directory: ${LOG_DIR}"
echo "Harmonizing ${EXPERIMENT_NAME} using ${CORES} cores..."

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
    "${SCRIPT}" "${CONFIG_FILE}" "${CORES}"