GROUP="team361"

# Set the experiment parameters
EXPERIMENT_NAME="${1:-xhl40-2b_2p}"
LOG_DIR="logs/${EXPERIMENT_NAME}/farm-job-outputs/bronze-to-silver"

SCRIPT="scripts/create_harmonized_data.sh"

case $EXPERIMENT_NAME in
    xhl40-2b_2p)
        RAM="25G"
        TIME="0:05"
        CORES=1
        QUEUE="normal"
        ;;
    xhb1002-76b_1p)
        RAM="80G"
        TIME="2:00"
        CORES=1
        QUEUE="normal"
        ;;
    *)
        echo "Invalid dataset name"
        exit 1
        ;;
esac

# If not provided, use the default value
CORES="${2:-$CORES}"
QUEUE="${3:-$QUEUE}"

# Set the configuration file
CONFIG_FILE="config/bronze_to_silver/${EXPERIMENT_NAME}.yaml"

# Set the output and error files
mkdir -p "${LOG_DIR}"
OUTPUT_FILE="${LOG_DIR}/${CORES}_%J.out"
ERROR_FILE="${LOG_DIR}/${CORES}_%J.err"

JOB_NAME="${EXPERIMENT_NAME}_bronze-to-silver_${CORES}"

echo "Log directory: ${LOG_DIR}"
echo "Harmonizing ${EXPERIMENT_NAME} using ${CORES} cores..."

bsub \
    -G "${GROUP}" \
    -n "${CORES}" \
    -q "${QUEUE}" \
    -M "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}]" \
    -W "${TIME}" \
    -cwd "${CWD}" \
    -o "${OUTPUT_FILE}" \
    -e "${ERROR_FILE}" \
    -J "${JOB_NAME}" \
    "${SCRIPT}" "${CONFIG_FILE}" "${CORES}"