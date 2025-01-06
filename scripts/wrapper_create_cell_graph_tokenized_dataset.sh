GROUP="team361"

# Set the experiment parameters
EXPERIMENT_NAME="${1:-sss2-1b_1p}"
TOKENIZER_NAME="cell-graph-tokenizer"
LOG_DIR="logs/${EXPERIMENT_NAME}/farm-job-outputs/${TOKENIZER_NAME}"

SCRIPT="scripts/create_tokenized_data.sh"

case $EXPERIMENT_NAME in
    mmb0-100k-1b_1p)
        RAM="25G"
        TIME="0:20"
        CORES=4
        QUEUE="parallel"
        ;;
    mmb0-300k-2b_1p)
        RAM="45G"
        TIME="0:45"
        CORES=4
        QUEUE="parallel"
        ;;
    mmb0-300k-2b_2p)
        RAM="45G"
        TIME="0:45"
        CORES=8
        QUEUE="parallel"
        ;;
    mmb0-1b_smb1-1b_1p)
        RAM="25G"
        TIME="0:20"
        CORES=4
        QUEUE="parallel"
        ;;
    mmb0-1b_smb1-1b_2p)
        RAM="25G"
        TIME="0:20"
        CORES=4
        QUEUE="parallel"
        ;;
    sss2-1b_1p)
        RAM="2G"
        TIME="0:10"
        CORES=1
        QUEUE="parallel"
        ;;
    sss3-1b_1p)
        RAM="25G"
        TIME="0:30"
        CORES=8
        QUEUE="parallel"
        ;;
    xhs1000-39b_1p)
        RAM="80G"
        TIME="2:00"
        CORES=12
        QUEUE="normal"
        ;;
    xhb1002-76b_1p)
        RAM="650G"
        TIME="48:00"
        CORES=4
        QUEUE="long"
        ;;
    human_cohort1_2m)
        RAM="256G"
        TIME="48:00"
        CORES=12
        QUEUE="long"
        ;;
    human_cohort1_10m)
        RAM="512G"
        TIME="48:00"
        CORES=12
        QUEUE="long"
        ;;
    human_cohort1_50m)
        RAM="200G"
        TIME="2:00"
        CORES=4
        QUEUE="normal"
        ;;
    xhln1004-1b_1p)
        RAM="300G"
        TIME="24:00"
        CORES=1
        QUEUE="long"
        ;;
    chb56-1b_1p)
        RAM="80G"
        TIME="6:00"
        CORES=1
        QUEUE="normal"
        ;;
    chl57-2b_1p)
        RAM="200G"
        TIME="6:00"
        CORES=2
        QUEUE="normal"
        ;;
    chl59-8b_1p)
        RAM="100G"
        TIME="6:00"
        CORES=2
        QUEUE="long"
        ;;
    chc61-9b_1p)
        RAM="100G"
        TIME="6:00"
        CORES=2
        QUEUE="long"
        ;;
    *)
        RAM="200G"
        TIME="12:00"
        CORES=1
        QUEUE="normal"
        ;;
esac

# If not provided, use the default value
CORES="${2:-$CORES}"
QUEUE="${3:-$QUEUE}"
OVERWRITE="${4:-False}"

# Set the configuration file
CONFIG_FILE="config/${TOKENIZER_NAME}/${TOKENIZER_NAME}_${EXPERIMENT_NAME}.yaml"

# Set the output and error files
mkdir -p "${LOG_DIR}"
OUTPUT_FILE="${LOG_DIR}/${TOKENIZER_NAME}_${CORES}_%J.out"
ERROR_FILE="${LOG_DIR}/${TOKENIZER_NAME}_${CORES}_%J.err"

JOB_NAME="${EXPERIMENT_NAME}_${TOKENIZER_NAME}_${CORES}"

echo "Log directory: ${LOG_DIR}"
echo "Running ${TOKENIZER_NAME} on ${EXPERIMENT_NAME} using ${CORES} cores, ${RAM} memory, ${TIME} time limit where token dictionary OVERWRITE = ${OVERWRITE}..."

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
    "${SCRIPT}" "${CONFIG_FILE}" "${OVERWRITE}"