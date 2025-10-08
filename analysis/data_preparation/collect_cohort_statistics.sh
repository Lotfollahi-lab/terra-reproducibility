#!/bin/bash

GROUP="team361"

# Set the experiment parameters
EXPERIMENT_NAME="${1:-human_cohort1_50m}" # Options are: human_cohort1_2m, human_cohort1_10m, human_cohort1_50m, or any dataset_name
LOG_DIR="logs/${EXPERIMENT_NAME}/farm-job-outputs/collect_cohort_statistics"

case $EXPERIMENT_NAME in
    human_cohort1_50m)
        RAM="60G"
        TIME="0:30"
        CORES=1
        QUEUE="normal"
        ;;
    *)
        RAM="25G"
        TIME="0:10"
        CORES=1
        QUEUE="normal"
        ;;
esac

# Set the output and error files
mkdir -p "${LOG_DIR}"
OUTPUT_FILE="${LOG_DIR}/${CORES}_%J.out"
ERROR_FILE="${LOG_DIR}/${CORES}_%J.err"

JOB_NAME="${EXPERIMENT_NAME}_collect_cohort_statistics_${CORES}"

echo "Log directory: ${LOG_DIR}"

# Add settings for different users
if [ "$USER" == "am84" ]; then
    source /etc/profile.d/modules.sh
    module load cellgen/conda
    conda activate nichejepa-reproducibility
elif [ "$USER" == "sb75" ]; then
    source ~/.venvs/nichejepa/bin/activate
fi

SCRIPT="python analysis/data_preparation/collect_cohort_statistics.py --cohort_name ${EXPERIMENT_NAME}"

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
    "${SCRIPT}"