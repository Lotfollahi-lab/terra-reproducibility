#!/bin/bash

GROUP="team361"

# Set the experiment parameters
EXPERIMENT_NAME="${1:-sss2-1b_1p}"
GT_TYPE="${2:-cell_graph}"
DEVICES="${3:-cuda:0}"
LOG_DIR="logs/${EXPERIMENT_NAME}/farm-job-outputs/model"

SCRIPT="analysis/model_training/train_model.sh"

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
        RAM="256G"
        TIME="72:00"
        CORES=24
        QUEUE="gpu-lotfollahi"
        N_GPU=1
        ;;
    human_cohort1_2m)
        RAM="256G"
        TIME="48:00"
        CORES=12
        QUEUE="gpu-lotfollahi"
        N_GPU=1
        ;;
    human_cohort1_10m)
        RAM="256G"
        TIME="48:00"
        CORES=12
        QUEUE="gpu-basement"
        N_GPU=1
        ;;
    human_cohort2_60m)
        RAM="256G"
        TIME="480:00"
        CORES=24
        QUEUE="gpu-lotfollahi"
        N_GPU=1
        ;;
    hst_corpus_70m)
        RAM="256G"
        TIME="240:00"
        CORES=12
        QUEUE="gpu-basement"
        N_GPU=2
        ;;
    *)
        echo "Invalid dataset name"
        exit 1
        ;;
esac

# If not provided, use the default value
CORES="${4:-$CORES}"
QUEUE="${5:-$QUEUE}"

# Set the configuration file
CONFIG_FILE="config/model/${EXPERIMENT_NAME}_${GT_TYPE}.yaml"

# Set the output and error files
mkdir -p "${LOG_DIR}"
OUTPUT_FILE="${LOG_DIR}/${CORES}_%J.out"
ERROR_FILE="${LOG_DIR}/${CORES}_%J.err"

JOB_NAME="${EXPERIMENT_NAME}_${GT_TYPE}_${CORES}"

echo "Log directory: ${LOG_DIR}"
echo "Training ${GT_TYPE} model on ${EXPERIMENT_NAME} dataset using ${CORES} cores and ${N_GPU} GPUs..."
echo "Running script ${SCRIPT} with config file ${CONFIG_FILE} and devices ${DEVICES}..."

bsub \
    -G "${GROUP}" \
    -n "${CORES}" \
    -q "${QUEUE}" \
    -gpu "mode=exclusive_process:num=${N_GPU}:block=yes" \
    -M "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}]" \
    -W "${TIME}" \
    -cwd "${CWD}" \
    -o "${OUTPUT_FILE}" \
    -e "${ERROR_FILE}" \
    -J "${JOB_NAME}" \
    "${SCRIPT}" "${CONFIG_FILE}" "${DEVICES}"
