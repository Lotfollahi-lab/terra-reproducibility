#!/bin/bash
#
# Submit a split job for a given task.
#
# Usage:
#   bash analysis/data_preparation/wrapper_create_splits.sh <task> [--full]
#
# Supported tasks:
#   niche-transfer
#   cell-transfer
#   disease-classification-patient+batch
#   disease-classification-sample
#
# Flags:
#   --full   Run full dataset build (h5ad + dataset files per fold).
#            Default is lightweight (cell_ids + labels only).

GROUP="s10396"
CWD="/lustre/scratch126/cellgen/lotfollahi/am84/NEMO/reproducibility"
CONDA_ENV="/software/cellgen/team361/am84/envs/nemo"
CONFIG_DIR="config/create-nested-split"
SCRIPT="analysis/data_preparation/create_nested_stratified_group_splits.py"
RAM="80G"
CORES=4
QUEUE="normal"

TASK="${1:-}"
MODE_FLAG="--lightweight"

if [[ -z "${TASK}" ]]; then
    echo "Usage: $0 <task> [--full]"
    echo "Tasks: niche-transfer | cell-transfer | disease-classification-patient+batch | disease-classification-sample"
    exit 1
fi

shift
for arg in "$@"; do
    case "${arg}" in
        --full) MODE_FLAG="" ;;
        *) echo "Unknown flag: ${arg}"; exit 1 ;;
    esac
done

case "${TASK}" in
    niche-transfer)
        CONFIG="${CONFIG_DIR}/create-split_xhs_niche_transfer.yaml"
        ;;
    cell-transfer)
        CONFIG="${CONFIG_DIR}/create-split_xhs_cell_transfer.yaml"
        ;;
    disease-classification-patient-batch)
        CONFIG="${CONFIG_DIR}/create-split_xhs_disease_classification_patient_batch.yaml"
        ;;
    disease-classification-sample)
        CONFIG="${CONFIG_DIR}/create-split_xhs_disease_classification_sample.yaml"
        ;;
    *)
        echo "Error: Unknown task '${TASK}'"
        echo "Tasks: niche-transfer | cell-transfer | disease-classification-patient+batch | disease-classification-sample"
        exit 1
        ;;
esac

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/lustre/scratch126/cellgen/lotfollahi/am84/NEMO/data/logs/create-splits/${TASK}/${TIMESTAMP}"
JOB_NAME="create-splits_${TASK}"

mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "Create Nested Splits"
echo "=============================================="
echo "Task:   ${TASK}"
echo "Config: ${CONFIG}"
echo "Mode:   ${MODE_FLAG:---lightweight}"
echo "Log dir: ${LOG_DIR}"
echo "Queue:  ${QUEUE}  RAM: ${RAM}  Cores: ${CORES}"
echo "=============================================="

bsub \
    -G  "${GROUP}" \
    -J  "${JOB_NAME}" \
    -n  "${CORES}" \
    -q  "${QUEUE}" \
    -M  "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}]" \
    -cwd "${CWD}" \
    -o  "${LOG_DIR}/%J.out" \
    -e  "${LOG_DIR}/%J.err" \
    bash -c "source /software/cellgen/team361/am84/miniconda3/etc/profile.d/conda.sh && \
             conda activate ${CONDA_ENV} && \
             python ${SCRIPT} --config ${CONFIG} ${MODE_FLAG}"

echo ""
