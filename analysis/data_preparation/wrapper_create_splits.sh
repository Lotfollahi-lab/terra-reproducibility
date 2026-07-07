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
#   disease-classification-patient          (unseen-patient holdout, group by Patient ID)
#   disease-classification-sample           (unseen-sample holdout, group by dataset_id+batch)
#   disease-classification-patient-batch    (DEPRECATED -- superseded by disease-classification-patient)
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
QUEUE="transfer"
USER_GROUP=""

TASK="${1:-}"
MODE_FLAG="--lightweight"

if [[ -z "${TASK}" ]]; then
    echo "Usage: $0 <task> [--full]"
    echo "Tasks: niche-transfer | cell-transfer | disease-classification-patient | disease-classification-sample | disease-classification-patient-batch (deprecated)"
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
    disease-classification-patient)
        CONFIG="${CONFIG_DIR}/create-split_xhs_disease_classification_patient.yaml"
        ;;
    disease-classification-sample)
        CONFIG="${CONFIG_DIR}/create-split_xhs_disease_classification_sample.yaml"
        ;;
    disease-classification-patient-batch)
        echo "Warning: 'disease-classification-patient-batch' is deprecated -- superseded by 'disease-classification-patient' (grouped by Patient ID alone for a true unseen-patient holdout). Proceeding with the deprecated config anyway."
        CONFIG="${CONFIG_DIR}/create-split_xhs_disease_classification_patient_batch.yaml"
        ;;
    *)
        echo "Error: Unknown task '${TASK}'"
        echo "Tasks: niche-transfer | cell-transfer | disease-classification-patient | disease-classification-sample | disease-classification-patient-batch (deprecated)"
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
    ${USER_GROUP:+-U "${USER_GROUP}"} \
    -M  "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}]" \
    -cwd "${CWD}" \
    -o  "${LOG_DIR}/%J.out" \
    -e  "${LOG_DIR}/%J.err" \
    bash -c "module load ISG/conda && \
             conda activate ${CONDA_ENV} && \
             python ${SCRIPT} --config ${CONFIG} ${MODE_FLAG}"

echo ""
