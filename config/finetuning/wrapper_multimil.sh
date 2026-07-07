#!/bin/bash
#
# Array job wrapper for MIL hyperparameter tuning with nested cross-validation.
#
# Usage:
#   ./wrapper_multimil.sh MODEL_NAME DATASET_NAME [NUM_OUTER_FOLDS] [CONFIG_FILE]
#
# Arguments:
#   MODEL_NAME      - Required: source embedding model, e.g. scgpt-spatial-zeroshot,
#                     novae-zeroshot, pantheome-zeroshot, cellplm-zeroshot, terra-cell, terra-nbr.
#                     Results are written to a sibling artifact dir with '-zeroshot'
#                     stripped (if present) and a 'multimil-' prefix added
#                     (e.g. scgpt-spatial-zeroshot -> multimil-scgpt-spatial,
#                     terra-cell -> multimil-terra-cell).
#   DATASET_NAME    - Required: see supported datasets below. The special value
#                     "xhs_disease-classification-abc" runs the A/B/C bagging-granularity
#                     comparison instead of a single config: each array task (one per
#                     outer fold) runs all three configs sequentially via multimil.sh
#                       A: config/finetuning/multimil_{MODEL_NAME}_xhs_disease-classification-patient_bag-sample.yaml
#                       B: config/finetuning/multimil_{MODEL_NAME}_xhs_disease-classification-sample_bag-sample.yaml
#                       C: config/finetuning/multimil_{MODEL_NAME}_xhs_disease-classification-patient_bag-patient.yaml
#                     CONFIG_FILE is ignored in this mode (all three are located by convention).
#   NUM_OUTER_FOLDS - Optional: number of outer folds (default: 4)
#   CONFIG_FILE     - Optional: path to config YAML (single-config mode only)
#                     (default: config/finetuning/multimil_{MODEL_NAME}_{DATASET_NAME}.yaml)
#
# Examples:
#   ./wrapper_multimil.sh scgpt-spatial-zeroshot xhs_disease-classification-patient-batch_5-folds
#   ./wrapper_multimil.sh novae-zeroshot xhs_disease-classification-patient-batch_5-folds 5
#   ./wrapper_multimil.sh terra-nbr xhs_disease-classification-abc
#

GROUP="s10396"
CWD="/lustre/scratch126/cellgen/lotfollahi/am84/NEMO/reproducibility"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: MODEL_NAME and DATASET_NAME are required."
    echo ""
    echo "Usage: ./wrapper_multimil.sh MODEL_NAME DATASET_NAME [NUM_OUTER_FOLDS] [CONFIG_FILE]"
    echo ""
    echo "Supported models (source embeddings):"
    echo "  terra-zeroshot, terra-cell, terra-nbr, nicheformer-zeroshot, scgpt-spatial-zeroshot,"
    echo "  cellplm-zeroshot, novae-zeroshot, pantheome-zeroshot"
    echo ""
    echo "Supported datasets:"
    echo "  xhs_disease-classification-patient, xhs_disease-classification-sample,"
    echo "  xhs_disease-classification-patient-batch, xhs_disease-classification-patient-batch_5-folds,"
    echo "  xhs_disease-classification-abc (A/B/C bagging comparison, all three configs per fold)"
    exit 1
fi

MODEL_NAME="$1"
DATASET_NAME="$2"

case "${MODEL_NAME}" in
    terra-zeroshot|terra-cell|terra-nbr|nicheformer-zeroshot|scgpt-spatial-zeroshot|cellplm-zeroshot|novae-zeroshot|pantheome-zeroshot)
        ;;
    *)
        echo "Error: Invalid MODEL_NAME '${MODEL_NAME}'"
        exit 1
        ;;
esac

case "${DATASET_NAME}" in
    xhs_disease-classification-patient|xhs_disease-classification-sample|xhs_disease-classification-patient-batch|xhs_disease-classification-patient-batch_5-folds)
        RAM="120G"
        CORES=6
        QUEUE="training-normal"
        ;;
    xhs_disease-classification-abc)
        RAM="40G"
        CORES=6
        QUEUE="training-normal"
        ;;
    *)
        echo "Error: Invalid DATASET_NAME '${DATASET_NAME}'"
        exit 1
        ;;
esac

case "${QUEUE}" in
    training-normal)   USER_GROUP="lotfollahi-training-normal" ;;
    training-parallel) USER_GROUP="lotfollahi-training-parallel" ;;
    *) USER_GROUP="" ;;
esac

NUM_OUTER_FOLDS="${3:-4}"

# Output artifact name: source model with '-zeroshot' stripped (if present), 'multimil-'
# prefixed (e.g. scgpt-spatial-zeroshot -> multimil-scgpt-spatial, terra-cell -> multimil-terra-cell),
# matching multimil_outer-fold.py's get_output_model_name()
OUTPUT_MODEL_NAME="multimil-${MODEL_NAME%-zeroshot}"
SCRIPT="config/finetuning/multimil.sh"

if [ "${DATASET_NAME}" == "xhs_disease-classification-abc" ]; then
    CONFIG_A="config/finetuning/multimil_${MODEL_NAME}_xhs_disease-classification-patient_bag-sample.yaml"
    CONFIG_B="config/finetuning/multimil_${MODEL_NAME}_xhs_disease-classification-sample_bag-sample.yaml"
    CONFIG_C="config/finetuning/multimil_${MODEL_NAME}_xhs_disease-classification-patient_bag-patient.yaml"

    for f in "${CONFIG_A}" "${CONFIG_B}" "${CONFIG_C}"; do
        if [ ! -f "${CWD}/${f}" ]; then
            echo "Error: Expected config not found: ${f}"
            exit 1
        fi
    done

    OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME}-abc"
    SCRIPT_ARGS=("${MODEL_NAME}" "${CONFIG_A}" "${CONFIG_B}" "${CONFIG_C}")
    CONFIG_DESC="A: ${CONFIG_A}
Config B: ${CONFIG_B}
Config C: ${CONFIG_C}"
else
    CONFIG_FILE="${4:-config/finetuning/multimil_${MODEL_NAME}_${DATASET_NAME}.yaml}"
    SCRIPT_ARGS=("${MODEL_NAME}" "${CONFIG_FILE}")
    CONFIG_DESC="${CONFIG_FILE}"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/lustre/scratch126/cellgen/lotfollahi/am84/NEMO/data/${DATASET_NAME}/logs/${OUTPUT_MODEL_NAME}/${TIMESTAMP}"
JOB_NAME="${OUTPUT_MODEL_NAME}"

echo "=============================================="
echo "MIL Hyperparameter Tuning Array Job Submission"
echo "=============================================="
echo "Model:   ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Config:  ${CONFIG_DESC}"
echo "Log dir: ${LOG_DIR}"
echo "Queue:   ${QUEUE}  RAM: ${RAM}  Cores: ${CORES}  GPU: 1 (exclusive)"
echo "Folds:   1-${NUM_OUTER_FOLDS}"
echo "=============================================="

mkdir -p "${LOG_DIR}"

bsub \
    -G  "${GROUP}" \
    -J  "${JOB_NAME}[1-${NUM_OUTER_FOLDS}]" \
    -n  "${CORES}" \
    -q  "${QUEUE}" \
    ${USER_GROUP:+-U "${USER_GROUP}"} \
    -M  "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}]" \
    -gpu "num=1:mode=exclusive_process:block=yes" \
    -cwd "${CWD}" \
    -o  "${LOG_DIR}/outer_fold_%I_%J.out" \
    -e  "${LOG_DIR}/outer_fold_%I_%J.err" \
    "${SCRIPT}" "${SCRIPT_ARGS[@]}"

echo ""
