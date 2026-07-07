#!/bin/bash
#
# Usage:
#   Single config:  multimil.sh MODEL_NAME CONFIG_FILE [OUTER_FOLD]
#   A/B/C bagging comparison (runs all three configs sequentially for the
#   fold; one config's failure does not block the others):
#                   multimil.sh MODEL_NAME CONFIG_A CONFIG_B CONFIG_C [OUTER_FOLD]
#
# Mode is inferred from argument count: 2-3 args = single config, 4-5 args = A/B/C.
# Outer fold comes from the trailing argument if present, else LSB_JOBINDEX (array jobs).

MODEL_NAME=$1       # source embedding model: scgpt-spatial-zeroshot, novae-zeroshot, pantheome-zeroshot, ...

case $# in
    2|3)
        CONFIGS=("$2")
        TRAILING_ARG=$3
        ;;
    4|5)
        CONFIGS=("$2" "$3" "$4")
        TRAILING_ARG=$5
        ;;
    *)
        echo "Error: Wrong number of arguments."
        echo "Usage: multimil.sh MODEL_NAME CONFIG_FILE [OUTER_FOLD]"
        echo "       multimil.sh MODEL_NAME CONFIG_A CONFIG_B CONFIG_C [OUTER_FOLD]"
        exit 1
        ;;
esac

# Get outer fold from trailing argument or LSB_JOBINDEX (for array jobs)
if [ -n "${TRAILING_ARG}" ]; then
    OUTER_FOLD=${TRAILING_ARG}
elif [ -n "$LSB_JOBINDEX" ]; then
    OUTER_FOLD=$LSB_JOBINDEX
else
    echo "Error: No outer fold specified. Provide as argument or run as array job."
    exit 1
fi

echo "Running multimil evaluation for model ${MODEL_NAME}, outer fold ${OUTER_FOLD}..."
echo "Config file(s): ${CONFIGS[@]}"

source /etc/profile.d/modules.sh

if [ "$USER" == "am84" ]; then
    module load ISG/conda

    case "${MODEL_NAME}" in
        terra-zeroshot|terra-cell|terra-nbr|nicheformer-zeroshot|scgpt-spatial-zeroshot|cellplm-zeroshot|novae-zeroshot|pantheome-zeroshot)
            echo "Loading multimil environment..."
            conda activate /software/cellgen/team361/am84/envs/multimil
            PYTHON_SCRIPT="analysis/finetune/multimil_outer-fold.py"
            ;;
        *)
            echo "Error: Invalid MODEL_NAME '${MODEL_NAME}'"
            echo "Supported models: terra-zeroshot, terra-cell, terra-nbr, nicheformer-zeroshot,"
            echo "                  scgpt-spatial-zeroshot, cellplm-zeroshot, novae-zeroshot, pantheome-zeroshot"
            exit 1
            ;;
    esac
else
    echo "User $USER not supported"
    exit 1
fi

echo "Using Python script: ${PYTHON_SCRIPT}"
echo ""

STATUS=0
for CONFIG_FILE in "${CONFIGS[@]}"; do
    if [ "${#CONFIGS[@]}" -gt 1 ]; then
        echo ""
        echo "============================================================"
        echo "Config: ${CONFIG_FILE}"
        echo "============================================================"
    fi
    python "${PYTHON_SCRIPT}" \
        --config "${CONFIG_FILE}" \
        --model "${MODEL_NAME}" \
        --outer-fold "${OUTER_FOLD}"
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "Config ${CONFIG_FILE} FAILED (exit code ${RC})"
        STATUS=1
    fi
done

exit $STATUS
