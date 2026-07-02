#!/bin/bash

MODEL_NAME=$1       # model name
CONFIG_FILE=$2      # path to the config file

# Get outer fold from argument or LSB_JOBINDEX (for array jobs)
if [ -n "$3" ]; then
    OUTER_FOLD=$3
elif [ -n "$LSB_JOBINDEX" ]; then
    OUTER_FOLD=$LSB_JOBINDEX
else
    echo "Error: No outer fold specified. Provide as argument or run as array job."
    exit 1
fi

echo "Running ${MODEL_NAME} evaluation for outer fold ${OUTER_FOLD}..."
echo "Config file: ${CONFIG_FILE}"

source /etc/profile.d/modules.sh

if [ "$USER" == "am84" ]; then
    module load ISG/conda

    case "${MODEL_NAME}" in
        terra-zeroshot|nicheformer-zeroshot|scgpt-spatial-zeroshot|cellplm-zeroshot|novae-zeroshot)
            echo "Loading TERRA environment..."
            conda activate /software/cellgen/team361/am84/envs/nemo
            PYTHON_SCRIPT="analysis/finetune/zeroshot-transfer_outer-fold.py"
            ;;
        *)
            echo "Error: Invalid MODEL_NAME '${MODEL_NAME}'"
            echo "Supported models: terra-zeroshot, nicheformer-zeroshot,"
            echo "                  scgpt-spatial-zeroshot, cellplm-zeroshot, novae-zeroshot"
            exit 1
            ;;
    esac
else
    echo "User $USER not supported"
    exit 1
fi

echo "Using Python script: ${PYTHON_SCRIPT}"
echo ""

python "${PYTHON_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --model "${MODEL_NAME}" \
    --outer-fold "${OUTER_FOLD}"
