#!/bin/bash

MODEL_NAME=$1       # model name: pantheome-zeroshot, novae-zeroshot, scgpt-spatial-zeroshot
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

# Modify to match your python environment activation command(s)
source /etc/profile.d/modules.sh

if [ "$USER" == "am84" ]; then
    module load cellgen/conda
    
    # Load appropriate environment based on model
    case "${MODEL_NAME}" in
        pantheome-zeroshot)
            echo "Loading Pantheome/NEMO environment..."
            conda activate /software/cellgen/team361/am84/envs/nemo
            PYTHON_SCRIPT="reproducibility/analysis/finetune/pantheome-zeroshot_single-outer-fold.py"
            ;;
        novae-zeroshot)
            echo "Loading Novae environment..."
            conda activate /software/cellgen/team361/am84/envs/novae
            PYTHON_SCRIPT="reproducibility/analysis/finetune/novae-zeroshot_single-outer-fold.py"
            ;;
        scgpt-spatial-zeroshot)
            echo "Loading scGPT-spatial environment..."
            conda activate /software/cellgen/team361/am84/envs/scgpt-spatial
            PYTHON_SCRIPT="reproducibility/analysis/finetune/scgpt-spatial-zeroshot_single-outer-fold.py"
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
    --outer-fold "${OUTER_FOLD}"
