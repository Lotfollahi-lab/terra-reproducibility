#!/bin/bash

MODEL_NAME=$1       # model name: terra, novae, scgpt_spatial, cellplm, nicheformer
CONFIG_FILE=$2      # path to the config file

echo "Running ${MODEL_NAME} zeroshot embedding computation..."
echo "Config file: ${CONFIG_FILE}"

source /etc/profile.d/modules.sh

if [ "$USER" == "am84" ]; then
    module load ISG/conda

    case "${MODEL_NAME}" in
        terra)
            echo "Loading TERRA/NEMO environment..."
            conda activate /software/cellgen/team361/am84/envs/nemo
            ;;
        novae)
            echo "Loading Novae environment..."
            conda activate /software/cellgen/team361/am84/envs/novae
            ;;
        scgpt_spatial)
            echo "Loading scGPT-spatial environment..."
            source /nfs/team361/sb75/.venvs/scgpt-spatial/bin/activate
            ;;
        cellplm)
            echo "Loading CellPLM environment..."
            conda activate /software/cellgen/team361/am84/envs/cellplm
            ;;
        nicheformer)
            echo "Loading Nicheformer environment..."
            conda activate /software/cellgen/team361/am84/envs/nicheformer
            ;;
        *)
            echo "Error: Invalid MODEL_NAME '${MODEL_NAME}'"
            echo "Supported models: terra, novae, scgpt_spatial, cellplm, nicheformer"
            exit 1
            ;;
    esac
else
    echo "User $USER not supported"
    exit 1
fi

PYTHON_SCRIPT="analysis/inference/compute_zeroshot_embeddings.py"
echo "Using Python script: ${PYTHON_SCRIPT}"
echo ""

python "${PYTHON_SCRIPT}" --config "${CONFIG_FILE}"
