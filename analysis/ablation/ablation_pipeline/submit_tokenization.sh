#!/bin/bash
# Submit the tokenization job array (tokenize_dataset.lsf) of a dataset for one memory class of the ablation plan
# (`needs_large_memory` in ablation_utils.py). Refuses to submit until the data preparation has written the silver files
# and the normalization factors, and until setup_code.sh has set up the pinned code.
#   standard: 200 GB each. With the plan's tokenizer options (ranked gene lists cut at the input size), this is every
#             tokenization; measured on OVCA (338,695 cells): about 83 GB for the base, dense-path and 256-gene
#             datasets, about 105 GB for Seurat v3 and Pearson residuals.
#   large:    1 TB each, only for tokenizations without the ranked-list cut: the dense tokenizer path (rank and count
#             streams normalized differently), Seurat v3 / Pearson residuals and sequences above 128 genes
#             (0.5-0.6 TB each for OVCA). Nothing to submit with the plan's options.
#
# Usage: bash submit_tokenization.sh ENV_FILE standard|large
# Example: bash submit_tokenization.sh datasets/ovca.env standard

set -euo pipefail

if [ "$#" -ne 2 ] || [ ! -f "$1" ]; then
    echo "Usage: bash submit_tokenization.sh ENV_FILE standard|large  (ENV_FILE: e.g. datasets/ovca.env)" >&2
    exit 1
fi
source "$1"
MEMORY_CLASS="$2"
cd "$(dirname "${BASH_SOURCE[0]}")"  # the jobs run tokenize_dataset.py from here
case "${MEMORY_CLASS}" in
    standard)
        QUEUE="normal"
        MEMORY="200G"
        TIME="12:00"
        MAX_CONCURRENT=6
        ;;
    large)
        QUEUE="hugemem"  # or "teramem"
        MEMORY="1000G"
        TIME="48:00"
        MAX_CONCURRENT=2
        ;;
    *)
        echo "Unknown memory class: ${MEMORY_CLASS} (standard or large)" >&2
        exit 1
        ;;
esac

# The data preparation writes the silver files, then the normalization factors
if [ -z "${SILVER_FILE_PATHS// /}" ]; then
    echo "ERROR: SILVER_FILE_PATHS is empty (no h5ad files in ${DATA_DIRECTORY_PATH}/silver/${DATASET_NAME}?)." >&2
    exit 1
fi
MISSING_FILE_PATHS=()
for FILE_PATH in ${SILVER_FILE_PATHS} "${NORM_FACTOR_FILE_PATH}"; do
    if [ ! -f "${FILE_PATH}" ]; then
        MISSING_FILE_PATHS+=("${FILE_PATH}")
    fi
done
# Written last by the data preparation (prepare_ovca_data.py, compute_norm_factors.py), so existing files of an
# earlier preparation that is being rerun do not pass
if [ ! -f "${NORM_FACTOR_FILE_PATH}.done" ]; then
    MISSING_FILE_PATHS+=("${NORM_FACTOR_FILE_PATH}.done")
fi
if [ "${#MISSING_FILE_PATHS[@]}" -gt 0 ]; then
    echo "ERROR: the data preparation must finish first (OVCA: prepare_ovca_data.lsf, check with bjobs and its log;" \
         "mouse: filter_silver_cells.py, then compute_norm_factors.py, README.md); missing: ${MISSING_FILE_PATHS[*]}" >&2
    exit 1
fi
if [ ! -f "${CODE_PATH}/CODE_VERSION" ]; then
    echo "ERROR: ${CODE_PATH}/CODE_VERSION does not exist; set up the pinned code with setup_code.sh first." >&2
    exit 1
fi

INDICES=$("${ENV_PATH}/bin/python" tokenize_dataset.py --list_indices "${MEMORY_CLASS}" \
    --plan_file_path "${PLAN_FILE_PATH}" --data_directory_path "${DATA_DIRECTORY_PATH}" --dataset_name "${DATASET_NAME}")
if [ -z "${INDICES}" ]; then
    echo "No tokenizations in the ${MEMORY_CLASS} memory class; nothing to submit."
    exit 0
fi
echo "Tokenizations (${MEMORY_CLASS}): ${INDICES}"

# tokenize_dataset.lsf receives these values through bsub -env "all" (the submission environment; it keeps the
# space-separated list of silver files intact)
export PLAN_FILE_PATH DATA_DIRECTORY_PATH DATASET_NAME SILVER_FILE_PATHS NORM_FACTOR_FILE_PATH \
    TOKEN_DICTIONARY_FILE_PATH CODE_PATH ENV_PATH
mkdir -p "${ARTIFACT_FOLDER_PATH}/logs"
bsub -J "${EXPERIMENT_TAG}_tokenize_${MEMORY_CLASS}[${INDICES}]%${MAX_CONCURRENT}" \
    -env "all" \
    -q "${QUEUE}" \
    -M "${MEMORY}" \
    -R "select[mem>${MEMORY}] rusage[mem=${MEMORY}] span[hosts=1]" \
    -W "${TIME}" \
    -o "${ARTIFACT_FOLDER_PATH}/logs/tokenize_%J_%I.out" \
    -e "${ARTIFACT_FOLDER_PATH}/logs/tokenize_%J_%I.err" \
    < tokenize_dataset.lsf
