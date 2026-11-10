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
# With EVAL_GENE_SET=shared_genes (env file), a second array tokenizes the shared-gene evaluation data of the dataset
# (filter_shared_genes.py; the same cells with the genes of every section) with the same tokenizations, token dictionary
# and factor file (job name <tag>_tokenize_shared_genes_<class>). It starts once the first array has ended, so that at
# most MAX_CONCURRENT tokenizations run at a time.
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
# The training data, and the shared-gene evaluation data if the evaluation uses it (written by filter_shared_genes.py
# under the silver file names)
case "${EVAL_GENE_SET:-}" in
    all_genes)
        GENE_SETS=(all_genes)
        ;;
    shared_genes)
        GENE_SETS=(all_genes shared_genes)
        SHARED_GENE_FOLDER_PATH="${DATA_DIRECTORY_PATH}/silver/${DATASET_NAME}_shared_genes"
        for FILE_PATH in ${SILVER_FILE_PATHS}; do
            SHARED_GENE_FILE_PATH="${SHARED_GENE_FOLDER_PATH}/$(basename "${FILE_PATH}")"
            if [ ! -f "${SHARED_GENE_FILE_PATH}" ]; then
                echo "ERROR: ${SHARED_GENE_FILE_PATH} does not exist; run filter_shared_genes.py first (README.md)." >&2
                exit 1
            fi
        done
        # Written last by filter_shared_genes.py (tokenize_dataset.py checks that it matches the silver files)
        if [ ! -f "${SHARED_GENE_FOLDER_PATH}/ablation_provenance.json" ]; then
            echo "ERROR: ${SHARED_GENE_FOLDER_PATH}/ablation_provenance.json does not exist; filter_shared_genes.py did" \
                 "not finish. Rerun it (README.md)." >&2
            exit 1
        fi
        ;;
    *)
        echo "ERROR: EVAL_GENE_SET must be all_genes or shared_genes in $1, not '${EVAL_GENE_SET:-}'." >&2
        exit 1
        ;;
esac

INDICES=$("${ENV_PATH}/bin/python" tokenize_dataset.py --list_indices "${MEMORY_CLASS}" \
    --plan_file_path "${PLAN_FILE_PATH}" --data_directory_path "${DATA_DIRECTORY_PATH}" --dataset_name "${DATASET_NAME}")
if [ -z "${INDICES}" ]; then
    echo "No tokenizations in the ${MEMORY_CLASS} memory class; nothing to submit."
    exit 0
fi
echo "Tokenizations (${MEMORY_CLASS}): ${INDICES}; gene sets: ${GENE_SETS[*]}"

# tokenize_dataset.lsf receives these values through bsub -env "all" (the submission environment; it keeps the
# space-separated list of silver files intact)
export PLAN_FILE_PATH DATA_DIRECTORY_PATH DATASET_NAME SILVER_FILE_PATHS NORM_FACTOR_FILE_PATH \
    TOKEN_DICTIONARY_FILE_PATH CODE_PATH ENV_PATH GENE_SET
mkdir -p "${ARTIFACT_FOLDER_PATH}/logs"
TRAINING_JOB_NAME="${EXPERIMENT_TAG}_tokenize_${MEMORY_CLASS}"
for GENE_SET in "${GENE_SETS[@]}"; do
    if [ "${GENE_SET}" = "all_genes" ]; then
        JOB_NAME="${TRAINING_JOB_NAME}"
        DEPENDENCY=()
    else
        # After the training data array (also if some of its elements failed)
        JOB_NAME="${EXPERIMENT_TAG}_tokenize_${GENE_SET}_${MEMORY_CLASS}"
        DEPENDENCY=(-w "ended(${TRAINING_JOB_NAME})")
    fi
    bsub -J "${JOB_NAME}[${INDICES}]%${MAX_CONCURRENT}" \
        ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} \
        -env "all" \
        -q "${QUEUE}" \
        -M "${MEMORY}" \
        -R "select[mem>${MEMORY}] rusage[mem=${MEMORY}] span[hosts=1]" \
        -W "${TIME}" \
        -o "${ARTIFACT_FOLDER_PATH}/logs/tokenize_%J_%I.out" \
        -e "${ARTIFACT_FOLDER_PATH}/logs/tokenize_%J_%I.err" \
        < tokenize_dataset.lsf
done
