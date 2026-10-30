#!/bin/bash
# Submit the tokenization job array (tokenize_ovca.lsf) for one memory class of the ablation plan. Run from this
# directory in the stjepa_v2 environment.
#   standard: sparse tokenizer path (base, sequence lengths <= 128, cell neighborhood, shifted log, mean/nonzero-mean
#             normalizations of both streams), ~70 GB each
#   large:    dense tokenizer path (rank and count streams normalized differently), Seurat v3 / Pearson residuals and
#             sequence length 256, ~0.5-0.6 TB each for the OVCA section
#
# Usage: bash submit_tokenization.sh standard|large

set -euo pipefail

MEMORY_CLASS="$1"
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
        echo "Unknown memory class: ${MEMORY_CLASS}"
        exit 1
        ;;
esac

ARTIFACT_FOLDER_PATH="${ARTIFACT_FOLDER_PATH:-/nfs/team361/sb75/stjepa-reproducibility/artifacts/ovca_ablation}"
INDICES=$(python tokenize_ovca.py --list_indices "${MEMORY_CLASS}")
echo "Tokenizations (${MEMORY_CLASS}): ${INDICES}"

mkdir -p "${ARTIFACT_FOLDER_PATH}/logs"
bsub -J "ovca_tokenize_${MEMORY_CLASS}[${INDICES}]%${MAX_CONCURRENT}" \
    -q "${QUEUE}" \
    -M "${MEMORY}" \
    -R "select[mem>${MEMORY}] rusage[mem=${MEMORY}] span[hosts=1]" \
    -W "${TIME}" \
    -o "${ARTIFACT_FOLDER_PATH}/logs/tokenize_%J_%I.out" \
    -e "${ARTIFACT_FOLDER_PATH}/logs/tokenize_%J_%I.err" \
    < tokenize_ovca.lsf
