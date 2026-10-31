#!/bin/bash
# Submit the OVCA ablation job array (run_ablation_array.lsf): one element per manifest row, run in manifest order with
# at most MAX_CONCURRENT elements of 4 GPUs at a time (default 2, i.e. 8 GPUs). Run from this directory.
#
# Usage: bash submit_ablation_array.sh [INDICES (default: all rows; e.g. 1 for a pilot)] [MAX_CONCURRENT (default: 2)]
# Submit one array at a time: each array has its own concurrency limit.
# Retry failed elements (they resume from their latest checkpoint): brequeue -e "<JOBID>[<INDEX>]"
# Change concurrency of a submitted array: bmod -J "%<N>" <JOBID>

set -euo pipefail

export ARTIFACT_FOLDER_PATH="${ARTIFACT_FOLDER_PATH:-/nfs/team361/sb75/terra-reproducibility/artifacts/ovca_ablation}"
N_RUNS=$(( $(wc -l < "${ARTIFACT_FOLDER_PATH}/manifest.tsv") - 1 ))
INDICES="${1:-1-${N_RUNS}}"
MAX_CONCURRENT="${2:-2}"

mkdir -p "${ARTIFACT_FOLDER_PATH}/logs"
bsub -J "stjepa_v2_ovca_ablation[${INDICES}]%${MAX_CONCURRENT}" \
    -o "${ARTIFACT_FOLDER_PATH}/logs/train_eval_%J_%I.out" \
    -e "${ARTIFACT_FOLDER_PATH}/logs/train_eval_%J_%I.err" \
    < run_ablation_array.lsf
