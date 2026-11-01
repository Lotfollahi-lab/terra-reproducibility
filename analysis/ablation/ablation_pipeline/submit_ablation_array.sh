#!/bin/bash
# Submit the ablation job array (run_ablation_array.lsf) of a dataset: one element per manifest row, run in manifest
# order with at most MAX_CONCURRENT elements of 4 GPUs at a time (default 2, i.e. 8 GPUs). The array reads the campaign
# from ARTIFACT_FOLDER_PATH (manifest.tsv, configs and dataset.env of generate_ablation_runs.py) and runs in ENV_PATH;
# both come from the env file and are passed through bsub -env.
#
# Usage: bash submit_ablation_array.sh ENV_FILE [INDICES (default: all rows; e.g. 1 for a pilot)] [MAX_CONCURRENT (default: 2)]
# Example: bash submit_ablation_array.sh datasets/ovca.env 1
# Submit one array at a time: each array has its own concurrency limit.
# Retry failed elements (they resume from their latest checkpoint): brequeue -e "<JOBID>[<INDEX>]"
# Change concurrency of a submitted array: bmod -J "%<N>" <JOBID>

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ] || [ ! -f "$1" ]; then
    echo "Usage: bash submit_ablation_array.sh ENV_FILE [INDICES] [MAX_CONCURRENT]  (ENV_FILE: e.g. datasets/ovca.env)" >&2
    exit 1
fi
source "$1"
cd "$(dirname "${BASH_SOURCE[0]}")"  # the jobs run package_run.py and evaluate_ablation_run.py from here

DATASET_ENV_FILE_PATH="${ARTIFACT_FOLDER_PATH}/dataset.env"
MANIFEST_FILE_PATH="${ARTIFACT_FOLDER_PATH}/manifest.tsv"
for FILE_PATH in "${DATASET_ENV_FILE_PATH}" "${MANIFEST_FILE_PATH}"; do
    if [ ! -f "${FILE_PATH}" ]; then
        echo "ERROR: ${FILE_PATH} does not exist; run generate_ablation_runs.py first." >&2
        exit 1
    fi
done
# The elements stop if the pinned code is not the version the configs were generated for; check before queueing
CAMPAIGN_CODE_PATH=$(source "${DATASET_ENV_FILE_PATH}" && echo "${CODE_PATH}")
CAMPAIGN_CODE_VERSION=$(source "${DATASET_ENV_FILE_PATH}" && echo "${CODE_VERSION}")
PINNED_CODE_VERSION=$(cat "${CAMPAIGN_CODE_PATH}/CODE_VERSION" 2>/dev/null || echo "missing")
if [ "${PINNED_CODE_VERSION}" != "${CAMPAIGN_CODE_VERSION}" ]; then
    echo "ERROR: ${CAMPAIGN_CODE_PATH}/CODE_VERSION (${PINNED_CODE_VERSION}) differs from the code version of the" \
         "configs (${CAMPAIGN_CODE_VERSION}); run setup_code.sh or regenerate the configs." >&2
    exit 1
fi
# bsub -env separates variables with commas
for VALUE in "${ARTIFACT_FOLDER_PATH}" "${ENV_PATH}"; do
    if [[ "${VALUE}" =~ [,[:space:]] ]]; then
        echo "ERROR: '${VALUE}' contains a comma or whitespace, which bsub -env cannot pass." >&2
        exit 1
    fi
done
N_RUNS=$(( $(wc -l < "${MANIFEST_FILE_PATH}") - 1 ))
INDICES="${2:-1-${N_RUNS}}"
MAX_CONCURRENT="${3:-2}"

# Two active arrays of one campaign would train the same runs into the same folders
ACTIVE_JOBS=$(bjobs -noheader -o "jobid stat" -J "${EXPERIMENT_TAG}_ablation*" 2>/dev/null \
    | grep -E "PEND|RUN|SUSP" || true)
if [ -n "${ACTIVE_JOBS}" ]; then
    echo "ERROR: an array of this campaign (${EXPERIMENT_TAG}_ablation) is still pending or running:" >&2
    echo "${ACTIVE_JOBS}" >&2
    echo "Wait for it to finish or bkill it before submitting again." >&2
    exit 1
fi

mkdir -p "${ARTIFACT_FOLDER_PATH}/logs"
bsub -J "${EXPERIMENT_TAG}_ablation[${INDICES}]%${MAX_CONCURRENT}" \
    -env "all,ARTIFACT_FOLDER_PATH=${ARTIFACT_FOLDER_PATH},ENV_PATH=${ENV_PATH}" \
    -o "${ARTIFACT_FOLDER_PATH}/logs/train_eval_%J_%I.out" \
    -e "${ARTIFACT_FOLDER_PATH}/logs/train_eval_%J_%I.err" \
    < run_ablation_array.lsf
