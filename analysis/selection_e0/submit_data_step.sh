#!/bin/bash
# Submit one data step of E0 as an LSF job on the CPU farm (analysis/selection_e0/README.md):
#   select    select_proxy.py: a dry run (strata, batches and crops in the .out file); with RUN=1 it writes the
#             proxy's silver folder and its manifests (<ARTIFACT_FOLDER_PATH>/proxy)
#   tokenize  the proxy's tokenization (config/cell-graph-tokenizer/cell-graph-tokenizer_<PROXY_NAME>.yaml: the corpus
#             settings plus add_spatial_index) with the pinned TERRA code, under run_tokenization_locked.sh's lock
#   lists     build_index_lists.py: the index lists of the E0 arms (<ARTIFACT_FOLDER_PATH>/index_lists)
# Usage (from anywhere): [RUN=1] bash analysis/selection_e0/submit_data_step.sh analysis/selection_e0/e0.env STEP
# The job runs in this checkout. Logs: logs/selection_e0/<step>_<jobid>.{out,err}; the tokenization's in
# logs/<PROXY_NAME>/farm-job-outputs/cell-graph-tokenizer/, as those of the corpus parts.
set -euo pipefail
USAGE="Usage: [RUN=1] bash submit_data_step.sh ENV_FILE select|tokenize|lists"
ENV_FILE=$(cd "$(dirname "${1:?${USAGE}}")" && pwd)/$(basename "$1")
STEP=${2:?${USAGE}}
source "${ENV_FILE}"
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"
if [ "$(cat "${CODE_PATH}/CODE_VERSION" 2>/dev/null)" != "terra ${COMMIT}" ]; then
    echo "ERROR: ${CODE_PATH} does not hold terra ${COMMIT}: run setup_code.sh first." >&2
    exit 1
fi
LOG_DIR="logs/selection_e0"
mkdir -p "${LOG_DIR}"
# bsub -env separates variables with commas
if [[ "${ENV_FILE}" =~ [,[:space:]] ]]; then
    echo "ERROR: '${ENV_FILE}' contains a comma or whitespace, which bsub -env cannot pass." >&2
    exit 1
fi
COMMON=(-G "${LSF_GROUP_CPU}" -cwd "${REPO_ROOT}" -env "all,E0_ENV_FILE=${ENV_FILE},RUN=${RUN:-0}")

case "${STEP}" in
    select)
        bsub "${COMMON[@]}" -J "e0_select_proxy" -q "${CPU_QUEUE}" -n 4 -W 12:00 \
            -M 64G -R "select[mem>64G] rusage[mem=64G] span[hosts=1]" \
            -o "${LOG_DIR}/select_%J.out" -e "${LOG_DIR}/select_%J.err" \
            analysis/selection_e0/select_proxy.lsf
        ;;
    tokenize)
        SILVER="${DATA_DIRECTORY_PATH}/silver/${PROXY_NAME}"
        if [ ! -d "${SILVER}" ] || [ -z "$(ls -A "${SILVER}")" ]; then
            echo "ERROR: ${SILVER} is missing or empty: run the select step with RUN=1 first." >&2
            exit 1
        fi
        JOB_NAME="${PROXY_NAME}_cell-graph-tokenizer_${TOKENIZE_CORES}"
        if bjobs -noheader -J "${PROXY_NAME}_cell-graph-tokenizer_*" 2>/dev/null | grep -q .; then
            echo "ERROR: a tokenization job of ${PROXY_NAME} is already in LSF (bjobs -J '${PROXY_NAME}_*')." >&2
            exit 1
        fi
        TOKENIZE_LOG_DIR="logs/${PROXY_NAME}/farm-job-outputs/cell-graph-tokenizer"
        mkdir -p "${TOKENIZE_LOG_DIR}"
        bsub "${COMMON[@]}" -J "${JOB_NAME}" -q "${TOKENIZE_QUEUE}" -n "${TOKENIZE_CORES}" -W "${TOKENIZE_WALLTIME}" \
            -M "${TOKENIZE_RAM}" -R "select[mem>${TOKENIZE_RAM}] rusage[mem=${TOKENIZE_RAM}] span[hosts=1]" \
            -o "${TOKENIZE_LOG_DIR}/cell-graph-tokenizer_${TOKENIZE_CORES}_%J.out" \
            -e "${TOKENIZE_LOG_DIR}/cell-graph-tokenizer_${TOKENIZE_CORES}_%J.err" \
            analysis/selection_e0/tokenize_proxy.lsf
        ;;
    lists)
        if [ ! -f "${PROXY_DATASET_PATH}/terra_tokenizer_params.json" ]; then
            echo "ERROR: ${PROXY_DATASET_PATH} has no settings record: tokenize the proxy first." >&2
            exit 1
        fi
        bsub "${COMMON[@]}" -J "e0_index_lists" -q "${CPU_QUEUE}" -n 4 -W 12:00 \
            -M 96G -R "select[mem>96G] rusage[mem=96G] span[hosts=1]" \
            -o "${LOG_DIR}/lists_%J.out" -e "${LOG_DIR}/lists_%J.err" \
            analysis/selection_e0/build_index_lists.lsf
        ;;
    *)
        echo "${USAGE}" >&2
        exit 1
        ;;
esac
