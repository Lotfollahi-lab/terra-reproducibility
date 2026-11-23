#!/bin/bash
# Submit a gene panel subsampling run: writes the run configuration (on this node, no data loaded), then submits the
# prepare job, the embed array (one GPU element per condition, after prepare), the evaluate array (after every embed
# element has ended) and the summarize job (after every evaluate element has ended).
#
# Usage: bash submit_gene_panel_subsampling.sh RUN_NAME [options of `gene_panel_subsampling.py configure`]
# Examples:
#   bash submit_gene_panel_subsampling.sh xhs1000_3sections_terra96m
#   bash submit_gene_panel_subsampling.sh xhs1000_3sections_terra112m \
#       --model_folder_path /nfs/team361/sb75/terra-reproducibility/artifacts/models/11012026_190057_836
#
# Results: ${ARTIFACTS}/RUN_NAME/results (default ARTIFACTS: /nfs/team361/sb75/terra-reproducibility/artifacts/
# panel_robustness). Logs: logs/panel_robustness/RUN_NAME/. Resubmitting the same RUN_NAME with the same options
# skips the conditions that are done (FORCE=1 recomputes them); other options need a new RUN_NAME (or --overwrite).
# Environment: MAX_GPU_JOBS (embed elements at a time, default 8), MAX_CPU_JOBS (evaluate elements at a time,
# default 16), FORCE=1.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RUN_NAME=${1:?Usage: bash submit_gene_panel_subsampling.sh RUN_NAME [configure options]}
shift
ARTIFACTS=${ARTIFACTS:-/nfs/team361/sb75/terra-reproducibility/artifacts/panel_robustness}
OUTPUT_FOLDER_PATH=${ARTIFACTS}/${RUN_NAME}
LOG_FOLDER=$(cd ../.. && pwd)/logs/panel_robustness/${RUN_NAME}
mkdir -p "${OUTPUT_FOLDER_PATH}" "${LOG_FOLDER}"

source /nfs/team361/sb75/.venvs/terra/bin/activate
N_CONDITIONS=$(python gene_panel_subsampling.py configure --output_folder_path "${OUTPUT_FOLDER_PATH}" "$@" | tail -n 1)
if ! [[ "${N_CONDITIONS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: configure did not return the number of conditions (got '${N_CONDITIONS}')." >&2
    exit 1
fi
export OUTPUT_FOLDER_PATH
if [ -n "${FORCE:-}" ]; then export FORCE; fi

# Job id from the bsub message "Job <id> is submitted to queue <queue>."
submit() {
    local output id
    output=$(bsub "$@")
    id=$(sed -n 's/^Job <\([0-9]*\)>.*/\1/p' <<< "${output}")
    if [ -z "${id}" ]; then
        echo "ERROR: bsub failed: ${output}" >&2
        exit 1
    fi
    echo "${id}"
}

PREPARE_ID=$(submit -J "gps_prepare_${RUN_NAME}" -o "${LOG_FOLDER}/prepare_%J.out" -e "${LOG_FOLDER}/prepare_%J.err" \
    < gene_panel_subsampling_prepare.lsf)
EMBED_ID=$(submit -J "gps_embed_${RUN_NAME}[1-${N_CONDITIONS}]%${MAX_GPU_JOBS:-8}" -w "done(${PREPARE_ID})" \
    -o "${LOG_FOLDER}/embed_%J_%I.out" -e "${LOG_FOLDER}/embed_%J_%I.err" < gene_panel_subsampling_embed.lsf)
EVALUATE_ID=$(submit -J "gps_evaluate_${RUN_NAME}[1-${N_CONDITIONS}]%${MAX_CPU_JOBS:-16}" -w "ended(${EMBED_ID})" \
    -o "${LOG_FOLDER}/evaluate_%J_%I.out" -e "${LOG_FOLDER}/evaluate_%J_%I.err" < gene_panel_subsampling_evaluate.lsf)
SUMMARIZE_ID=$(submit -J "gps_summarize_${RUN_NAME}" -w "ended(${EVALUATE_ID})" \
    -o "${LOG_FOLDER}/summarize_%J.out" -e "${LOG_FOLDER}/summarize_%J.err" < gene_panel_subsampling_summarize.lsf)

cat <<EOF
Run ${RUN_NAME}: ${N_CONDITIONS} conditions.
  prepare   ${PREPARE_ID}
  embed     ${EMBED_ID} (array, GPU)
  evaluate  ${EVALUATE_ID} (array)
  summarize ${SUMMARIZE_ID}
Output: ${OUTPUT_FOLDER_PATH}
Logs:   ${LOG_FOLDER}
If prepare fails, the other jobs never start: bkill ${EMBED_ID} ${EVALUATE_ID} ${SUMMARIZE_ID}, fix, resubmit.
EOF
