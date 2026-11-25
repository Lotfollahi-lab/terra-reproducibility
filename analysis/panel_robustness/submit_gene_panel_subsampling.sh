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
#
# Environment:
#   LSF_GROUP     LSF group of every job (default s10396, the cost code; team361 does not exist on the GPU cluster)
#   GPU_QUEUE     queue of the embed array (default inference)
#   CPU_QUEUE     queue of prepare, evaluate and summarize (default: the first of normal, long, cpu-interactive that
#                 this cluster has; without one, the GPU queue with one GPU per job)
#   MAX_GPU_JOBS  embed elements at a time (default 8); MAX_CPU_JOBS: evaluate elements at a time (default 16)
#   FORCE=1       embed and evaluate the conditions again

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RUN_NAME=${1:?Usage: bash submit_gene_panel_subsampling.sh RUN_NAME [configure options]}
shift
ARTIFACTS=${ARTIFACTS:-/nfs/team361/sb75/terra-reproducibility/artifacts/panel_robustness}
OUTPUT_FOLDER_PATH=${ARTIFACTS}/${RUN_NAME}
LOG_FOLDER=$(cd ../.. && pwd)/logs/panel_robustness/${RUN_NAME}
LSF_GROUP=${LSF_GROUP:-s10396}
GPU_QUEUE=${GPU_QUEUE:-inference}
GPU_REQUEST="num=1:mode=exclusive_process"

# CPU steps: a CPU queue of this cluster if there is one, else the GPU queue (which needs a GPU request)
if [ -z "${CPU_QUEUE:-}" ]; then
    for queue in normal long cpu-interactive; do
        if bqueues "${queue}" > /dev/null 2>&1; then
            CPU_QUEUE=${queue}
            break
        fi
    done
fi
CPU_QUEUE=${CPU_QUEUE:-${GPU_QUEUE}}
CPU_GPU_OPTIONS=()
if [ "${CPU_QUEUE}" = "${GPU_QUEUE}" ]; then
    CPU_GPU_OPTIONS=(-gpu "${GPU_REQUEST}")
    echo "The CPU steps run on ${GPU_QUEUE} with one GPU each (no normal, long or cpu-interactive queue found;" \
         "set CPU_QUEUE to use another queue)." >&2
fi
echo "LSF group ${LSF_GROUP}; embed on ${GPU_QUEUE}; prepare, evaluate and summarize on ${CPU_QUEUE}."

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
    output=$(bsub "$@" 2>&1) || true
    id=$(sed -n 's/^Job <\([0-9]*\)>.*/\1/p' <<< "${output}")
    if [ -z "${id}" ]; then
        echo "ERROR: bsub failed: ${output}" >&2
        exit 1
    fi
    echo "${id}"
}

# If a submission fails, remove the jobs of this run that were already submitted (they would wait forever)
SUBMITTED=()
COMPLETE=0
cleanup() {
    if [ "${COMPLETE}" != 1 ] && [ "${#SUBMITTED[@]}" -gt 0 ]; then
        echo "Removing the jobs submitted before the failure: ${SUBMITTED[*]}" >&2
        bkill "${SUBMITTED[@]}" > /dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

CPU_OPTIONS=(-G "${LSF_GROUP}" -q "${CPU_QUEUE}" ${CPU_GPU_OPTIONS[@]+"${CPU_GPU_OPTIONS[@]}"})
PREPARE_ID=$(submit "${CPU_OPTIONS[@]}" -J "gps_prepare_${RUN_NAME}" \
    -o "${LOG_FOLDER}/prepare_%J.out" -e "${LOG_FOLDER}/prepare_%J.err" < gene_panel_subsampling_prepare.lsf)
SUBMITTED+=("${PREPARE_ID}")
EMBED_ID=$(submit -G "${LSF_GROUP}" -q "${GPU_QUEUE}" -gpu "${GPU_REQUEST}" \
    -J "gps_embed_${RUN_NAME}[1-${N_CONDITIONS}]%${MAX_GPU_JOBS:-8}" -w "done(${PREPARE_ID})" \
    -o "${LOG_FOLDER}/embed_%J_%I.out" -e "${LOG_FOLDER}/embed_%J_%I.err" < gene_panel_subsampling_embed.lsf)
SUBMITTED+=("${EMBED_ID}")
EVALUATE_ID=$(submit "${CPU_OPTIONS[@]}" -J "gps_evaluate_${RUN_NAME}[1-${N_CONDITIONS}]%${MAX_CPU_JOBS:-16}" \
    -w "ended(${EMBED_ID})" -o "${LOG_FOLDER}/evaluate_%J_%I.out" -e "${LOG_FOLDER}/evaluate_%J_%I.err" \
    < gene_panel_subsampling_evaluate.lsf)
SUBMITTED+=("${EVALUATE_ID}")
SUMMARIZE_ID=$(submit "${CPU_OPTIONS[@]}" -J "gps_summarize_${RUN_NAME}" -w "ended(${EVALUATE_ID})" \
    -o "${LOG_FOLDER}/summarize_%J.out" -e "${LOG_FOLDER}/summarize_%J.err" < gene_panel_subsampling_summarize.lsf)
SUBMITTED+=("${SUMMARIZE_ID}")
# The prepare job kills the jobs that wait for it if it fails (they would stay pending forever)
echo "${EMBED_ID} ${EVALUATE_ID} ${SUMMARIZE_ID}" > "${OUTPUT_FOLDER_PATH}/.waiting_jobs_${PREPARE_ID}"
COMPLETE=1

cat <<EOF
Run ${RUN_NAME}: ${N_CONDITIONS} conditions.
  prepare   ${PREPARE_ID} (${CPU_QUEUE})
  embed     ${EMBED_ID} (array, ${GPU_QUEUE}, GPU)
  evaluate  ${EVALUATE_ID} (array, ${CPU_QUEUE})
  summarize ${SUMMARIZE_ID} (${CPU_QUEUE})
Output: ${OUTPUT_FOLDER_PATH}
Logs:   ${LOG_FOLDER}
If prepare fails, it kills the other jobs (see ${LOG_FOLDER}/prepare_${PREPARE_ID}.err).
EOF
