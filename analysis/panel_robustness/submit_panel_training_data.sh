#!/bin/bash
# Tokenize the training data of a panel-training run: writes the training configuration (on this node), then submits
# one CPU job per panel (panel_training_prepare_data.lsf).
#
# Usage: bash submit_panel_training_data.sh NAME [options of `panel_training.py configure`]
# Example:
#   bash submit_panel_training_data.sh xhs1000_skin_panels
#   bash submit_panel_training_data.sh xhs1000_skin_panels_disjoint --panels full disjoint_1000_r0 disjoint_2000_r0
#
# Training folder: ${TRAINING_ROOT}/NAME (default TRAINING_ROOT: /nfs/team361/sb75/terra-reproducibility/artifacts/
# panel_training); data: <data root>/NAME/<panel> (panel_training.py configure --data_root). Logs:
# logs/panel_training/NAME/. Next step: `panel_training.py make_configs` (see README.md).
#
# The jobs only need CPUs: on the main farm (farm22) they run on its normal queue with the team group, on the GPU
# cluster (tiger, no CPU queue) on the inference queue with the cost-code group and one GPU each.
#
# Environment: CPU_QUEUE (default: the first of normal, long, cpu-interactive that this cluster has, else GPU_QUEUE
# with one GPU), GPU_QUEUE (default inference), LSF_GROUP (default: team361 on a CPU queue, s10396 on the GPU queue;
# the esubs accept only team groups on the CPU queues and only cost-code groups on the AI acceleration queues),
# FORCE=1 (tokenize again).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

NAME=${1:?Usage: bash submit_panel_training_data.sh NAME [configure options]}
shift
TRAINING_ROOT=${TRAINING_ROOT:-/nfs/team361/sb75/terra-reproducibility/artifacts/panel_training}
TRAINING_FOLDER_PATH=${TRAINING_ROOT}/${NAME}
LOG_FOLDER=$(cd ../.. && pwd)/logs/panel_training/${NAME}
GPU_QUEUE=${GPU_QUEUE:-inference}

if [ -z "${CPU_QUEUE:-}" ]; then
    for queue in normal long cpu-interactive; do
        if bqueues "${queue}" > /dev/null 2>&1; then
            CPU_QUEUE=${queue}
            break
        fi
    done
fi
CPU_QUEUE=${CPU_QUEUE:-${GPU_QUEUE}}
if [ "${CPU_QUEUE}" = "${GPU_QUEUE}" ]; then
    LSF_GROUP=${LSF_GROUP:-s10396}
    QUEUE_OPTIONS=(-G "${LSF_GROUP}" -q "${CPU_QUEUE}" -gpu "num=1:mode=exclusive_process")
    echo "No normal, long or cpu-interactive queue: the CPU jobs run on ${GPU_QUEUE} with one GPU each." >&2
else
    LSF_GROUP=${LSF_GROUP:-team361}
    QUEUE_OPTIONS=(-G "${LSF_GROUP}" -q "${CPU_QUEUE}")
fi

mkdir -p "${TRAINING_FOLDER_PATH}" "${LOG_FOLDER}"
source /nfs/team361/sb75/.venvs/terra/bin/activate
N_PANELS=$(python panel_training.py configure --training_folder_path "${TRAINING_FOLDER_PATH}" --name "${NAME}" "$@" \
    | tail -n 1)
if ! [[ "${N_PANELS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: configure did not return the number of panels (got '${N_PANELS}')." >&2
    exit 1
fi
export TRAINING_FOLDER_PATH
if [ -n "${FORCE:-}" ]; then export FORCE; fi

OUTPUT=$(bsub "${QUEUE_OPTIONS[@]}" -J "pt_data_${NAME}[1-${N_PANELS}]" \
    -o "${LOG_FOLDER}/prepare_data_%J_%I.out" -e "${LOG_FOLDER}/prepare_data_%J_%I.err" \
    < panel_training_prepare_data.lsf 2>&1) || true
JOB_ID=$(sed -n 's/^Job <\([0-9]*\)>.*/\1/p' <<< "${OUTPUT}")
if [ -z "${JOB_ID}" ]; then
    echo "ERROR: bsub failed: ${OUTPUT}" >&2
    exit 1
fi
cat <<EOF
Training data of ${NAME}: ${N_PANELS} panels, job array ${JOB_ID} (queue ${CPU_QUEUE}, group ${LSF_GROUP}).
Training folder: ${TRAINING_FOLDER_PATH}
Logs:            ${LOG_FOLDER}
When it has finished, write the training configs (see README.md):
  python panel_training.py make_configs --training_folder_path ${TRAINING_FOLDER_PATH}
EOF
