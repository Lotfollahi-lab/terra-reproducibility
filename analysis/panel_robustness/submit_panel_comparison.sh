#!/bin/bash
# Submit the comparison of a panel-training run's embedding runs with the reference run (compare_panel_runs.lsf).
#
# Usage: bash submit_panel_comparison.sh NAME [options of compare_panel_runs.py, e.g. --representations terra_cell]
#   or:  RUNS="terra96m=/path/run1 other=/path/run2" bash submit_panel_comparison.sh NAME   (any runs)
# Output: ${TRAINING_ROOT}/NAME/comparison (results and figures); logs: logs/panel_training/NAME/.
# Environment: LSF_GROUP (default s10396), CPU_QUEUE / GPU_QUEUE as in submit_panel_training_data.sh.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

NAME=${1:?Usage: bash submit_panel_comparison.sh NAME [compare options]}
shift
TRAINING_ROOT=${TRAINING_ROOT:-/nfs/team361/sb75/terra-reproducibility/artifacts/panel_training}
TRAINING_FOLDER_PATH=${TRAINING_ROOT}/${NAME}
OUTPUT_FOLDER_PATH=${OUTPUT_FOLDER_PATH:-${TRAINING_FOLDER_PATH}/comparison}
LOG_FOLDER=$(cd ../.. && pwd)/logs/panel_training/${NAME}
LSF_GROUP=${LSF_GROUP:-s10396}
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
QUEUE_OPTIONS=(-G "${LSF_GROUP}" -q "${CPU_QUEUE}")
if [ "${CPU_QUEUE}" = "${GPU_QUEUE}" ]; then
    QUEUE_OPTIONS+=(-gpu "num=1:mode=exclusive_process")
fi

source /nfs/team361/sb75/.venvs/terra/bin/activate
if [ -z "${RUNS:-}" ]; then
    RUNS=$(python panel_training.py runs --training_folder_path "${TRAINING_FOLDER_PATH}")
fi
echo "Runs: ${RUNS}"
mkdir -p "${OUTPUT_FOLDER_PATH}" "${LOG_FOLDER}"
export OUTPUT_FOLDER_PATH RUNS
COMPARE_OPTIONS="$*"
export COMPARE_OPTIONS

OUTPUT=$(bsub "${QUEUE_OPTIONS[@]}" -J "pt_compare_${NAME}" -o "${LOG_FOLDER}/compare_%J.out" \
    -e "${LOG_FOLDER}/compare_%J.err" < compare_panel_runs.lsf 2>&1) || true
JOB_ID=$(sed -n 's/^Job <\([0-9]*\)>.*/\1/p' <<< "${OUTPUT}")
if [ -z "${JOB_ID}" ]; then
    echo "ERROR: bsub failed: ${OUTPUT}" >&2
    exit 1
fi
echo "Comparison job ${JOB_ID} (${CPU_QUEUE}); results in ${OUTPUT_FOLDER_PATH}."
