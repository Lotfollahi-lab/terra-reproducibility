#!/bin/bash
# Submit an extra tokenization job for a dataset on another queue, e.g. a
# copy on cpu-interactive of a job that is waiting on hugemem. The job runs
# run_tokenization_locked.sh, so only one job per dataset tokenizes; give
# the waiting job the same lock with bmod -Z (see below), or it would not
# know about the copy.
#
# Usage, from the terra-reproducibility root:
#   bash analysis/tokenization/submit_tokenization_twin.sh <dataset> <queue> <ram> <walltime> [cores]
#   e.g. bash analysis/tokenization/submit_tokenization_twin.sh xht43-2b_1p cpu-interactive 672G 12:00
# Make a waiting job use the lock too (keeps its place in the queue):
#   bmod -G team361 -Z "analysis/tokenization/run_tokenization_locked.sh config/cell-graph-tokenizer/cell-graph-tokenizer_<dataset>.yaml" <jobid>
GROUP="team361"
TOKENIZER_NAME="cell-graph-tokenizer"
DATASET="${1:?usage: submit_tokenization_twin.sh <dataset> <queue> <ram> <walltime> [cores]}"
QUEUE="${2:?queue}"
RAM="${3:?ram, e.g. 672G}"
TIME="${4:?walltime, e.g. 12:00}"
CORES="${5:-12}"

if [[ "$QUEUE" == *gpu* || "$QUEUE" == inference ]]; then
    echo "Queue '${QUEUE}' is a GPU queue; tokenization runs on CPU queues." >&2
    exit 1
fi
CWD="${CWD:-$(pwd)}"
CONFIG_FILE="config/${TOKENIZER_NAME}/${TOKENIZER_NAME}_${DATASET}.yaml"
SCRIPT="analysis/tokenization/run_tokenization_locked.sh"
if [[ ! -f "${CWD}/${CONFIG_FILE}" || ! -f "${CWD}/${SCRIPT}" ]]; then
    echo "${CWD}/${CONFIG_FILE} or ${CWD}/${SCRIPT} not found; run from the terra-reproducibility root." >&2
    exit 1
fi
LOG_DIR="logs/${DATASET}/farm-job-outputs/${TOKENIZER_NAME}"
mkdir -p "${CWD}/${LOG_DIR}"

echo "Submitting ${DATASET} on ${QUEUE}: ${CORES} cores, ${RAM}, ${TIME} (locked)"
bsub \
    -G "${GROUP}" \
    -n "${CORES}" \
    -q "${QUEUE}" \
    -M "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}] span[hosts=1]" \
    -W "${TIME}" \
    -cwd "${CWD}" \
    -o "${LOG_DIR}/${TOKENIZER_NAME}_${CORES}_%J.out" \
    -e "${LOG_DIR}/${TOKENIZER_NAME}_${CORES}_%J.err" \
    -J "${DATASET}_${TOKENIZER_NAME}_${CORES}" \
    "${SCRIPT}" "${CONFIG_FILE}"
