#!/bin/bash
# Submit the ablation job array (run_ablation_array.lsf) of a dataset: one element per manifest row that still needs
# work, run in manifest order with at most MAX_CONCURRENT elements of 4 GPUs at a time (default 2, i.e. 8 GPUs). Before
# submitting, list_pending_runs.py checks every row of INDICES on the login node with the jobs' own checks and leaves
# out (and prints) the rows whose evaluation is up to date; a row is submitted if its training has not finished or its
# evaluation is missing or stale (evaluation code, TERRA version, checkpoint, evaluation data, labels or precision
# changed). The array reads the campaign from ARTIFACT_FOLDER_PATH (manifest.tsv, configs and dataset.env of
# generate_ablation_runs.py) and runs in ENV_PATH; both come from the env file and are passed through bsub -env.
#
# Usage: bash submit_ablation_array.sh ENV_FILE [INDICES (default: all rows; e.g. 1-4 for the pilot)]
#            [MAX_CONCURRENT (default: 2)] [RUN_LIMIT (default: RUN_LIMIT of the env file)]
# Example: bash submit_ablation_array.sh datasets/mouse.env 1-4        (pilot)
#          bash submit_ablation_array.sh datasets/mouse.env 5-51 1     (rows 5-51, 4 GPUs at a time)
#          bash submit_ablation_array.sh datasets/mouse.env "" 1 48:00 (all rows, run limit 48 h)
# Options (env file; the environment of the command where the env file does not set them):
#   RUN_LIMIT: bsub -W of each training element ([hours:]minutes; empty: the queue's default RUNLIMIT, check it with
#     `bqueues -l training-parallel | grep -A 2 RUNLIMIT`). An element killed at the limit is not requeued; resubmitting
#     it resumes from its latest checkpoint.
#   SEPARATE_EVALUATION=1: the training elements only train and package; an evaluation array of 1 GPU per element
#     (evaluate_ablation_array.lsf, job name <tag>_ablation_eval) evaluates each run once its training element has
#     ended (bsub -w "ended(<training job>[*])"; at most MAX_CONCURRENT at a time), and the rows that need only their
#     evaluation go to a second evaluation array instead of waiting for 4 GPUs. Default: evaluation in the training job.
#   EVAL_FP32=1: embed in fp32 instead of bf16 autocast (--fp32_embeddings; recorded in evaluation_complete, so changing
#     it evaluates every run again).
#   FORCE_EVAL=1 (environment only, e.g. FORCE_EVAL=1 bash submit_ablation_array.sh ...): evaluate every run of INDICES
#     again, whatever evaluation_complete records (e.g. after editing a label file in place).
# Submit one array at a time (the script refuses while an array of the campaign, training or evaluation, is pending or
# running): each array has its own concurrency limit, and two arrays would work on the same runs. The exception is the
# arrays this script submits together with SEPARATE_EVALUATION=1, which never work on a run at the same time.
# Retry failed elements: submit the same INDICES again (finished rows are left out), or brequeue -e "<JOBID>[<INDEX>]"
# while LSF still knows the array. Change concurrency of a submitted array: bmod -J "%<N>" <JOBID>
# Re-evaluate finished runs on 1 GPU each: bash submit_evaluation_array.sh ENV_FILE [INDICES] [MAX_CONCURRENT]

set -euo pipefail

# submit_evaluation_array.sh runs this script with SUBMIT_EVALUATION_ONLY=1: only the evaluation array, for the rows
# that have finished training and need their evaluation (default concurrency 4, i.e. 4 GPUs)
EVALUATION_ONLY="${SUBMIT_EVALUATION_ONLY:-0}"
if [ "${EVALUATION_ONLY}" = "1" ]; then
    USAGE="submit_evaluation_array.sh ENV_FILE [INDICES] [MAX_CONCURRENT]"
    MAX_ARGS=3
else
    USAGE="submit_ablation_array.sh ENV_FILE [INDICES] [MAX_CONCURRENT] [RUN_LIMIT]"
    MAX_ARGS=4
fi
if [ "$#" -lt 1 ] || [ "$#" -gt "${MAX_ARGS}" ] || [ ! -f "$1" ]; then
    echo "Usage: bash ${USAGE}  (ENV_FILE: e.g. datasets/ovca.env)" >&2
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
INDICES="${2:-}"
if [ "${EVALUATION_ONLY}" = "1" ]; then
    MAX_CONCURRENT="${3:-4}"
else
    MAX_CONCURRENT="${3:-2}"
fi
RUN_LIMIT="${4:-${RUN_LIMIT:-}}"
# Evaluation elements: two embedding passes and the metrics take minutes; the limit only catches hung jobs
EVALUATION_RUN_LIMIT="4:00"
FORCE_EVAL="${FORCE_EVAL:-0}"
EVAL_FP32="${EVAL_FP32:-0}"
SEPARATE_EVALUATION="${SEPARATE_EVALUATION:-0}"
if ! [[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_CONCURRENT must be a positive integer, not '${MAX_CONCURRENT}'." >&2
    exit 1
fi
if [ -n "${RUN_LIMIT}" ] && ! [[ "${RUN_LIMIT}" =~ ^([0-9]+:)?[0-9]+$ ]]; then
    echo "ERROR: RUN_LIMIT must be [hours:]minutes (bsub -W), e.g. 48:00, not '${RUN_LIMIT}'." >&2
    exit 1
fi
for OPTION in "FORCE_EVAL=${FORCE_EVAL}" "EVAL_FP32=${EVAL_FP32}" "SEPARATE_EVALUATION=${SEPARATE_EVALUATION}"; do
    if ! [[ "${OPTION#*=}" =~ ^[01]$ ]]; then
        echo "ERROR: ${OPTION%%=*} must be 0 or 1, not '${OPTION#*=}'." >&2
        exit 1
    fi
done

# Two active arrays of one campaign would train or evaluate the same runs into the same folders (the pattern also
# matches the evaluation arrays, <tag>_ablation_eval)
ACTIVE_JOBS=$(bjobs -noheader -o "jobid stat" -J "${EXPERIMENT_TAG}_ablation*" 2>/dev/null \
    | grep -E "PEND|RUN|SUSP" || true)
if [ -n "${ACTIVE_JOBS}" ]; then
    echo "ERROR: an array of this campaign (${EXPERIMENT_TAG}_ablation*) is still pending or running:" >&2
    echo "${ACTIVE_JOBS}" >&2
    echo "Wait for it to finish or bkill it before submitting again." >&2
    exit 1
fi

# Rows that still need work, with the checks of the jobs and the values of dataset.env (as the jobs read them)
LIST_OPTIONS=()
if [ "${FORCE_EVAL}" = "1" ]; then
    LIST_OPTIONS+=(--force_evaluation)
fi
if [ "${EVAL_FP32}" = "1" ]; then
    LIST_OPTIONS+=(--fp32_embeddings)
fi
if [ "${EVALUATION_ONLY}" = "1" ]; then
    LIST_OPTIONS+=(--evaluation_only)
fi
# shellcheck disable=SC2086
PENDING_INDICES=$(
    source "${DATASET_ENV_FILE_PATH}"
    PYTHONPATH="${CODE_PATH}/src" "${ENV_PATH}/bin/python" list_pending_runs.py \
        --manifest_file_path "${MANIFEST_FILE_PATH}" --indices "${INDICES}" \
        --norm_factor_file_path "${NORM_FACTOR_FILE_PATH}" --dataset_name "${DATASET_NAME}" \
        --label_file_paths ${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}} --label_keys ${LABEL_KEYS} \
        --unassigned_label "${UNASSIGNED_LABEL}" ${LIST_OPTIONS[@]+"${LIST_OPTIONS[@]}"}
)
TRAIN_INDICES=$(printf '%s\n' "${PENDING_INDICES}" | awk -F'\t' '$1 == "train" { print $2 }')
EVALUATE_INDICES=$(printf '%s\n' "${PENDING_INDICES}" | awk -F'\t' '$1 == "evaluate" { print $2 }')
WORK_INDICES=$(printf '%s\n' "${PENDING_INDICES}" | awk -F'\t' '$1 == "work" { print $2 }')
if [ -z "${WORK_INDICES}" ]; then
    echo "Nothing to submit for rows ${INDICES:-(all)} (see above)."
    exit 0
fi

mkdir -p "${ARTIFACT_FOLDER_PATH}/logs"
JOB_ENV="all,ARTIFACT_FOLDER_PATH=${ARTIFACT_FOLDER_PATH},ENV_PATH=${ENV_PATH}"
JOB_ENV="${JOB_ENV},FORCE_EVAL=${FORCE_EVAL},EVAL_FP32=${EVAL_FP32}"
# Submits an evaluation array: submit_evaluation_array INDICES [DEPENDENCY]
submit_evaluation_array() {
    local dependency=()
    if [ -n "${2:-}" ]; then
        dependency=(-w "$2")
    fi
    echo "Evaluation array, rows ${1}${2:+ (element N waits for training element N: -w $2)}:"
    bsub -J "${EXPERIMENT_TAG}_ablation_eval[${1}]%${MAX_CONCURRENT}" ${dependency[@]+"${dependency[@]}"} \
        -env "${JOB_ENV}" \
        -W "${EVALUATION_RUN_LIMIT}" \
        -o "${ARTIFACT_FOLDER_PATH}/logs/evaluate_%J_%I.out" \
        -e "${ARTIFACT_FOLDER_PATH}/logs/evaluate_%J_%I.err" \
        < evaluate_ablation_array.lsf
}
if [ "${EVALUATION_ONLY}" = "1" ]; then
    submit_evaluation_array "${WORK_INDICES}"
    exit 0
fi

# Training array: every row that needs work, or with SEPARATE_EVALUATION=1 only the rows to train, each followed by
# its 1-GPU evaluation element (the same indices, as the one-to-one dependency [*] requires)
if [ "${SEPARATE_EVALUATION}" = "1" ]; then
    ARRAY_INDICES="${TRAIN_INDICES}"
else
    ARRAY_INDICES="${WORK_INDICES}"
fi
if [ -n "${ARRAY_INDICES}" ]; then
    RUN_LIMIT_OPTION=()
    if [ -n "${RUN_LIMIT}" ]; then
        RUN_LIMIT_OPTION=(-W "${RUN_LIMIT}")
    fi
    echo "Training array, rows ${ARRAY_INDICES} (at most ${MAX_CONCURRENT} at a time, run limit" \
         "${RUN_LIMIT:-of the queue}, SEPARATE_EVALUATION=${SEPARATE_EVALUATION}):"
    SUBMISSION=$(bsub -J "${EXPERIMENT_TAG}_ablation[${ARRAY_INDICES}]%${MAX_CONCURRENT}" \
        ${RUN_LIMIT_OPTION[@]+"${RUN_LIMIT_OPTION[@]}"} \
        -env "${JOB_ENV},SEPARATE_EVALUATION=${SEPARATE_EVALUATION}" \
        -o "${ARTIFACT_FOLDER_PATH}/logs/train_eval_%J_%I.out" \
        -e "${ARTIFACT_FOLDER_PATH}/logs/train_eval_%J_%I.err" \
        < run_ablation_array.lsf)
    echo "${SUBMISSION}"
    if [ "${SEPARATE_EVALUATION}" = "1" ]; then
        TRAINING_JOB_ID=$(printf '%s\n' "${SUBMISSION}" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p')
        if [ -z "${TRAINING_JOB_ID}" ]; then
            echo "ERROR: no job id in the bsub output; submit the evaluation with submit_evaluation_array.sh once the" \
                 "training array has ended." >&2
            exit 1
        fi
        submit_evaluation_array "${ARRAY_INDICES}" "ended(${TRAINING_JOB_ID}[*])"
    fi
fi
if [ "${SEPARATE_EVALUATION}" = "1" ] && [ -n "${EVALUATE_INDICES}" ]; then
    submit_evaluation_array "${EVALUATE_INDICES}"
fi
