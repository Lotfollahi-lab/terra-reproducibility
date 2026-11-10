#!/bin/bash
# Sourced by the job arrays run_ablation_array.lsf and evaluate_ablation_array.lsf (from this directory, their working
# directory): checks the campaign, sources its dataset.env and reads the run of manifest.tsv row LSB_JOBINDEX (INDEX,
# RUN_ID, ..., RUN_FOLDER_PATH, EVAL_FOLDER_PATH, CONFIG_HASH); defines the evaluation of the run (EVALUATION_ARGS,
# check_evaluation, evaluate_run). Options of the submit scripts, passed through bsub -env: FORCE_EVAL=1 evaluates the
# run again even if its evaluation is up to date, EVAL_FP32=1 embeds in fp32 instead of bf16 autocast.

if [ -z "${ARTIFACT_FOLDER_PATH:-}" ] || [ -z "${ENV_PATH:-}" ]; then
    echo "ERROR: ARTIFACT_FOLDER_PATH and ENV_PATH must be set; submit with submit_ablation_array.sh or" \
         "submit_evaluation_array.sh <env file>." >&2
    exit 1
fi
DATASET_ENV_FILE_PATH="${ARTIFACT_FOLDER_PATH}/dataset.env"
MANIFEST_FILE_PATH="${ARTIFACT_FOLDER_PATH}/manifest.tsv"
for FILE_PATH in "${DATASET_ENV_FILE_PATH}" "${MANIFEST_FILE_PATH}"; do
    if [ ! -f "${FILE_PATH}" ]; then
        echo "ERROR: ${FILE_PATH} does not exist; run generate_ablation_runs.py first." >&2
        exit 1
    fi
done
# DATASET_NAME, SILVER_FILE_PATHS, LABEL_FILE_PATHS, LABEL_KEYS, UNASSIGNED_LABEL, NORM_FACTOR_FILE_PATH,
# EXPERIMENT_TAG, CODE_PATH, CODE_VERSION, EVAL_GENE_SET (only from dataset.env, not from the submitting shell's
# environment that bsub -env all passes on)
unset EVAL_GENE_SET
source "${DATASET_ENV_FILE_PATH}"
# dataset.env written before LABEL_FILE_PATHS existed: the labels are in the silver files
LABEL_FILE_PATHS="${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}}"
ADVANCED_BASH_SCRIPT_PATH="/software/isg/advanced-bsub-script"
if [ -z "${EVAL_GENE_SET:-}" ]; then
    echo "ERROR: ${DATASET_ENV_FILE_PATH} has no EVAL_GENE_SET; regenerate it with generate_ablation_runs.py." >&2
    exit 1
fi

# The configs were generated for the pinned code at CODE_PATH; stop if it has another version
if [ "$(cat "${CODE_PATH}/CODE_VERSION" 2>/dev/null)" != "${CODE_VERSION}" ]; then
    echo "ERROR: ${CODE_PATH}/CODE_VERSION ($(cat "${CODE_PATH}/CODE_VERSION" 2>/dev/null || echo missing)) differs" \
         "from the code version of the configs (${CODE_VERSION}); set up that code version or regenerate the configs." >&2
    exit 1
fi

# Run of this array element: manifest row LSB_JOBINDEX, with the columns written by generate_ablation_runs.py
MANIFEST_COLUMNS="index run_id run_name variant category label replicate seed code_version config_file_path"
MANIFEST_COLUMNS="${MANIFEST_COLUMNS} run_folder_path eval_folder_path"
if [ "$(head -n 1 "${MANIFEST_FILE_PATH}" | tr '\t' ' ')" != "${MANIFEST_COLUMNS}" ]; then
    echo "ERROR: the columns of ${MANIFEST_FILE_PATH} are not '${MANIFEST_COLUMNS}'; regenerate it with" \
         "generate_ablation_runs.py." >&2
    exit 1
fi
ROWS=$(awk -F'\t' -v idx="${LSB_JOBINDEX}" 'NR > 1 && $1 == idx' "${MANIFEST_FILE_PATH}")
N_ROWS=$(printf '%s' "${ROWS}" | grep -c '^' || true)
if [ "${N_ROWS}" -ne 1 ]; then
    echo "ERROR: ${MANIFEST_FILE_PATH} has ${N_ROWS} rows with index ${LSB_JOBINDEX} (expected 1); check the array" \
         "indices against the manifest." >&2
    exit 1
fi
N_FIELDS=$(printf '%s\n' "${ROWS}" | awk -F'\t' '{ print NF }')
N_EMPTY_FIELDS=$(printf '%s\n' "${ROWS}" | awk -F'\t' '{ n = 0; for (i = 1; i <= NF; i++) if ($i == "") n++; print n }')
read -r -a COLUMN_NAMES <<< "${MANIFEST_COLUMNS}"
N_COLUMNS="${#COLUMN_NAMES[@]}"
if [ "${N_FIELDS}" -ne "${N_COLUMNS}" ] || [ "${N_EMPTY_FIELDS}" -ne 0 ]; then
    echo "ERROR: manifest row ${LSB_JOBINDEX} has ${N_FIELDS} tab-separated fields (${N_EMPTY_FIELDS} empty);" \
         "expected ${N_COLUMNS} non-empty fields. Regenerate the manifest with generate_ablation_runs.py." >&2
    exit 1
fi
IFS=$'\t' read -r INDEX RUN_ID PLAN_RUN_NAME VARIANT CATEGORY LABEL REPLICATE SEED RUN_CODE_VERSION CONFIG_FILE \
    RUN_FOLDER_PATH EVAL_FOLDER_PATH <<< "${ROWS}"
echo "Run ${INDEX}: ${RUN_ID} (${CATEGORY}: ${LABEL}; replicate ${REPLICATE}, seed ${SEED}; ${PLAN_RUN_NAME})"
if [ "${RUN_CODE_VERSION}" != "${CODE_VERSION}" ]; then
    echo "ERROR: manifest row ${INDEX} has code version '${RUN_CODE_VERSION}', dataset.env '${CODE_VERSION}';" \
         "regenerate the configs." >&2
    exit 1
fi
CONFIG_HASH="${RUN_ID##*_}"
if ! [[ "${CONFIG_HASH}" =~ ^[0-9a-f]{8}$ ]]; then
    echo "ERROR: run id ${RUN_ID} does not end with an 8-digit config hash." >&2
    exit 1
fi

# Arguments of evaluate_ablation_run.py, the same for its --check_only (lists are space-separated, hence unquoted;
# UNASSIGNED_LABEL is one value or <key>=<value> pairs separated by ';', passed as one argument). The evaluation data
# follow from the run's training dataset and the current EVAL_GENE_SET, not from the run config
# shellcheck disable=SC2206
EVALUATION_ARGS=(--model_folder_path "${RUN_FOLDER_PATH}" --output_folder_path "${EVAL_FOLDER_PATH}"
    --dataset_name "${DATASET_NAME}" --label_file_paths ${LABEL_FILE_PATHS} --label_keys ${LABEL_KEYS}
    --unassigned_label "${UNASSIGNED_LABEL}" --eval_gene_set "${EVAL_GENE_SET}")
if [ "${EVAL_FP32:-0}" = "1" ]; then
    EVALUATION_ARGS+=(--fp32_embeddings)
fi

# Stops the job (exit 0) if the evaluation of the run is up to date: evaluation_complete records the current inputs
# (evaluation code, TERRA version, checkpoint, evaluation data, labels and precision) and every metric file is
# complete (evaluate_ablation_run.py --check_only, which loads torch and terra: call it after activating the
# environment). Returns if the run needs its evaluation (exit code 3: also if it is not trained or packaged yet) or with
# FORCE_EVAL=1; stops the job on any other error.
check_evaluation() {
    if [ "${FORCE_EVAL:-0}" = "1" ]; then
        echo "FORCE_EVAL=1: evaluating again whatever evaluation_complete records."
        return
    fi
    local status=0
    python3 -u evaluate_ablation_run.py "${EVALUATION_ARGS[@]}" --check_only || status=$?
    if [ "${status}" -eq 0 ]; then
        echo "Already trained and evaluated."
        exit 0
    elif [ "${status}" -ne 3 ]; then
        echo "ERROR: could not check the evaluation of ${RUN_ID} (exit code ${status}); inspect it before" \
             "resubmitting." >&2
        exit 1
    fi
}

# Evaluates the packaged run: all modes again with FORCE_EVAL=1, otherwise unless its evaluation is up to date
evaluate_run() {
    if [ "${FORCE_EVAL:-0}" = "1" ]; then
        python3 -u evaluate_ablation_run.py "${EVALUATION_ARGS[@]}" --overwrite
    else
        python3 -u evaluate_ablation_run.py "${EVALUATION_ARGS[@]}"
    fi
}
