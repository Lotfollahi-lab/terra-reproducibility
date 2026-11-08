#!/bin/bash
# One-time setup of the pinned TERRA code of the ablations: a plain copy (git archive) of Lotfollahi-lab/terra at
# COMMIT (dev 5d8ecbc, 8 Nov 2026), which holds all fixes of the ablation code reviews and the corpus-mean tie-break of
# ranked genes; those that change results are opt-in, and the plan switches them on. CODE_PATH and ENV_PATH are taken
# from the env file. The jobs put its src/ first on PYTHONPATH, so later changes of the farm checkout (editable install
# of the venv) cannot change running jobs. The commit is taken from REPO_PATH if that checkout has it, otherwise fetched
# from TERRA_URL into a temporary repository (REPO_PATH is not modified). The copy is built next to CODE_PATH and
# checked (it has the options of the plan; ENV_PATH imports it) before CODE_VERSION (the commit) is written and it is
# moved to CODE_PATH; generate_ablation_runs.py records CODE_VERSION in every run config. If CODE_PATH is up to date,
# only the environment check is rerun. If it exists with another version, the script stops (jobs may be running on it):
# move it away and rerun.
#
# Usage: bash setup_code.sh ENV_FILE          (e.g. datasets/mouse.env)
# Optional: REPO_PATH=<checkout> TERRA_URL=<git url> bash setup_code.sh ENV_FILE

set -euo pipefail

if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    echo "Usage: bash setup_code.sh ENV_FILE  (e.g. datasets/mouse.env)" >&2
    exit 1
fi
source "$1"
REPO_PATH="${REPO_PATH:-/nfs/team361/sb75/terra}"
TERRA_URL="${TERRA_URL:-https://github.com/Lotfollahi-lab/terra.git}"
COMMIT="5d8ecbc6a813788a0de5711964b50f8a4cf44d99"
CODE_VERSION="terra ${COMMIT}"
if [[ "${CODE_PATH}" != /* ]]; then
    echo "ERROR: CODE_PATH must be an absolute path, not ${CODE_PATH}." >&2
    exit 1
fi

# The jobs run the pinned code in ENV_PATH (fix ENV_PATH in the env file if this fails); scib_metrics (with pynndescent
# and jax) computes the batch-integration metrics of datasets with two or more sections, wandb logs the training
check_env() {
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$1/src" "${ENV_PATH}/bin/python" -c "
import terra, terra.inference.embed, terra.tokenizers.cell_tokenizers, terra.training.train
import datasets, igraph, leidenalg, scanpy, scib_metrics, squidpy, torch, wandb, yaml
assert terra.__file__.startswith('$1/'), terra.__file__
print('Environment ${ENV_PATH} imports the pinned TERRA code at $1 (torch', torch.__version__ + ').')"
}

if [ -e "${CODE_PATH}" ]; then
    if [ "$(cat "${CODE_PATH}/CODE_VERSION" 2>/dev/null)" == "${CODE_VERSION}" ]; then
        echo "${CODE_PATH} is up to date: ${CODE_VERSION}"
        check_env "${CODE_PATH}"
        exit 0
    fi
    echo "${CODE_PATH} exists with another version ($(cat "${CODE_PATH}/CODE_VERSION" 2>/dev/null || echo unknown))."
    echo "If no jobs use it: mv ${CODE_PATH} ${CODE_PATH}_old && bash setup_code.sh $1"
    exit 1
fi

# Temporary folders are removed on exit (the build folder only remains if it was not moved to CODE_PATH)
TEMPORARY_PATHS=()
trap 'rm -rf ${TEMPORARY_PATHS[@]+"${TEMPORARY_PATHS[@]}"}' EXIT
if git -C "${REPO_PATH}" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
    SOURCE_REPO_PATH="${REPO_PATH}"
    echo "Using ${COMMIT} from ${REPO_PATH}."
else
    SOURCE_REPO_PATH="$(mktemp -d)"
    TEMPORARY_PATHS+=("${SOURCE_REPO_PATH}")
    echo "${COMMIT} is not in ${REPO_PATH}; fetching it from ${TERRA_URL}..."
    git init -q --bare "${SOURCE_REPO_PATH}"
    git -C "${SOURCE_REPO_PATH}" fetch -q --depth 1 "${TERRA_URL}" "${COMMIT}"
fi
BUILD_PATH="$(mktemp -d "${CODE_PATH}.setup.XXXXXX")"
TEMPORARY_PATHS+=("${BUILD_PATH}")
git -C "${SOURCE_REPO_PATH}" archive "${COMMIT}" | tar -x -C "${BUILD_PATH}"
# The options the plan switches on (ablation_plan.yaml, tokenizer_options and training_options) and the metatoken
# inference option of the evaluation
TERRA_SRC="${BUILD_PATH}/src/terra"
for CHECK in "pad_masks_to_batch_max:masks/block_masking.py" "keep_special_tokens:inference/embed.py" \
             "stable_rank_ties:tokenizers/cell_tokenizers.py" "rank_tie_break_norm_factor:tokenizers/cell_tokenizers.py" \
             "zero_expr_from_raw_counts:tokenizers/cell_tokenizers.py" \
             "seurat_loess_span:tokenizers/cell_tokenizers.py" "aggregate_own_neighbors:tokenizers/cell_tokenizers.py"; do
    if ! grep -q "${CHECK%%:*}" "${TERRA_SRC}/${CHECK#*:}"; then
        echo "ERROR: ${CHECK%%:*} is missing from ${TERRA_SRC}/${CHECK#*:}." >&2
        exit 1
    fi
done
check_env "${BUILD_PATH}"
echo "${CODE_VERSION}" > "${BUILD_PATH}/CODE_VERSION"
if [ -e "${CODE_PATH}" ]; then
    echo "ERROR: ${CODE_PATH} appeared during the setup; not replacing it." >&2
    exit 1
fi
mv "${BUILD_PATH}" "${CODE_PATH}"
echo "Pinned TERRA code at ${CODE_PATH}: $(cat "${CODE_PATH}/CODE_VERSION")"
