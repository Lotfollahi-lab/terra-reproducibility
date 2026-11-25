#!/bin/bash
# One-time setup of the pinned TERRA code of E0: a plain copy (git archive) of Lotfollahi-lab/terra at COMMIT
# (exp/effective-exposures) at CODE_PATH, both from the env file. The E0 jobs put its src/ first on PYTHONPATH, so
# later changes of the farm checkout (the venv's editable install) cannot change running jobs. The commit is taken
# from REPO_PATH if that checkout has it (`git -C "${REPO_PATH}" fetch origin exp/effective-exposures` fetches it
# without touching the checkout), otherwise fetched from TERRA_URL into a temporary repository. The copy is built next
# to CODE_PATH, checked (it has the E0 options; ENV_PATH imports it), and moved to CODE_PATH with CODE_VERSION. If
# CODE_PATH is up to date, only the environment check is rerun; if it holds another version, the script stops (jobs
# may be running on it): move it away and rerun. As analysis/ablation/ablation_pipeline/setup_code.sh.
#
# Usage: bash analysis/selection_e0/setup_code.sh analysis/selection_e0/e0.env
# Optional: REPO_PATH=<checkout> TERRA_URL=<git url> bash ...

set -euo pipefail

if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    echo "Usage: bash setup_code.sh ENV_FILE  (e.g. analysis/selection_e0/e0.env)" >&2
    exit 1
fi
source "$1"
REPO_PATH="${REPO_PATH:-/nfs/team361/sb75/terra}"
TERRA_URL="${TERRA_URL:-https://github.com/Lotfollahi-lab/terra.git}"
CODE_VERSION="terra ${COMMIT}"
if [[ "${CODE_PATH}" != /* ]]; then
    echo "ERROR: CODE_PATH must be an absolute path, not ${CODE_PATH}." >&2
    exit 1
fi

check_env() {
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$1/src" "${ENV_PATH}/bin/python" -c "
import terra, terra.training.exposures, terra.tokenizers.cell_tokenizers, terra.training.train
import datasets, pyarrow, torch, wandb, yaml
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
# The options E0 needs
TERRA_SRC="${BUILD_PATH}/src/terra"
for CHECK in "add_spatial_index:tokenizers/cell_tokenizers.py" "epoch_index_lists:training/train.py" \
             "spread_repeats:datasets/dataloaders.py" "class ExposureTracker:training/exposures.py" \
             "def check_neighbor_rows:training/exposures.py"; do
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
