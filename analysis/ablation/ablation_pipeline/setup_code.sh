#!/bin/bash
# One-time setup of the pinned ST-JEPA v2 code of the ablations: a plain copy (git archive) of sebastianbirk/stjepa at
# b433d5e, the v2 commit the April mouse ablation runs used (later v2 commits only add tokenizer debug prints and
# configs), with stjepa_ablation.patch applied (see its header: opt-in seed, atomic latest checkpoint, metatoken fixes,
# opt-in tokenizer fixes). CODE_PATH and ENV_PATH are taken from the env file.
# The jobs put its src/ first on PYTHONPATH, so later changes of the farm checkout (editable install of the venv)
# cannot change running jobs. The commit is taken from REPO_PATH if that checkout has it, otherwise fetched from
# STJEPA_URL into a temporary repository (REPO_PATH is not modified). The copy is built next to CODE_PATH and checked
# (patch applied; ENV_PATH imports it) before CODE_VERSION (commit and patch checksum) is written and it is moved to
# CODE_PATH; generate_ablation_runs.py records CODE_VERSION in every run config. If CODE_PATH is up to date, only the
# environment check is rerun. If it exists with another version, the script stops (jobs may be running on it): move it
# away and rerun.
#
# Usage: bash setup_code.sh ENV_FILE          (e.g. datasets/ovca.env)
# Optional: REPO_PATH=<checkout> STJEPA_URL=<git url> bash setup_code.sh ENV_FILE

set -euo pipefail

if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
    echo "Usage: bash setup_code.sh ENV_FILE  (e.g. datasets/ovca.env)" >&2
    exit 1
fi
source "$1"
REPO_PATH="${REPO_PATH:-/nfs/team361/sb75/terra}"  # formerly /nfs/team361/sb75/stjepa
STJEPA_URL="${STJEPA_URL:-https://github.com/sebastianbirk/stjepa.git}"
COMMIT="b433d5ee7f9b341b2da13b19d7b5ac33de258994"
PATCH_FILE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/stjepa_ablation.patch"
CODE_VERSION="${COMMIT} + stjepa_ablation.patch (sha256 $(sha256sum "${PATCH_FILE_PATH}" | cut -c1-12))"
if [[ "${CODE_PATH}" != /* ]]; then
    echo "ERROR: CODE_PATH must be an absolute path, not ${CODE_PATH}." >&2
    exit 1
fi

# The jobs run the pinned code in ENV_PATH (fix ENV_PATH in the env file if this fails); scib_metrics (with pynndescent
# and jax) computes the batch-integration metrics of datasets with two or more sections
check_env() {
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$1/src" "${ENV_PATH}/bin/python" -c "
import app.utils, stjepa, datasets, igraph, scanpy, scib_metrics, squidpy, torch, wandb, yaml
assert stjepa.__file__.startswith('$1/'), stjepa.__file__
print('Environment ${ENV_PATH} imports the pinned code at $1 (torch', torch.__version__ + ').')"
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
    echo "${COMMIT} is not in ${REPO_PATH}; fetching it from ${STJEPA_URL}..."
    git init -q --bare "${SOURCE_REPO_PATH}"
    git -C "${SOURCE_REPO_PATH}" fetch -q --depth 1 "${STJEPA_URL}" "${COMMIT}"
fi
BUILD_PATH="$(mktemp -d "${CODE_PATH}.setup.XXXXXX")"
TEMPORARY_PATHS+=("${BUILD_PATH}")
git -C "${SOURCE_REPO_PATH}" archive "${COMMIT}" | tar -x -C "${BUILD_PATH}"
# Inside an enclosing repository, git apply resolves the patch paths against that repository's root and silently skips
# files outside the current folder; the ceiling keeps it from looking above CODE_PATH's folder
(cd "${BUILD_PATH}" && GIT_CEILING_DIRECTORIES="$(dirname "${CODE_PATH}")" git apply "${PATCH_FILE_PATH}")
if ! grep -q pad_masks_to_batch_max "${BUILD_PATH}/src/stjepa/masks/block_masking.py" \
        || ! grep -q keep_special_tokens "${BUILD_PATH}/src/app/inference/embed.py"; then
    echo "ERROR: ${PATCH_FILE_PATH} did not apply (pad_masks_to_batch_max or keep_special_tokens missing)." >&2
    exit 1
fi
check_env "${BUILD_PATH}"
echo "${CODE_VERSION}" > "${BUILD_PATH}/CODE_VERSION"
if [ -e "${CODE_PATH}" ]; then
    echo "ERROR: ${CODE_PATH} appeared during the setup; not replacing it." >&2
    exit 1
fi
mv "${BUILD_PATH}" "${CODE_PATH}"
echo "Pinned code at ${CODE_PATH}: $(cat "${CODE_PATH}/CODE_VERSION")"
