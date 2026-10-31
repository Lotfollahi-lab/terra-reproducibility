#!/bin/bash
# One-time setup of the pinned ST-JEPA v2 code for the OVCA ablations: a plain copy (git archive) of
# sebastianbirk/stjepa at b433d5e, the v2 commit the April mouse ablation runs used (later v2 commits only add
# tokenizer debug prints and configs), with stjepa_ovca_ablation.patch applied (see its header: opt-in seed, atomic
# latest checkpoint, metatoken fixes, opt-in tokenizer fixes).
# The jobs put its src/ first on PYTHONPATH, so later changes of the farm checkout (editable install of the venv)
# cannot change running jobs. The commit is taken from REPO_PATH if that checkout has it, otherwise fetched from
# STJEPA_URL into a temporary repository (REPO_PATH is not modified). Finally checks that ENV_PATH can import the
# pinned code. Run from this directory. CODE_VERSION records the patch checksum; if CODE_PATH exists with another
# patch version, the script stops (jobs may be running on it): move it away and rerun.
#
# Usage: bash setup_code.sh
# Optional: REPO_PATH=<checkout> STJEPA_URL=<git url> ENV_PATH=<venv> bash setup_code.sh

set -euo pipefail

REPO_PATH="${REPO_PATH:-/nfs/team361/sb75/terra}"  # formerly /nfs/team361/sb75/stjepa
STJEPA_URL="${STJEPA_URL:-https://github.com/sebastianbirk/stjepa.git}"
CODE_PATH="${CODE_PATH:-/nfs/team361/sb75/stjepa_ovca_ablation}"
ENV_PATH="${ENV_PATH:-/nfs/team361/sb75/.venvs/stjepa_v2}"
COMMIT="b433d5ee7f9b341b2da13b19d7b5ac33de258994"
PATCH_FILE_PATH="$(pwd)/stjepa_ovca_ablation.patch"
CODE_VERSION="${COMMIT} + stjepa_ovca_ablation.patch (sha256 $(sha256sum "${PATCH_FILE_PATH}" | cut -c1-12))"

if [ -e "${CODE_PATH}" ]; then
    if [ "$(cat "${CODE_PATH}/CODE_VERSION" 2>/dev/null)" == "${CODE_VERSION}" ]; then
        echo "${CODE_PATH} is up to date: ${CODE_VERSION}"
        exit 0
    fi
    echo "${CODE_PATH} exists with another version ($(cat "${CODE_PATH}/CODE_VERSION" 2>/dev/null || echo unknown))."
    echo "If no jobs use it: mv ${CODE_PATH} ${CODE_PATH}_old && bash setup_code.sh"
    exit 1
fi

if git -C "${REPO_PATH}" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
    SOURCE_REPO_PATH="${REPO_PATH}"
    echo "Using ${COMMIT} from ${REPO_PATH}."
else
    SOURCE_REPO_PATH="$(mktemp -d)"
    trap 'rm -rf "${SOURCE_REPO_PATH}"' EXIT
    echo "${COMMIT} is not in ${REPO_PATH}; fetching it from ${STJEPA_URL}..."
    git init -q --bare "${SOURCE_REPO_PATH}"
    git -C "${SOURCE_REPO_PATH}" fetch -q --depth 1 "${STJEPA_URL}" "${COMMIT}"
fi
mkdir -p "${CODE_PATH}"
git -C "${SOURCE_REPO_PATH}" archive "${COMMIT}" | tar -x -C "${CODE_PATH}"
(cd "${CODE_PATH}" && git apply "${PATCH_FILE_PATH}")
echo "${CODE_VERSION}" > "${CODE_PATH}/CODE_VERSION"
echo "Pinned code at ${CODE_PATH}: $(cat "${CODE_PATH}/CODE_VERSION")"

# The jobs run the pinned code in ENV_PATH (override ENV_PATH when submitting if this fails)
PYTHONPATH="${CODE_PATH}/src" "${ENV_PATH}/bin/python" -c "
import app.utils, stjepa, datasets, scanpy, squidpy, torch, wandb, yaml
assert stjepa.__file__.startswith('${CODE_PATH}/'), stjepa.__file__
print('Environment ${ENV_PATH} imports the pinned code (torch', torch.__version__ + ').')"
