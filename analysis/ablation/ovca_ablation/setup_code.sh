#!/bin/bash
# One-time setup of the pinned ST-JEPA v2 code for the OVCA ablations: a plain copy (git archive) of
# sebastianbirk/stjepa at b433d5e, the v2 commit the April mouse ablation runs used (later v2 commits only add
# tokenizer debug prints and configs), with stjepa_ovca_ablation.patch applied (opt-in seed, atomic latest checkpoint).
# The jobs put its src/ first on PYTHONPATH, so later branch switches of /nfs/team361/sb75/stjepa (editable install of
# the stjepa_v2 venv) cannot change running jobs. Run from this directory.
#
# Usage: bash setup_code.sh

set -euo pipefail

REPO_PATH="${REPO_PATH:-/nfs/team361/sb75/stjepa}"
CODE_PATH="${CODE_PATH:-/nfs/team361/sb75/stjepa_ovca_ablation}"
COMMIT="b433d5ee7f9b341b2da13b19d7b5ac33de258994"

if [ -e "${CODE_PATH}" ]; then
    echo "${CODE_PATH} already exists."
    exit 1
fi
git -C "${REPO_PATH}" fetch origin v2
mkdir -p "${CODE_PATH}"
git -C "${REPO_PATH}" archive "${COMMIT}" | tar -x -C "${CODE_PATH}"
(cd "${CODE_PATH}" && git apply "${OLDPWD}/stjepa_ovca_ablation.patch")
echo "${COMMIT} + stjepa_ovca_ablation.patch" > "${CODE_PATH}/CODE_VERSION"
echo "Pinned code at ${CODE_PATH}: $(cat "${CODE_PATH}/CODE_VERSION")"
