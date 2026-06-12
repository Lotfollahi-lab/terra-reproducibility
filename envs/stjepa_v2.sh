#!/usr/bin/env bash
# Creates the stjepa_v2 virtual environment and installs the stjepa package.
# Script location: /nfs/team361/sb75/stjepa-reproducibility/envs/stjepa_v2.sh

set -euo pipefail

VENV_PATH="/nfs/team361/sb75/.venvs/stjepa_v2"
PACKAGE_PATH="/nfs/team361/sb75/stjepa"

echo "Creating virtual environment at ${VENV_PATH} ..."
uv venv "${VENV_PATH}" --python 3.10

echo "Activating virtual environment ..."
source "${VENV_PATH}/bin/activate"

echo "Installing package from ${PACKAGE_PATH} ..."
uv pip install -e "${PACKAGE_PATH}[eval]"

echo "Done. Environment stjepa_v2 is ready."
echo "To activate it in your shell, run:"
echo "  source ${VENV_PATH}/bin/activate"