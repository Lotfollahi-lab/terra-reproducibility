#!/usr/bin/env bash
# Creates the novae virtual environment and installs all dependencies.
# Script location: /nfs/team361/sb75/stjepa-reproducibility/envs/novae.sh

set -euo pipefail

VENV_PATH="/nfs/team361/sb75/.venvs/novae"

echo "Creating virtual environment at ${VENV_PATH} ..."
uv venv "${VENV_PATH}" --python 3.12

echo "Activating virtual environment ..."
source "${VENV_PATH}/bin/activate"

echo "Installing novae ..."
uv pip install novae

echo "Installing squidpy ..."
uv pip install squidpy

echo "Installing scib_metrics ..."
uv pip install scib_metrics

echo "Done. Environment novae is ready."
echo "To activate it in your shell, run:"
echo "  source ${VENV_PATH}/bin/activate"