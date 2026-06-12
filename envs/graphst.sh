#!/usr/bin/env bash
# Creates the graphst virtual environment and installs all dependencies.
# Script location: /nfs/team361/sb75/stjepa-reproducibility/envs/graphst.sh

set -euo pipefail

VENV_PATH="/nfs/team361/sb75/.venvs/graphst"

echo "Creating virtual environment at ${VENV_PATH} ..."
uv venv "${VENV_PATH}" --python 3.8

echo "Activating virtual environment ..."
source "${VENV_PATH}/bin/activate"

echo "Installing GraphST ..."
uv pip install graphst

echo "Installing PyTorch with CUDA 12.4 ..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "Installing squidpy ..."
uv pip install squidpy

echo "Installing POT ..."
uv pip install "POT==0.9.0"

echo "Installing scikit-misc ..."
uv pip install scikit-misc

echo "Installing scib_metrics ..."
uv pip install scib_metrics

echo "Installing paste-bio ..."
uv pip install paste-bio

echo "Done. Environment graphst is ready."
echo "To activate it in your shell, run:"
echo "  source ${VENV_PATH}/bin/activate"