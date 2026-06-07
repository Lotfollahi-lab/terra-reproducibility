#!/usr/bin/env bash
# Creates the banksy virtual environment and installs all dependencies.
# Script location: /nfs/team361/sb75/stjepa-reproducibility/envs/banksy.sh

set -euo pipefail

VENV_PATH="/nfs/team361/sb75/.venvs/banksy"

echo "Creating virtual environment at ${VENV_PATH} ..."
uv venv "${VENV_PATH}" --python 3.12

echo "Activating virtual environment ..."
source "${VENV_PATH}/bin/activate"

echo "Installing pyBANKSY ..."
uv pip install pybanksy

echo "Installing squidpy ..."
uv pip install squidpy

echo "Installing dask ..."
uv pip install "dask[array]"

echo "Installing harmonypy ..."
uv pip install harmonypy

echo "Installing PyTorch with CUDA 12.4 ..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "Installing scib_metrics ..."
uv pip install scib_metrics

echo "Done. Environment banksy is ready."
echo "To activate it in your shell, run:"
echo "  source ${VENV_PATH}/bin/activate"