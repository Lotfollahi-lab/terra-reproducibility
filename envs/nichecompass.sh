#!/usr/bin/env bash
# Creates the nichecompass virtual environment and installs all dependencies.
# Script location: /nfs/team361/sb75/stjepa-reproducibility/envs/nichecompass.sh

set -euo pipefail

VENV_PATH="/nfs/team361/sb75/.venvs/nichecompass"

echo "Creating virtual environment at ${VENV_PATH} ..."
uv venv "${VENV_PATH}" --python 3.12

echo "Activating virtual environment ..."
source "${VENV_PATH}/bin/activate"

echo "Installing PyTorch with CUDA 12.4 ..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "Installing PyG libraries ..."
uv pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

echo "Installing NicheCompass ..."
uv pip install nichecompass[all]

echo "Done. Environment nichecompass is ready."
echo "To activate it in your shell, run:"
echo "  source ${VENV_PATH}/bin/activate"