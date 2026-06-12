#!/usr/bin/env bash
# Creates the cellcharter virtual environment and installs all dependencies.
# Script location: /nfs/team361/sb75/stjepa-reproducibility/envs/cellcharter.sh

set -euo pipefail

VENV_PATH="/nfs/team361/sb75/.venvs/cellcharter"

echo "Creating virtual environment at ${VENV_PATH} ..."
uv venv "${VENV_PATH}" --python 3.12

echo "Activating virtual environment ..."
source "${VENV_PATH}/bin/activate"

echo "Installing scvi-tools ..."
uv pip install scvi-tools

echo "Installing CellCharter ..."
uv pip install cellcharter

echo "Installing igraph ..."
uv pip install igraph

echo "Installing PyTorch with CUDA 12.4 ..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "Installing scib_metrics ..."
uv pip install scib_metrics

echo "Done. Environment cellcharter is ready."
echo "To activate it in your shell, run:"
echo "  source ${VENV_PATH}/bin/activate"