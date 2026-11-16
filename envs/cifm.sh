#!/usr/bin/env bash
# Creates the CIFM virtual environment and downloads the CIFM model snapshot.
# Used by analysis/benchmarking/human_niche_benchmark/run_cifm.py (PYTHON_CIFM in
# its config.sh). Run on a login node: compute nodes have no internet, so the
# model is loaded from the local snapshot at run time.
#
# CIFM: https://huggingface.co/ynyou/CIFM (You et al., bioRxiv 2025,
# 10.1101/2025.01.25.634867). The authors used torch 2.1 + cu118; torch 2.6 +
# cu124 is used here because it is the build that runs on the farm GPUs (as in
# envs/nichecompass.sh). numpy, anndata and scanpy match the benchmark's scoring
# environment, so the h5ad files written here open there.

set -euo pipefail

VENV_PATH="${CIFM_VENV_PATH:-/nfs/team361/jr43/.venvs/cifm}"
MODEL_PATH="${CIFM_MODEL_PATH:-/lustre/scratch126/cellgen/lotfollahi/jr43/models/CIFM}"
CIFM_REVISION="19c507f00b354c7dc202d850292c268b06c2ec50"   # ynyou/CIFM, 2025-02-12

echo "Creating virtual environment at ${VENV_PATH} ..."
uv venv "${VENV_PATH}" --python 3.12

echo "Activating virtual environment ..."
source "${VENV_PATH}/bin/activate"

echo "Installing PyTorch with CUDA 12.4 ..."
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "Installing PyG libraries ..."
uv pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
uv pip install torch_geometric==2.7.0

echo "Installing the remaining dependencies ..."
uv pip install numpy==2.4.4 anndata==0.12.10 scanpy==1.11.5 \
    huggingface_hub safetensors

echo "Downloading CIFM (revision ${CIFM_REVISION}) to ${MODEL_PATH} ..."
python - "$MODEL_PATH" "$CIFM_REVISION" <<'EOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download('ynyou/CIFM', revision=sys.argv[2], local_dir=sys.argv[1])
print('Model snapshot at', path)
EOF

echo "Done. Environment cifm is ready."
echo "To activate it in your shell, run:"
echo "  source ${VENV_PATH}/bin/activate"
