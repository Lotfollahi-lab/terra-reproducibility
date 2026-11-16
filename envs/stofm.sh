#!/usr/bin/env bash
# Creates the SToFM virtual environment, clones the SToFM code and downloads the
# SToFM checkpoint and demo data. Used by
# analysis/benchmarking/human_niche_benchmark/run_stofm.py (PYTHON_STOFM in its
# config.sh). Run on a login node: compute nodes have no internet, so code and
# model are loaded from the local copies at run time.
#
# SToFM (Zhao et al., ICML 2025, arXiv 2507.11588): code at
# https://github.com/PharMolix/SToFM. The weights are only on Google Drive (README),
# without versions: run_stofm.py pins the sha256 of every checkpoint file instead.
#
# Versions: the authors' requirements.txt (python 3.9, torch 2.0.1, scanpy 1.9.3,
# rapids_singlecell 0.10.10) is not self-consistent: model/extraction.py calls
# sc.tl.leiden(flavor='igraph'), which needs scanpy >= 1.10. Here torch 2.6 + cu124
# (runs on the farm and tiger GPUs, as in envs/cifm.sh) and transformers 4.46.2 (the
# version that saved the cell encoder) are used. rapids_singlecell and cupy are not
# installed: run_stofm.py stubs them and computes the exact kNN of the authors'
# default GPU path on the CPU. numpy, pandas, anndata, scanpy, scikit-learn, igraph,
# pynndescent, umap-learn and numba match the benchmark's scoring environment (SToFM's
# virtual cells come from scanpy's kNN + Leiden). geneformer 0.0.1 is installed from
# the copy bundled in the SToFM repo (no longer on PyPI), with what it imports.

set -euo pipefail

VENV_PATH="${STOFM_VENV_PATH:-/nfs/team361/jr43/.venvs/stofm}"
CODE_PATH="${STOFM_CODE_PATH:-/lustre/scratch126/cellgen/lotfollahi/jr43/models/SToFM_code}"
MODEL_PATH="${STOFM_MODEL_PATH:-/lustre/scratch126/cellgen/lotfollahi/jr43/models/SToFM}"
CODE_REVISION="2354d5799347867578793752e8c2dd93ae6587b7"   # GitHub main, 2025-09-02
DRIVE_FOLDER="https://drive.google.com/drive/folders/1mHE8gf8MAPwzZoEB0vwOOfQ4lz3H_-xo"

echo "Creating virtual environment at ${VENV_PATH} ..."
uv venv "${VENV_PATH}" --python 3.12

echo "Activating virtual environment ..."
source "${VENV_PATH}/bin/activate"

echo "Installing PyTorch with CUDA 12.4 ..."
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "Installing the remaining dependencies ..."
uv pip install transformers==4.46.2 safetensors \
    numpy==2.4.4 pandas==2.3.3 anndata==0.12.10 scanpy==1.11.5 \
    scipy==1.17.1 scikit-learn==1.8.0 igraph==1.0.0 \
    pynndescent==0.6.0 umap-learn==0.5.11 numba==0.65.0 \
    datasets loompy seaborn matplotlib tqdm pytz gdown

echo "Cloning SToFM (commit ${CODE_REVISION}) to ${CODE_PATH} ..."
if [ ! -d "${CODE_PATH}/.git" ]; then
    git clone https://github.com/PharMolix/SToFM "${CODE_PATH}"
fi
git -C "${CODE_PATH}" fetch origin
git -C "${CODE_PATH}" checkout --detach "${CODE_REVISION}"

echo "Installing the bundled geneformer 0.0.1 ..."
# Built from a copy: an in-tree build would write build/ and *.egg-info into the
# checkout (run_stofm.py requires a clean one).
BUILD_DIR=$(mktemp -d)
cp -r "${CODE_PATH}/geneformer_001" "${BUILD_DIR}/"
rm -rf "${BUILD_DIR}/geneformer_001/build" "${BUILD_DIR}/geneformer_001/"*.egg-info
uv pip install --no-deps "${BUILD_DIR}/geneformer_001"
rm -rf "${BUILD_DIR}"

echo "Downloading the SToFM checkpoint and demo data to ${MODEL_PATH} ..."
mkdir -p "${MODEL_PATH}"
gdown --folder "${DRIVE_FOLDER}" -O "${MODEL_PATH}"
echo "sha256 of the downloaded files (pinned in run_stofm.py):"
(cd "${MODEL_PATH}" && find . -type f -not -path '*/.*' | sort | xargs sha256sum)

echo "Done. Environment stofm is ready."
echo "To activate it in your shell, run:"
echo "  source ${VENV_PATH}/bin/activate"
