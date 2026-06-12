#!/usr/bin/env bash
# Creates the CellPLM uv virtual environment and installs everything
# needed to run `cellplm_mouse_benchmarking.ipynb`.
#
# Script location (cluster): /nfs/team361/sb75/terra-reproducibility/envs/cellplm.sh
#
# CellPLM is the binding constraint on Python / torch versions:
#   - CellPLM supports Python 3.9-3.10. We use 3.10 because the notebook's
#     other deps (squidpy >=1.4, scib_metrics >=0.5) require >=3.10.
#   - CellPLM does NOT pin a CUDA build of torch -- you install torch for
#     YOUR CUDA driver. Below defaults to cu121 (works with CUDA 12.x
#     drivers, incl. 12.4). For CUDA 11.x use cu118; for CPU-only use the
#     plain `uv pip install torch`.
#
# NOTE: Harmony batch correction was removed from the notebook, so
# `harmonypy` is intentionally NOT installed.

set -euo pipefail

VENV_PATH="/nfs/team361/sb75/.venvs/cellplm"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"   # <- change for your CUDA

echo "Creating virtual environment at ${VENV_PATH} (Python 3.10) ..."
uv venv "${VENV_PATH}" --python 3.10

echo "Activating virtual environment ..."
source "${VENV_PATH}/bin/activate"

# uv venvs are minimal and do NOT ship setuptools/wheel. CellPLM (and a
# number of older scientific packages) still `import pkg_resources`,
# which lives in setuptools -- without it you get
# `ModuleNotFoundError: No module named 'pkg_resources'`. Install it
# FIRST so subsequent builds/imports have it. Pin <81 because newer
# setuptools is removing pkg_resources.
echo "Installing setuptools (<81, provides pkg_resources) + wheel ..."
uv pip install "setuptools<81" wheel

echo "Installing torch (CUDA build: ${TORCH_INDEX}) ..."
uv pip install torch --index-url "${TORCH_INDEX}"

echo "Installing CellPLM ..."
# CellPLM is pip-installable; the import name is `CellPLM`. If PyPI
# resolution conflicts with the science stack, instead clone the repo
# (git clone https://github.com/OmicsML/CellPLM.git) and run
# `uv pip install -e ./CellPLM`.
uv pip install cellplm

echo "Installing single-cell / spatial science stack ..."
# scanpy pulls in numpy, scipy, pandas, scikit-learn, anndata, umap-learn,
# matplotlib, leidenalg. We list the clustering / spatial / metric extras
# the notebook imports explicitly.
uv pip install \
    "scanpy" \
    "squidpy" \
    "scib-metrics" \
    "anndata" \
    "matplotlib" \
    "scikit-learn" \
    "igraph" \
    "leidenalg" \
    "umap-learn"

echo "Installing gene-mapping deps (mouse -> human orthologs) ..."
# mygene = the 4-step ortholog pipeline; requests = Ensembl REST homology.
uv pip install mygene requests

echo "Installing Jupyter ..."
uv pip install jupyterlab ipykernel

# Register THIS env as a Jupyter kernel so the notebook actually runs in
# it. Without this, a notebook opened from some other Jupyter install
# would run in the wrong interpreter and you'd hit ModuleNotFoundError
# (pkg_resources / CellPLM / ...) even though everything is installed here.
echo "Registering Jupyter kernel 'cellplm' ..."
python -m ipykernel install --user --name cellplm --display-name "Python (cellplm)"

echo
echo "Done. Environment 'cellplm' is ready."
echo "To activate it in your shell, run:"
echo "  source ${VENV_PATH}/bin/activate"
echo
echo "In Jupyter, select the kernel 'Python (cellplm)' (Kernel -> Change"
echo "Kernel) and restart it. Verify with:  import sys; print(sys.executable)"
echo "-> should print ${VENV_PATH}/bin/python"
echo
echo "Optional (NOT installed): rapids-singlecell for GPU-accelerated"
echo "Leiden in the resolution sweep. The notebook falls back to scanpy's"
echo "CPU Leiden automatically if it is absent, so this is not required."
