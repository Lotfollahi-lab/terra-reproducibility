#!/usr/bin/env bash
# Creates the HEIST virtual environment, clones the HEIST code and downloads the
# HEIST model snapshot. Used by analysis/benchmarking/human_niche_benchmark/run_heist.py
# (PYTHON_HEIST in its config.sh). Run on a login node: compute nodes have no
# internet, so code and model are loaded from the local copies at run time.
#
# HEIST (Madhu et al., ICLR 2026, arXiv 2506.11152): code at
# https://github.com/Graph-and-Geometric-Learning/HEIST, weights at
# https://huggingface.co/HirenMadhu/HEIST. Only the nonmarker_all checkpoint (the
# model card's default) and the shared gene vocabulary are downloaded.
#
# Versions: the authors' uv.lock pins torch 2.11 with CUDA 13, which needs a newer
# GPU driver than the farm nodes have, so torch 2.9.1 (pyproject.toml: >= 2.9.1)
# with CUDA 12.6 is used here. The lock does not cover the inference path
# (scanpy, magic-impute, geovoronoi, igraph); those are added below. numpy,
# pandas, anndata and scanpy match the benchmark's scoring environment, so the
# h5ad files written here open there. So do igraph, pynndescent, umap-learn and
# numba, which HEIST's internal kNN + Leiden use: pynndescent 0.7 returns
# scipy sparse arrays that scanpy 1.11.5's sc.pp.neighbors cannot handle
# (.getnnz). MAGIC's dependencies (graphtools, scprep, pygsp) compute the gene
# inputs, so they are pinned to the versions the benchmark runs used; so are
# geovoronoi and shapely (run_heist.py no longer builds Voronoi polygons, but the
# authors' code imports them) and the loaders. The other pins follow uv.lock.

set -euo pipefail

VENV_PATH="${HEIST_VENV_PATH:-/nfs/team361/jr43/.venvs/heist}"
CODE_PATH="${HEIST_CODE_PATH:-/lustre/scratch126/cellgen/lotfollahi/jr43/models/HEIST_code}"
MODEL_PATH="${HEIST_MODEL_PATH:-/lustre/scratch126/cellgen/lotfollahi/jr43/models/HEIST}"
CODE_REVISION="3679a981d8edebc3d0cc25f20ea929e8a1d5b933"    # GitHub main, 2026-10-08
MODEL_REVISION="b3cfb2a82f22fe051dd5b7cabdbf13a39bebf621"   # HF main, 2026-11-02

echo "Creating virtual environment at ${VENV_PATH} ..."
uv venv "${VENV_PATH}" --python 3.12

echo "Activating virtual environment ..."
source "${VENV_PATH}/bin/activate"

echo "Installing PyTorch with CUDA 12.6 ..."
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126

echo "Installing the remaining dependencies ..."
uv pip install torch_geometric==2.7.0 pymetis==2025.2.2 networkx==3.6.1 \
    scipy==1.17.1 scikit-learn==1.8.0 matplotlib==3.10.9 torchinfo==1.8.0 \
    pyyaml==6.0.3 tqdm==4.70.1 \
    numpy==2.4.4 pandas==2.3.3 anndata==0.12.10 scanpy==1.11.5 \
    magic-impute==3.0.0 graphtools==2.1.0 scprep==1.1.0 pygsp==0.6.1 \
    geovoronoi==0.4.0 shapely==2.2.0 igraph==1.0.0 \
    pynndescent==0.6.0 umap-learn==0.5.11 numba==0.65.0 \
    huggingface_hub==2.4.0 safetensors==0.8.0

echo "Cloning HEIST (commit ${CODE_REVISION}) to ${CODE_PATH} ..."
if [ ! -d "${CODE_PATH}/.git" ]; then
    git clone https://github.com/Graph-and-Geometric-Learning/HEIST "${CODE_PATH}"
fi
git -C "${CODE_PATH}" fetch origin
git -C "${CODE_PATH}" checkout --detach "${CODE_REVISION}"

echo "Downloading HEIST nonmarker_all (revision ${MODEL_REVISION}) to ${MODEL_PATH} ..."
python - "$MODEL_PATH" "$MODEL_REVISION" <<'EOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download('HirenMadhu/HEIST', revision=sys.argv[2], local_dir=sys.argv[1],
                         allow_patterns=['nonmarker_all/*', 'gene_vocab.json', 'README.md'])
print('Model snapshot at', path)
EOF

echo "Done. Environment heist is ready."
echo "To activate it in your shell, run:"
echo "  source ${VENV_PATH}/bin/activate"
