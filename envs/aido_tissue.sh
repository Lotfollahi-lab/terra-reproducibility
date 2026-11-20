#!/usr/bin/env bash
# Creates the AIDO.Tissue virtual environment, clones the authors' code and downloads
# the checkpoints, the authors' embedding demo and the gene map. Used by
# analysis/benchmarking/human_niche_benchmark/run_aido_tissue.py (PYTHON_AIDO_TISSUE in
# its config.sh). Run on a login node: compute nodes have no internet, so code, model
# and gene map are loaded from the local copies at run time.
#
# AIDO.Tissue (Gong et al., bioRxiv 2025, 10.1101/2025.07.04.663102), renamed GB.Tissue
# by GenBio AI in July 2026: code in https://github.com/genbio-ai/ModelGenerator
# (modelgenerator.backbones.gb_tissue_60m), weights in genbio-ai/GB.Tissue-60M (the
# benchmark model) and genbio-ai/GB.Tissue-3M (the checkpoint of the authors' embedding
# config, used only by checks/test_aido_tissue.py).
#
# Versions: torch 2.6.0 + cu124 and flash_attn 2.7.4.post1 are ModelGenerator's
# `flash_attn` extra (the model has no attention without flash-attn), transformers 4.38.0
# and numpy < 2 are its pins. ModelGenerator itself is installed without dependencies,
# plus the packages its backbone, data, task and CLI modules import: its openfold,
# dllogger and tiledbsoma pins are used only by protein-structure and SOMA code.
# scanpy and pandas match the benchmark's scoring environment, so the h5ad files written
# here open there.
#
# Gene map: GB.Tissue's gene list holds HGNC symbols. The authors' GB.Cell data modules
# (same 19,264 genes) map them to Ensembl IDs with bionty (modelgenerator.cell.utils.
# build_map). The map is computed here once and saved; run_aido_tissue.py pins its sha256
# and uses it for panel genes whose symbol is not in the list (e.g. the old Visium HD
# names H3F3A -> H3-3A).

set -euo pipefail

VENV_PATH="${AIDO_TISSUE_VENV_PATH:-/nfs/team361/jr43/.venvs/aido_tissue}"
MODEL_ROOT="${AIDO_TISSUE_MODEL_ROOT:-/lustre/scratch126/cellgen/lotfollahi/jr43/models}"
CODE_PATH="${MODEL_ROOT}/ModelGenerator_code"
CODE_REVISION="bda6f2da297ba886e3662b49eea9928feacbfd98"     # GitHub main, 2026-08-19
REVISION_60M="e414516407edb487a0df07ad33ab031ade3f95b6"      # genbio-ai/GB.Tissue-60M
REVISION_3M="7ac498c8fb452844e33b592418e6dc468cb8c5c4"       # genbio-ai/GB.Tissue-3M
REVISION_DATA="a1d14f5ce1faaf83bcbe2eca69c12a7d5e74259b"     # genbio-ai/tissue-downstream-tasks
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

echo "Creating virtual environment at ${VENV_PATH} ..."
uv venv "${VENV_PATH}" --python 3.12

echo "Activating virtual environment ..."
source "${VENV_PATH}/bin/activate"

echo "Installing PyTorch with CUDA 12.4 and flash-attn ..."
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install "${FLASH_ATTN_WHEEL}"

echo "Installing the remaining dependencies ..."
# bionty 1.3.2 (ModelGenerator's pin; modelgenerator.data imports it) needs lamindb 1.10,
# which caps anndata at 0.12.1; files it writes open with the scoring env's 0.12.10.
uv pip install transformers==4.38.0 numpy==1.26.4 bionty==1.3.2 lamindb==1.10.2 \
    anndata==0.12.1 scanpy==1.11.5 pandas==2.3.3 scipy==1.14.1 scikit-learn==1.9.1 \
    lightning==2.6.6 torchmetrics==1.9.0 peft==0.13.2 'jsonargparse[signatures]==4.53.0' \
    docstring-inheritance==2.2.1 datasets==4.0.0 pyfaidx==0.9.2.2 lmdb==3.0.0 \
    einops==0.8.2 omegaconf==2.4.0 huggingface_hub==0.36.2 safetensors==0.8.0 \
    pyarrow==20.0.0   # bionty reads its gene tables from parquet; >= 21 needs numpy 2

echo "Cloning ModelGenerator (commit ${CODE_REVISION}) to ${CODE_PATH} ..."
if [ ! -d "${CODE_PATH}/.git" ]; then
    git clone https://github.com/genbio-ai/ModelGenerator "${CODE_PATH}"
fi
git -C "${CODE_PATH}" fetch origin
git -C "${CODE_PATH}" checkout --detach "${CODE_REVISION}"

echo "Installing ModelGenerator (no dependencies) ..."
# Built from a copy: an in-tree build would write into the checkout (run_aido_tissue.py
# requires a clean one).
BUILD_DIR=$(mktemp -d)
git -C "${CODE_PATH}" archive "${CODE_REVISION}" | tar -x -C "${BUILD_DIR}"
uv pip install --no-deps "${BUILD_DIR}"
rm -rf "${BUILD_DIR}"

echo "Downloading the checkpoints and the authors' embedding demo to ${MODEL_ROOT} ..."
python - "$MODEL_ROOT" "$REVISION_60M" "$REVISION_3M" "$REVISION_DATA" <<'EOF'
import sys
from huggingface_hub import snapshot_download
root, rev60, rev3, revdata = sys.argv[1:]
for repo, rev in [('GB.Tissue-60M', rev60), ('GB.Tissue-3M', rev3)]:
    print(snapshot_download(f'genbio-ai/{repo}', revision=rev, local_dir=f'{root}/{repo}'))
print(snapshot_download('genbio-ai/tissue-downstream-tasks', repo_type='dataset',
                        revision=revdata, local_dir=f'{root}/GB.Tissue_data',
                        allow_patterns=['README.md', 'scRNA_genename_and_index.tsv',
                                        'processed_fetal_lung_visium_xenium.xenium.convert.h5ad']))
EOF

echo "Mapping the gene list to Ensembl IDs with the authors' build_map (bionty) ..."
mkdir -p "${MODEL_ROOT}/GB.Tissue_gene_map"
python - "${MODEL_ROOT}/GB.Tissue_gene_map/gene_list_ensembl.tsv" <<'EOF'
import sys
from importlib.resources import files
import importlib.metadata
import pandas as pd
from modelgenerator.cell.utils import build_map
genes = pd.read_csv(files('modelgenerator.cell') / 'gene_lists' / 'OS_scRNA_gene_index.19264.tsv',
                    sep='\t')['gene_name'].astype(str).values
mapping = build_map(genes)
out = pd.DataFrame({'gene_name': genes, 'ensembl_id': [mapping.get(g, '') for g in genes]})
out.to_csv(sys.argv[1], sep='\t', index=False)
print(f'{(out.ensembl_id != "").sum()} of {len(out)} genes mapped '
      f'(bionty {importlib.metadata.version("bionty")}) -> {sys.argv[1]}')
EOF
echo "sha256 of the gene map (pinned in run_aido_tissue.py):"
sha256sum "${MODEL_ROOT}/GB.Tissue_gene_map/gene_list_ensembl.tsv"

echo "Done. Environment aido_tissue is ready."
echo "To activate it in your shell, run:"
echo "  source ${VENV_PATH}/bin/activate"
