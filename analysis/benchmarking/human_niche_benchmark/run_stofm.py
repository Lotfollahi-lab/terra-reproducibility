"""SToFM zero-shot cell embeddings for the human niche benchmark.

SToFM (Zhao et al., ICML 2025, arXiv 2507.11588; code
https://github.com/PharMolix/SToFM, weights in the Google Drive folder linked from
its README) has two parts:
- a Geneformer-style cell encoder: a BERT with 12 layers and width 512; its CLS
  output is projected to 256-d;
- an SE(2) transformer over sub-slices of ~1000 cells, with distance-based
  attention biases. Whole-section "virtual cells" (Leiden clusters of coordinates
  and cell embeddings) are added to every sub-slice.

This script runs the authors' inference recipe (get_embeddings.py,
model/extraction.py, preprocessing/preprocess.py) with the pretrained weights and
no fine-tuning:
- cell-encoder input as in preprocess.py (Geneformer 0.0.1 tokenizer):
  - raw counts / the cell's total over all panel genes x 10,000 / Geneformer's
    gene medians;
  - the dictionary genes ranked by that value (rank-value encoding, at most
    2048);
  - <cls> prepended by the authors' CellEncoderCollator;
  - CLS output -> cell_proj -> L2 norm (the Pooler of get_embeddings.py);
- the authors' load_data on each section:
  - coordinates centred and divided by the section's bounding-box diagonal (x100);
  - sub-slices from a square grid (split_num 1000);
  - virtual cells from Leiden (resolution 1.0) on 0.2 xy + 0.8 PCA2(cell
    embedding);
  - attention bias from 50-NN distances;
- the SE(2) transformer on batches of 4 sub-slices. The embedding is the last
  hidden state of each cell token (256-d), as get_embeddings.py saves it.

Deviations. Each is needed because the released code crashes, silently corrupts
results or depends on the machine; checks/test_stofm.py compares each with the
authors' code:
- get_embeddings' encode_cell crashes with KeyError 'label': Geneformer's
  collator deletes a 'label' column that encode_cell has just removed. Each cell
  gets a dummy label 0 here, so the authors' CellEncoderCollator runs unchanged.
- preprocess.py densifies the whole cells x genes matrix (~75 GB for a Visium HD
  section). Its tokenize_anndata is re-implemented here on row chunks of the
  sparse matrix (identical tokens).
- extraction.py imports rapids_singlecell and cupy. On a GPU it computes the
  virtual-cell kNN graph exactly with rapids. If that fails it silently falls back
  to scanpy, which is approximate (pynndescent) for >= 8192 cells. Here rapids is
  not installed (stubs satisfy the import), load_data runs on its CPU path
  (device=-1), and scanpy computes that graph exactly too (brute force). Below
  8192 cells, which includes every sub-slice, scanpy already does so, so those
  graphs are unchanged.
- The SE(2) forward pass runs under torch.no_grad() (same output, less memory).
- Embeddings start as NaN instead of 0, so a cell missing from every sub-slice
  stops the run instead of getting a zero embedding.

Benchmark choices:
- one load_data call per section (the authors embed one slide per call), so
  sub-slices and virtual cells never cross sections;
- the authors' defaults everywhere: split_num 1000, Leiden resolution 1.0,
  alpha 0.2, batch sizes 32 (cell encoder) and 4 (SE(2));
- their script's seed (1), set before each section; deterministic torch
  algorithms;
- genes matched to Geneformer's dictionary by the silver Ensembl ID. IDs the
  dictionary lacks fall back to the gene symbol through Geneformer's
  gene_name_id_dict, e.g. the pseudo-autosomal CD99 and IL3RA, whose silver IDs
  are the chrY copies;
- the sub-slice split as released: a grid square with >= 2000 cells is cut into
  ~1000-cell chunks in obs order, not by position. run_info.json records, per
  section, how many cells this affects and how compact their chunks are.

The result h5ad keeps, for interpretation (not scored):
- obs['stofm_virtual_cell']: SToFM's whole-section Leiden clusters (no labels
  used);
- obs['stofm_subslice'];
- obsm['X_stofm_cell']: the cell-encoder embedding, without spatial context.

Writes the embedding only; job_wrapper.sh then clusters and scores it with
score.py. Run in the stofm environment (envs/stofm.sh) on a GPU:
    SCRIPT=run_stofm.py PYTHON_ENV=PYTHON_STOFM DATASET=xhs49-1b_1p \\
        bsub -G s10396 -q inference -n 8 -M 64GB \\
        -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]" \\
        -gpu "num=1:mode=exclusive_process" -W 12:00 < run_method.lsf
(README: settings per dataset.)
"""
import argparse
import hashlib
import importlib.metadata
import os
import pickle
import random
import shutil
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')   # compute nodes have no internet
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # deterministic cuBLAS
# The SToFM checkout tracks .pyc files: writing bytecode would make it dirty
# (code_revision).
sys.dont_write_bytecode = True

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import BertModel  # noqa: E402

# Local copies (envs/stofm.sh): the authors' code (GitHub) and checkpoint (Google Drive).
STOFM_CODE_PATH = Path(os.environ.get(
    'STOFM_CODE_PATH', '/lustre/scratch126/cellgen/lotfollahi/jr43/models/SToFM_code'))
STOFM_MODEL_PATH = Path(os.environ.get(
    'STOFM_MODEL_PATH', '/lustre/scratch126/cellgen/lotfollahi/jr43/models/SToFM'))
CODE_REVISION = '2354d5799347867578793752e8c2dd93ae6587b7'
# Google Drive keeps no versions, so the checkpoint is pinned by the sha256 of each file
# (printed by envs/stofm.sh; downloaded 2026-11-16, files dated 2025-07-24).
WEIGHT_SHA256 = {
    'ckpt/config.json':
        '4baa758a7eaa7917e2cd0705aec152360dda9971a0c15cdf28b1a29aa1bf8431',
    'ckpt/se2transformer.pth':
        'eb60cbe9adec902861a14af8d7a4b261c933dfc0a1189d22428f2d3cd442ddf1',
    'ckpt/cell_encoder/cell_proj.bin':
        '1d62968b96f39453e117323f2945290939553022217e38939dbe29e89038b57d',
    'ckpt/cell_encoder/cell_bert/config.json':
        '448dd7d18c84da9634189bb85b5de22bd487b479ac5874634827767b3f7ec772',
    'ckpt/cell_encoder/cell_bert/model.safetensors':
        '1e4ddba891c8d50513c6747fabaab1a39844b7d9760230aedefac40b0476ea66',
}
sys.path.insert(0, str(STOFM_CODE_PATH))
# geneformer 0.0.1 (installed from the copy bundled with SToFM) imports anndata,
# which probes for cupy: import it before the stubs below exist.
import geneformer  # noqa: E402
from geneformer.tokenizer import (GENE_MEDIAN_FILE, TOKEN_DICTIONARY_FILE,  # noqa: E402
                                  tokenize_cell)

sys.path.append(str(Path(__file__).parent))
from bench_utils import (CACHE_FOLDER_PATH, SECTION_KEY, Timer,  # noqa: E402
                         ensembl_ids, frac_duplicate_rows, git_commit,
                         load_benchmark_dataset, patched, save_embedding)


@contextmanager
def stub_modules(names):
    """Inside: importing any of `names` gives an empty module whose attributes do not
    exist. Afterwards they are removed again, so that later imports and library probes
    (anndata, scanpy, scikit-learn) see them as not installed."""
    present = [n for n in names if n in sys.modules]
    if present:
        raise RuntimeError(f'{present} already imported: cannot stub them.')

    def stub(name):
        module = types.ModuleType(name)

        def missing(attr):
            raise AttributeError(f'{name} is a stub ({attr}): run_stofm.py uses the CPU '
                                 'path of the SToFM code only')
        module.__getattr__ = missing
        return module

    sys.modules.update({n: stub(n) for n in names})
    try:
        yield
    finally:
        for n in names:
            sys.modules.pop(n, None)


# extraction.py imports rapids_singlecell and cupy unconditionally; they are only used
# on its GPU path (device >= 0), which this script never takes.
with stub_modules(['rapids_singlecell', 'cupy']):
    from model.extraction import SToFM_Collator, load_data  # noqa: E402
    from model.se2transformer import SToFMConfig, SToFMModel  # noqa: E402
    from model.utils import CellEncoderCollator  # noqa: E402

METHOD = 'stofm'
EMB_KEY = 'X_stofm'
CELL_EMB_KEY = 'X_stofm_cell'
# get_embeddings.sh passes --seed 1 (the script's default is 42). It seeds python, numpy
# and torch; numpy's state reaches the PCA of the virtual cells (sklearn's randomized
# solver on sections of 500-2560 cells).
SEED = 1
SPLIT_NUM = 1000          # get_embeddings.py defaults
LEIDEN_RES = 1.0
ALPHA = 0.2
BATCH_SIZE = 4            # sub-slices per SE(2) batch (padding is masked)
CELL_BATCH_SIZE = 32      # encode_cell's batch size in get_embeddings.py
MAX_INPUT_SIZE = 2048     # Geneformer's create_dataset truncation
CELL_EMB_DIM = 256        # cell_proj
TOKENIZE_CHUNK = 2048     # cells densified at a time by tokenize (memory only)
PACKAGES = ['torch', 'transformers', 'numpy', 'scipy', 'scikit-learn', 'anndata',
            'scanpy', 'igraph', 'pynndescent', 'umap-learn', 'numba', 'pandas',
            'geneformer', 'datasets']


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check_weights(model_path=STOFM_MODEL_PATH):
    """The checkpoint files must be the ones pinned in WEIGHT_SHA256."""
    for name, expected in WEIGHT_SHA256.items():
        found = file_sha256(model_path / name)
        if found != expected:
            raise ValueError(f'{model_path / name} has sha256 {found}, expected {expected} '
                             '(envs/stofm.sh).')
    return dict(WEIGHT_SHA256)


def code_revision(code_path=STOFM_CODE_PATH):
    """The SToFM checkout must be the pinned commit without local changes, and the
    installed geneformer must have the dictionaries of the copy bundled in it."""
    commit = git_commit(code_path)
    if commit.endswith('+dirty') or not CODE_REVISION.startswith(commit):
        raise ValueError(f'SToFM code at {code_path} is {commit}, expected a clean '
                         f'checkout of {CODE_REVISION} (envs/stofm.sh).')
    installed = Path(geneformer.__file__).parent
    bundled = code_path / 'geneformer_001' / 'geneformer'
    for name in ['token_dictionary.pkl', 'gene_median_dictionary.pkl',
                 'gene_name_id_dict.pkl']:
        if (installed / name).read_bytes() != (bundled / name).read_bytes():
            raise ValueError(f'{installed / name} differs from SToFM\'s bundled '
                             f'geneformer {bundled / name} (envs/stofm.sh).')
    return CODE_REVISION


def seed_everything(seed=SEED):
    """As get_embeddings.py, plus torch's deterministic algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class Pooler(nn.Module):
    """get_embeddings.py's Pooler (unchanged except map_location, so that it also
    loads without a GPU): CLS output -> cell_proj -> L2 norm."""
    def __init__(self, config, pretrained_proj, proj_dim):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj, map_location='cpu'))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output


def load_models(device, model_path=STOFM_MODEL_PATH):
    """The SE(2) transformer and the cell encoder, loaded as in get_embeddings.py.

    The SE(2) transformer is built from the shipped config.json (12 layers; the paper's
    Table A.8 says 4) and its state dict must match exactly. The cell encoder's own BERT
    pooler is not in the checkpoint (transformers warns that it is newly initialized)
    and is replaced by the Pooler, as in get_embeddings.py. Its attention is the eager
    implementation, the only one of the authors' transformers 4.39."""
    ckpt = model_path / 'ckpt'
    config = SToFMConfig.from_pretrained(str(ckpt / 'config.json'))
    model = SToFMModel(config)
    model.load_state_dict(torch.load(ckpt / 'se2transformer.pth', map_location='cpu',
                                     weights_only=True))
    model = model.to(device).eval()
    cell_encoder = BertModel.from_pretrained(str(ckpt / 'cell_encoder' / 'cell_bert'),
                                             attn_implementation='eager')
    cell_encoder.pooler = Pooler(cell_encoder.config,
                                 pretrained_proj=str(ckpt / 'cell_encoder' / 'cell_proj.bin'),
                                 proj_dim=CELL_EMB_DIM)
    cell_encoder = cell_encoder.to(device).eval()
    return model, cell_encoder, config


def gene_tokens(dataset_name, gene_names):
    """The panel genes Geneformer's tokenizer uses, as preprocess.py selects them from
    var['ensembl_id'] (genes in the gene-median dictionary).

    The Ensembl IDs are the silver files'. A gene whose ID is not in the dictionary
    falls back to Geneformer's own ID for its symbol (gene_name_id_dict), if that one
    is in the dictionary and no other panel gene has it. Returns the var['ensembl_id']
    the tokenizer is given, the positions of the dictionary genes in the panel, their
    tokens and gene medians, and a report."""
    with open(GENE_MEDIAN_FILE, 'rb') as f:
        medians = pickle.load(f)
    with open(TOKEN_DICTIONARY_FILE, 'rb') as f:
        token_dict = pickle.load(f)
    with open(Path(GENE_MEDIAN_FILE).parent / 'gene_name_id_dict.pkl', 'rb') as f:
        name_to_id = pickle.load(f)
    gene_names = np.asarray(gene_names, dtype=str)
    silver = ensembl_ids(dataset_name, gene_names)
    ids = silver.copy()
    used = set(silver[[e in medians for e in silver]])
    by_symbol = {}
    for i, (name, e) in enumerate(zip(gene_names, silver)):
        alt = name_to_id.get(name)
        if e not in medians and alt in medians and alt not in used:
            ids[i] = alt
            used.add(alt)
            by_symbol[str(name)] = f'{e} -> {alt}'
    loc = np.where([e in medians for e in ids])[0]
    if len(set(ids[loc])) != len(loc):
        raise ValueError('Two panel genes have the same Geneformer gene.')
    report = dict(n_genes_panel=len(gene_names), n_genes_in_dictionary=len(loc),
                  genes_matched_by_symbol=by_symbol,
                  genes_not_in_dictionary=gene_names[np.setdiff1d(np.arange(len(ids)), loc)]
                  .tolist())
    return (ids, loc, np.array([token_dict[e] for e in ids[loc]]),
            np.array([medians[e] for e in ids[loc]]), report)


def tokenize(X, loc, tokens, norm_factors, chunk=TOKENIZE_CHUNK):
    """Geneformer rank-value tokens of each cell (row of X: raw counts of all panel
    genes): SToFMTranscriptomeTokenizer.tokenize_anndata (preprocess.py) followed by the
    truncation in Geneformer's create_dataset, on row chunks of a sparse matrix. The
    total count is over all panel genes and every cell is kept, as in preprocess.py.
    Returns the tokens and the number of dictionary genes detected per cell."""
    X = sp.csr_matrix(X)
    n_counts = np.asarray(X.sum(axis=1)).ravel()
    if (n_counts <= 0).any():
        raise ValueError(f'{(n_counts <= 0).sum()} cells without counts: the authors\' '
                         'normalization divides by their total.')
    cells, n_detected = [], []
    for start in range(0, X.shape[0], chunk):
        rows = slice(start, start + chunk)
        # preprocess.py, l.59-64 (same operations in the same order)
        norm = (X[rows][:, loc].toarray().T / n_counts[rows] * 10_000
                / norm_factors[:, None])
        for i in range(norm.shape[1]):
            sentence = tokenize_cell(norm[:, i], tokens)
            n_detected.append(len(sentence))
            cells.append(sentence[:MAX_INPUT_SIZE])
    return cells, np.array(n_detected)


@torch.no_grad()
def encode_cells(cell_encoder, cells, device, batch_size=CELL_BATCH_SIZE):
    """extraction.encode_cell: batches in cell order, the authors' CellEncoderCollator
    (prepends <cls>, truncates to 2048, pads), pooler output. Each cell gets a dummy
    label: without one, Geneformer's collator raises KeyError 'label'."""
    collator = CellEncoderCollator(add_cls=True)
    out = np.empty((len(cells), CELL_EMB_DIM), dtype=np.float32)
    for start in range(0, len(cells), batch_size):
        batch = collator([{'input_ids': c.tolist(), 'label': 0}
                          for c in cells[start:start + batch_size]])
        result = cell_encoder(batch['input_ids'].to(device),
                              batch['attention_mask'].to(device))
        out[start:start + batch_size] = result.pooler_output.float().cpu().numpy()
    return out


def exact_neighbors():
    """Inside: scanpy's kNN graphs are exact (brute force) for any number of points.

    scanpy itself switches to approximate neighbours (pynndescent) at 8192 points; the
    authors' default GPU path (rapids) is exact. Below 8192 points, which includes every
    sub-slice, this changes nothing."""
    original = sc.pp.neighbors

    def neighbors(adata, *args, **kwargs):
        if kwargs.get('transformer') is None:
            kwargs['transformer'] = 'sklearn'
        return original(adata, *args, **kwargs)

    return patched(sc.pp, 'neighbors', neighbors)


@contextmanager
def capture_leiden(store):
    """Inside: the labels of every sc.tl.leiden call are appended to store. load_data
    calls it once, for the virtual cells of the whole section (cells in input order)."""
    original = sc.tl.leiden

    def leiden(adata, *args, **kwargs):
        result = original(adata, *args, **kwargs)
        store.append(adata.obs[kwargs.get('key_added', 'leiden')].astype(str).to_numpy())
        return result

    with patched(sc.tl, 'leiden', leiden):
        yield


def normalized_coordinates(xy):
    """Coordinates as load_data normalizes them (extraction.py l.256-263)."""
    max_x, min_x = np.max(xy[:, 0]), np.min(xy[:, 0])
    max_y, min_y = np.max(xy[:, 1]), np.min(xy[:, 1])
    x_mid, y_mid = (max_x + min_x) / 2, (max_y + min_y) / 2
    graph_size = ((max_x - min_x) ** 2 + (max_y - min_y) ** 2) ** 0.5
    return (xy - [x_mid, y_mid]) / graph_size * 100


def grid_squares(nodes, num=SPLIT_NUM):
    """The square of extraction.split_graph's grid each cell lies in (diagnostics only;
    embed_section checks that every sub-slice lies in one square)."""
    node_num = nodes.shape[0]
    square = np.full(node_num, -1)
    if node_num <= num:
        square[:] = 0
        return square
    max_x, min_x = np.max(nodes[:, 0]), np.min(nodes[:, 0])
    max_y, min_y = np.max(nodes[:, 1]), np.min(nodes[:, 1])
    x_len, y_len = max_x - min_x, max_y - min_y
    split_num = node_num // num + 1
    split_len = np.sqrt(x_len * y_len / split_num)
    split_x, split_y = int(x_len / split_len) + 1, int(y_len // split_len) + 1
    for i in range(split_x):
        for j in range(split_y):
            mask = ((nodes[:, 0] >= min_x + i * split_len)
                    & (nodes[:, 0] < min_x + (i + 1) * split_len)
                    & (nodes[:, 1] >= min_y + j * split_len)
                    & (nodes[:, 1] < min_y + (j + 1) * split_len))
            square[mask] = i * split_y + j
    return square


def knn_distance(xy, k=10):
    """Distance of each point to its k-th nearest other point."""
    return cKDTree(xy).query(xy, k=k + 1)[0][:, k]


def obs_order_split(nodes, square, subslice, num=SPLIT_NUM):
    """Cells in grid squares that split_graph cuts into chunks in obs order (squares with
    >= 2 num cells), and for them the 10-NN distance within their chunk / within their
    square (1: chunks as compact as the square; a random obs order gives about the
    square root of the number of chunks)."""
    affected, ratios = 0, []
    for s in np.unique(square):
        cells = np.where(square == s)[0]
        if len(cells) // num < 2:
            continue
        affected += len(cells)
        full = knn_distance(nodes[cells])
        for k in np.unique(subslice[cells]):
            part = subslice[cells] == k
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios.append(knn_distance(nodes[cells[part]]) / full[part])
    ratios = np.concatenate(ratios) if ratios else np.array([])
    ratios = ratios[np.isfinite(ratios)]
    return dict(
        frac_cells_obs_order_split=affected / len(nodes),
        obs_order_chunk_knn_ratio_median=float(np.median(ratios)) if len(ratios) else None,
        obs_order_chunk_knn_ratio_p90=float(np.quantile(ratios, 0.9)) if len(ratios) else None)


def embed_section(model, cell_emb, xy, folder, device, batch_size=BATCH_SIZE):
    """The authors' load_data and the embedding loop of get_embeddings.py on one section
    (cell embeddings cell_emb, coordinates xy). load_data reads its inputs from files,
    written to `folder` (deleted afterwards).

    Returns the embedding, each cell's sub-slice and virtual-cell cluster, and
    diagnostics."""
    n = len(xy)
    if n > SPLIT_NUM and np.ptp(xy, axis=0).min() <= 0:
        raise ValueError('Degenerate section: split_graph divides by its extent.')
    folder.mkdir(parents=True)
    ad.AnnData(obs=pd.DataFrame(index=np.arange(n).astype(str)),
               obsm={'spatial': np.asarray(xy, dtype=np.float64)}).write_h5ad(folder / 'data.h5ad')
    np.save(folder / 'ce_emb.npy', cell_emb)
    clusters = []
    with exact_neighbors(), capture_leiden(clusters):
        graphs = load_data(data_path=str(folder / 'data.h5ad'),
                           emb_path=str(folder / 'ce_emb.npy'), new_emb=False,
                           device=-1, filter=False, split_num=SPLIT_NUM,
                           leiden_res=LEIDEN_RES, alpha=ALPHA)
    shutil.rmtree(folder)
    if len(clusters) != 1:
        raise RuntimeError(f'Expected one Leiden call (virtual cells), got {len(clusters)}.')

    subslice = np.full(n, -1)
    for k, g in enumerate(graphs):
        cells = g['indices'][g['indices'] != -1].numpy()
        if (subslice[cells] != -1).any():
            raise RuntimeError('A cell is in two sub-slices.')
        subslice[cells] = k
    if (subslice == -1).any():
        raise RuntimeError(f'{(subslice == -1).sum()} cells are in no sub-slice.')

    emb = torch.full((n, model.config.embedding_dim), float('nan'))
    loader = DataLoader(graphs, collate_fn=SToFM_Collator(mask=False, mask_pair=False),
                        batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for graph in loader:
            indices = graph['indices']
            graph = {k: v.to(device) for k, v in graph.items()}
            node_rep = model(**graph)['last_hidden_state']
            keep = indices != -1
            emb[indices[keep].long()] = node_rep[keep.to(node_rep.device)].float().cpu()

    nodes = normalized_coordinates(np.asarray(xy, dtype=np.float64))
    square = grid_squares(nodes)
    if any(len(np.unique(square[subslice == k])) != 1 for k in range(len(graphs))):
        raise RuntimeError('A sub-slice spans two grid squares: grid_squares does not '
                           'match the authors\' split_graph.')
    tokens = [len(g['token_types']) for g in graphs]
    diagnostics = dict(
        n_cells=n, n_subslices=len(graphs),
        max_subslice_cells=int(np.bincount(subslice).max()), max_tokens=max(tokens),
        n_virtual_cells=int((graphs[0]['token_types'] == 2).sum()),
        **obs_order_split(nodes, square, subslice))
    return emb.numpy(), subslice, clusters[0], diagnostics


def main(args):
    adata = load_benchmark_dataset(args.dataset, subsample=args.subsample)
    dataset = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    adata.uns['benchmark']['dataset_name'] = dataset
    print(adata)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = check_weights()
    code_commit = code_revision()
    cache_root = Path(CACHE_FOLDER_PATH) / METHOD
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(tempfile.mkdtemp(prefix=f'{dataset}_', dir=cache_root))

    with Timer() as t_load:
        model, cell_encoder, config = load_models(device)
    seconds = t_load.seconds
    _, loc, tokens, norm_factors, matching = gene_tokens(
        args.dataset, adata.var_names.astype(str))
    print(f'{matching["n_genes_in_dictionary"]} of {matching["n_genes_panel"]} panel genes '
          f'in Geneformer\'s dictionary; by symbol: {matching["genes_matched_by_symbol"]}; '
          f'not in the dictionary: {len(matching["genes_not_in_dictionary"])}', flush=True)

    X = sp.csr_matrix(adata.X)
    xy_all = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    emb = np.full((adata.n_obs, config.embedding_dim), np.nan, dtype=np.float32)
    cell_emb = np.full((adata.n_obs, CELL_EMB_DIM), np.nan, dtype=np.float32)
    n_detected = np.zeros(adata.n_obs, dtype=np.int64)
    subslice = np.empty(adata.n_obs, dtype=object)
    virtual_cell = np.empty(adata.n_obs, dtype=object)
    sections, n_dup_xy = [], 0.0
    for i, section in enumerate(adata.obs[SECTION_KEY].cat.categories):
        rows = np.where((adata.obs[SECTION_KEY] == section).values)[0]
        if not len(rows):
            continue
        seed_everything(SEED)
        with Timer() as t_cells:
            cells, n_detected[rows] = tokenize(X[rows], loc, tokens, norm_factors)
            cell_emb[rows] = encode_cells(cell_encoder, cells, device)
        with Timer() as t_se2:
            emb[rows], sub, clusters, diag = embed_section(
                model, cell_emb[rows], xy_all[rows], cache_dir / f'section{i}', device,
                args.batch_size)
        seconds += t_cells.seconds + t_se2.seconds
        subslice[rows] = [f'{section}:{k}' for k in sub]
        virtual_cell[rows] = [f'{section}:{c}' for c in clusters]
        n_dup_xy += frac_duplicate_rows(xy_all[rows]) * len(rows)
        sections.append(dict(section=section, **diag,
                             n_cells_without_dictionary_genes=int((n_detected[rows] == 0).sum()),
                             # Geneformer keeps 2048 tokens, then <cls> replaces the last one
                             n_cells_truncated=int((n_detected[rows] >= MAX_INPUT_SIZE).sum()),
                             cell_encoder_seconds=t_cells.seconds,
                             se2_seconds=t_se2.seconds))
        print(f'{section}: {len(rows)} cells, {diag["n_subslices"]} sub-slices '
              f'(max {diag["max_tokens"]} tokens), {diag["n_virtual_cells"]} virtual cells, '
              f'{diag["frac_cells_obs_order_split"]:.2f} of cells in obs-order chunks; '
              f'{t_cells.seconds:.0f} s + {t_se2.seconds:.0f} s', flush=True)
    shutil.rmtree(cache_dir)
    adata.obsm[EMB_KEY] = emb
    adata.obsm[CELL_EMB_KEY] = cell_emb
    # SToFM's sub-slices and virtual-cell clusters (built without labels; for
    # interpreting the embedding, not scored)
    adata.obs['stofm_subslice'] = pd.Categorical(subslice)
    adata.obs['stofm_virtual_cell'] = pd.Categorical(virtual_cell)

    if not (np.isfinite(emb).all() and np.isfinite(cell_emb).all()):
        raise ValueError('Missing or non-finite SToFM embeddings.')
    save_embedding(adata, METHOD, EMB_KEY, dict(
        embedding_seconds=seconds,
        model=dict(repo='PharMolix/SToFM',
                   weights='Google Drive folder 1mHE8gf8MAPwzZoEB0vwOOfQ4lz3H_-xo (README)',
                   weights_sha256=weights, path=str(STOFM_MODEL_PATH),
                   n_parameters_se2=sum(p.numel() for p in model.parameters()),
                   n_parameters_cell_encoder=sum(p.numel() for p in cell_encoder.parameters()),
                   se2_config=config.to_dict()),
        code=dict(repo='PharMolix/SToFM', commit=code_commit, path=str(STOFM_CODE_PATH),
                  geneformer=str(Path(geneformer.__file__).parent)),
        **matching,
        n_cells_without_dictionary_genes=int((n_detected == 0).sum()),
        n_cells_truncated=int((n_detected >= MAX_INPUT_SIZE).sum()),
        median_dictionary_genes_per_cell=float(np.median(n_detected)),
        frac_duplicate_coordinates=n_dup_xy / adata.n_obs,   # within a section
        frac_duplicate_embeddings=frac_duplicate_rows(emb),
        frac_duplicate_cell_embeddings=frac_duplicate_rows(cell_emb),
        cell_embedding_key=CELL_EMB_KEY,
        device=torch.cuda.get_device_name() if device == 'cuda' else device,
        peak_gpu_memory_gb=(torch.cuda.max_memory_allocated() / 1e9
                            if device == 'cuda' else None),
        package_versions={p: importlib.metadata.version(p) for p in PACKAGES},
        params=dict(seed=SEED, split_num=SPLIT_NUM, leiden_res=LEIDEN_RES, alpha=ALPHA,
                    batch_size=args.batch_size, cell_batch_size=CELL_BATCH_SIZE,
                    max_input_size=MAX_INPUT_SIZE,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    normalization='counts / total over all panel genes x 1e4 / Geneformer '
                                  'gene median; rank-value tokens (preprocess.py)',
                    gene_matching='silver Ensembl ID in Geneformer\'s dictionary; else '
                                  'Geneformer\'s ID for the symbol',
                    cell_encoder='CLS -> cell_proj -> L2 norm; eager attention',
                    knn='exact (scanpy brute force) for the virtual cells and sub-slices',
                    subslices='authors\' split_graph: square grid; squares with >= 2000 '
                              'cells cut into chunks in obs order',
                    virtual_cells='Leiden on the whole section, added to every sub-slice',
                    spatial_context='per section; 50-NN attention bias within ~1000-cell '
                                    'sub-slices, plus the section\'s virtual cells',
                    embedding='last hidden state of the cell tokens',
                    integration='none'),
        sections=sections))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--subsample', type=int, default=None,
                        help='Random subset of cells (smoke tests only).')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Sub-slices per SE(2) batch (GPU memory); padding is masked, '
                             'so the embedding does not depend on it.')
    main(parser.parse_args())
