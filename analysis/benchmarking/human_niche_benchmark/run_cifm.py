"""CIFM zero-shot microenvironment embeddings for the human niche benchmark.

CIFM (Cellular Interaction Foundation Model; You et al., bioRxiv 2025,
10.1101/2025.01.25.634867; https://huggingface.co/ynyou/CIFM) is an E(n)-
equivariant GNN over a 20 um radius graph of cells. ``CIFM.embed(adata)`` gives
one 1024-d embedding per cell for its microenvironment. This script follows the
authors' recipe (test.ipynb and the preprint) with the pretrained weights and
no fine-tuning:

- raw counts of the panel genes, mitochondrial genes removed (preprint),
  normalize_total(1e4) + log1p (test.ipynb);
- genes matched to the model's 18,289 input channels by Ensembl ID. The panel
  is written into the model's gene order with zeros for unmeasured genes. For
  one Ensembl ID per gene this equals the authors' ``channel_matching`` (zero
  weights for unmatched genes) without its quadratic Python loop
  (checks/test_cifm.py compares the two);
- ``model.encode`` on ``radius_graph(r=20 um, loop=True)``, as in ``embed``.

The authors' ``embed`` runs the whole graph at once, which does not fit on a GPU
for the larger sections. Each section is therefore split into spatial tiles,
and each tile is embedded with a halo of HALO_UM around it. The model has
2 message-passing layers over a 20 um graph, so a cell's embedding depends only
on cells within 40 um: with a 45 um halo the tiled embeddings equal the
whole-section ones (checks/test_cifm.py). Graphs never cross sections.

Writes the embedding only; job_wrapper.sh then clusters and scores it with
score.py. Run in the cifm environment (envs/cifm.sh) on a GPU:
    SCRIPT=run_cifm.py PYTHON_ENV=PYTHON_CIFM DATASET=xhs49-1b_1p \\
        bsub -G s10396 -q inference -n 8 -M 64GB \\
        -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]" \\
        -gpu "num=1:mode=exclusive_process" -W 12:00 < run_method.lsf
(README: settings for vhb3003-3b_1p.)
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')   # compute nodes have no internet

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

# Local snapshot of ynyou/CIFM (envs/cifm.sh); its models_cifm package is the
# authors' model code.
CIFM_MODEL_PATH = Path(os.environ.get(
    'CIFM_MODEL_PATH', '/lustre/scratch126/cellgen/lotfollahi/jr43/models/CIFM'))
CIFM_REVISION = '19c507f00b354c7dc202d850292c268b06c2ec50'
sys.path.insert(0, str(CIFM_MODEL_PATH))
from models_cifm.cifm import CIFM  # noqa: E402
from torch_geometric.nn import radius_graph  # noqa: E402

sys.path.append(str(Path(__file__).parent))
from bench_utils import (DATASETS, SECTION_KEY, SILVER_FOLDER_PATH,  # noqa: E402
                         Timer, frac_duplicate_rows, hf_snapshot_revision,
                         load_benchmark_dataset, save_embedding)

METHOD = 'cifm'
EMB_KEY = 'X_cifm'
TARGET_SUM = 1e4
HALO_UM = 45.0              # > receptive field: num_layer (2) x radius (20 um)
MAX_TILE_CELLS = 30000      # core cells per tile (GPU memory); halo comes on top


def snapshot_revision(model_path=CIFM_MODEL_PATH):
    """Commit and sha256 of model.safetensors, from huggingface_hub's download metadata."""
    return hf_snapshot_revision(model_path, 'model.safetensors', CIFM_REVISION, 'envs/cifm.sh')


def load_model(device, model_path=CIFM_MODEL_PATH):
    """Pretrained CIFM and its input channels (one Ensembl ID per channel)."""
    torch.serialization.add_safe_globals([argparse.Namespace])
    args = torch.load(model_path / 'models_cifm' / 'args.pt', weights_only=True)
    model = CIFM.from_pretrained(str(model_path), args=args).to(device).eval()
    channels = torch.load(model_path / 'models_cifm' / 'channel2ensembl.pt',
                          weights_only=True)
    if any(len(c) != 1 for c in channels):
        raise ValueError('Expected one Ensembl ID per CIFM input channel.')
    model_genes = np.array([c[0] for c in channels])
    if len(set(model_genes)) != len(model_genes) or len(model_genes) != args.in_dim:
        raise ValueError('CIFM input channels are not unique Ensembl IDs.')
    return model, model_genes, args


def ensembl_ids(dataset_name, gene_names):
    """Ensembl IDs of the panel genes, from the silver files' var.

    As in run_nicheformer.py (branch human-niche-benchmark):
    load_benchmark_dataset concatenates sections with ad.concat, which drops
    var columns, so they are read from the source files here.
    """
    cfg = DATASETS[dataset_name]
    folder = Path(cfg.get('silver_folder', SILVER_FOLDER_PATH)) / dataset_name
    mapping = {}
    for file in sorted(folder.glob('*.h5ad')):
        if not any(p in file.name for p in cfg['file_patterns']):
            continue
        var = ad.read_h5ad(file, backed='r').var
        for name, ensembl in zip(var.index.astype(str), var['ensembl_id'].astype(str)):
            if mapping.setdefault(name, ensembl) != ensembl:
                raise ValueError(f'{name} has two Ensembl IDs across sections.')
    return np.array([mapping[g] for g in gene_names])


def normalized_panel(adata):
    """log1p(CP10k) of the panel without mitochondrial genes (AnnData, sparse X)."""
    mt = adata.var_names.str.upper().str.startswith('MT-')
    panel = ad.AnnData(X=sp.csr_matrix(adata.X[:, ~mt], dtype=np.float32),
                       obs=pd.DataFrame(index=adata.obs_names),
                       var=pd.DataFrame(index=adata.var_names[~mt]))
    sc.pp.normalize_total(panel, target_sum=TARGET_SUM)
    sc.pp.log1p(panel)
    return panel, int(mt.sum())


def to_model_genes(X, ensembl, model_genes):
    """Columns of X in the model's channel order; unmeasured channels are 0."""
    pos = {g: i for i, g in enumerate(model_genes)}
    keep = np.array([g in pos for g in ensembl])
    cols = np.array([pos[g] for g in ensembl[keep]], dtype=np.int64)
    if len(set(cols)) != len(cols):
        raise ValueError('Duplicate Ensembl IDs in the panel.')
    X = sp.csr_matrix(X)[:, np.where(keep)[0]].tocoo()
    X = sp.csr_matrix((X.data, (X.row, cols[X.col])),
                      shape=(X.shape[0], len(model_genes)), dtype=np.float32)
    return X, keep


def split_tiles(xy, idx, max_cells):
    """Recursive median bisection along the longer axis until <= max_cells."""
    if len(idx) <= max_cells:
        return [idx]
    pts = xy[idx]
    axis = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
    order = idx[np.argsort(pts[:, axis], kind='stable')]
    half = len(order) // 2
    return (split_tiles(xy, order[:half], max_cells)
            + split_tiles(xy, order[half:], max_cells))


@torch.no_grad()
def encode(model, X, xy, device):
    """model.encode on the radius graph, as in CIFM.embed (X: csr, xy: um)."""
    expressions = torch.tensor(X.toarray(), dtype=torch.float32).to(device)
    coordinates = torch.tensor(xy, dtype=torch.float32)
    coordinates = torch.cat([coordinates, torch.zeros(coordinates.shape[0], 1)], dim=1).to(device)
    edge_index = radius_graph(coordinates, r=model.radius_spatial_graph,
                              max_num_neighbors=10000, loop=True)
    emb = model.encode(expressions, coordinates, edge_index)
    degree = torch.bincount(edge_index[1], minlength=xy.shape[0])   # incl. self loop
    return emb.float().cpu().numpy(), degree.cpu().numpy()


def embed_section(model, X, xy, device, max_tile_cells=MAX_TILE_CELLS, halo=HALO_UM):
    """Embeddings of one section, tile by tile with a halo (exact; see module doc)."""
    emb = np.zeros((X.shape[0], model.hidden_dim), dtype=np.float32)
    degree = np.zeros(X.shape[0], dtype=np.int64)
    tiles = split_tiles(xy, np.arange(X.shape[0]), max_tile_cells)
    max_nodes = 0
    for core in tiles:
        lo, hi = xy[core].min(axis=0) - halo, xy[core].max(axis=0) + halo
        nodes = np.where(np.all((xy >= lo) & (xy <= hi), axis=1))[0]
        is_core = np.zeros(X.shape[0], dtype=bool)
        is_core[core] = True
        part, deg = encode(model, X[nodes], xy[nodes], device)
        emb[nodes[is_core[nodes]]] = part[is_core[nodes]]
        degree[nodes[is_core[nodes]]] = deg[is_core[nodes]]
        max_nodes = max(max_nodes, len(nodes))
    return emb, degree, len(tiles), max_nodes


def main(args):
    adata = load_benchmark_dataset(args.dataset, subsample=args.subsample)
    dataset = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    adata.uns['benchmark']['dataset_name'] = dataset
    print(adata)

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    commit, weights_sha256 = snapshot_revision()

    with Timer() as t_embed:
        model, model_genes, model_args = load_model(device)
        panel, n_mt = normalized_panel(adata)
        ensembl = ensembl_ids(args.dataset, panel.var_names.astype(str))
        X, matched = to_model_genes(panel.X, ensembl, model_genes)
        unmatched = panel.var_names[~matched].astype(str).tolist()
        print(f'{matched.sum()} of {len(matched)} panel genes matched to CIFM channels '
              f'({n_mt} mitochondrial genes removed); unmatched: {unmatched}')
        n_zero = int((np.asarray(panel.X.sum(axis=1)).ravel() == 0).sum())

        xy_all = np.asarray(adata.obsm['spatial'], dtype=np.float64)
        emb = np.zeros((adata.n_obs, model.hidden_dim), dtype=np.float32)
        degree = np.zeros(adata.n_obs, dtype=np.int64)
        n_tiles, max_nodes = 0, 0
        for section in adata.obs[SECTION_KEY].cat.categories:
            rows = np.where((adata.obs[SECTION_KEY] == section).values)[0]
            e, d, n, m = embed_section(model, X[rows], xy_all[rows], device,
                                       args.max_tile_cells)
            emb[rows], degree[rows] = e, d
            n_tiles, max_nodes = n_tiles + n, max(max_nodes, m)
            print(f'{section}: {len(rows)} cells, {n} tiles, median degree '
                  f'{np.median(d):.0f} (incl. self)', flush=True)
        adata.obsm[EMB_KEY] = emb

    if not np.isfinite(emb).all():
        raise ValueError('Non-finite CIFM embeddings.')
    peak_gpu_gb = (torch.cuda.max_memory_allocated() / 1e9
                   if device == 'cuda' else None)
    save_embedding(adata, METHOD, EMB_KEY, dict(
        embedding_seconds=t_embed.seconds,
        model=dict(repo='ynyou/CIFM', revision=commit,
                   weights_sha256=weights_sha256, path=str(CIFM_MODEL_PATH),
                   n_parameters=sum(p.numel() for p in model.parameters()),
                   **vars(model_args)),
        n_genes_panel=int(adata.n_vars), n_genes_mitochondrial_removed=n_mt,
        n_genes_matched=int(matched.sum()), unmatched_genes=unmatched,
        n_zero_count_cells=n_zero,
        frac_isolated_cells=float((degree == 1).mean()),
        median_degree=float(np.median(degree)),
        frac_duplicate_embeddings=frac_duplicate_rows(emb),
        n_tiles=n_tiles, max_tile_nodes=max_nodes, device=device,
        peak_gpu_memory_gb=peak_gpu_gb,
        params=dict(target_sum=TARGET_SUM, log1p=True, mitochondrial_genes='removed',
                    gene_matching='Ensembl ID, unmatched channels 0',
                    radius_um=model.radius_spatial_graph, self_loops=True,
                    halo_um=HALO_UM, max_tile_cells=args.max_tile_cells,
                    spatial_context='20 um radius graph, 2 EGNN layers, per section',
                    integration='none')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--subsample', type=int, default=None,
                        help='Random subset of cells (smoke tests only).')
    parser.add_argument('--max_tile_cells', type=int, default=MAX_TILE_CELLS,
                        help='Core cells per tile (GPU memory).')
    main(parser.parse_args())
