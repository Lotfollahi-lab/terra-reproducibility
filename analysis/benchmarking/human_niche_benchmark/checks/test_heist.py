"""Checks of run_heist.py against the authors' HEIST code (heist env, GPU).

The authors' tutorial (cell_embeddings.ipynb) stores no outputs, so unlike
test_cifm.py there are no published values to reproduce. The checks:

(a) The weights are the pinned snapshot (sha256 of model.safetensors equals the
    download metadata) and the code is a clean checkout of the pinned commit.
(b) Input path, on a benchmark section (smallest xhs1000 section, ~18k cells,
    one tile): run_heist (cells from load_benchmark_dataset -> section_adata ->
    embed_section) equals the authors' ``embed_adata`` on the same section read
    straight from its silver file, each with its own fresh graph cache, the same
    seed and run_heist's reproducible gene networks: identical graphs (cell
    graph, Leiden clusters, HVGs, gene inputs, gene networks) and embeddings.
    This checks cell order, genes and coordinates, and that seeded runs are
    reproducible. The authors' path builds the Voronoi polygons and run_heist
    does not, so it also shows that the model does not read them.
(c) Every cell has its own gene input: only cells with identical HVG expression
    (e.g. no counts in the HVGs) share one. The bug the authors fixed made all
    cells of a cluster share one.
(d) Gene networks: run_heist's (fixed KSG batch size, torch reseeded) against the
    authors' unpatched estimator, run twice on each cluster of (b). The patch only
    changes which tie-breaking noise each gene pair gets, so its networks may
    differ from the authors' no more than the authors' two runs differ from each
    other (mean fraction of differing edges over the clusters; up to 2x + 0.01).
(e) Informational: how far another seed moves the embedding (MAGIC and the MI
    estimator draw random numbers).

    bsub -G s10396 -q inference -n 8 -M 32GB \\
        -R "select[mem>32GB] rusage[mem=32GB] span[hosts=1]" \\
        -gpu "num=1:mode=exclusive_process" -W 2:00 -o test_heist_%J.out \\
        "$PYTHON_HEIST checks/test_heist.py"      (from the benchmark folder, after: source config.sh)
"""
import hashlib
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from run_heist import (CHECKPOINT, HEIST_MODEL_PATH, SEED,  # noqa: E402
                       check_gpu_memory, code_revision, embed_section, load_model,
                       patched, reproducible_gene_networks, section_adata,
                       seed_everything, snapshot_revision, tile_file)
# HEIST code, on sys.path via run_heist
import utils.create_coexpression_networks as heist_networks  # noqa: E402
import utils.preprocess as heist_preprocess  # noqa: E402
from utils.inference import embed_adata  # noqa: E402
from utils.preprocess import spatial_chunks  # noqa: E402
from bench_utils import (DATASETS, SECTION_KEY, SILVER_FOLDER_PATH,  # noqa: E402
                         load_benchmark_dataset)

DATASET = 'xhs1000-39b_1p'
TOL = 1e-4   # absolute; an input mismatch gives O(1) differences

failures = []


def report(name, ok, detail=''):
    print(f'{"PASS" if ok else "FAIL"}  {name}{": " + detail if detail else ""}', flush=True)
    if not ok:
        failures.append(name)


def check_snapshot():
    commit, sha256 = snapshot_revision()
    weights = HEIST_MODEL_PATH / CHECKPOINT / 'model.safetensors'
    found = hashlib.sha256(weights.read_bytes()).hexdigest()
    report('(a) model.safetensors sha256 == HF download metadata', found == sha256,
           f'{found[:12]}, revision {commit[:7]}')
    try:
        report('(a) HEIST code is a clean checkout of the pinned commit', True, code_revision()[:7])
    except ValueError as e:
        report('(a) HEIST code is a clean checkout of the pinned commit', False, str(e))


def silver_section(section):
    """The section as stored in its silver file, with the benchmark's cell filter and scale."""
    cfg = DATASETS[DATASET]
    folder = Path(cfg.get('silver_folder', SILVER_FOLDER_PATH)) / DATASET
    for file in sorted(folder.glob('*.h5ad')):
        if not any(p in file.name for p in cfg['file_patterns']):
            continue
        a = sc.read_h5ad(file)
        if not a.obs['cell_id'].astype(str).str.startswith(section + '_').all():
            continue
        if cfg['drop_unlabeled']:
            a = a[~a.obs[cfg['niche_key']].isna()].copy()
        a.obsm['spatial'] = np.asarray(a.obsm['spatial'], dtype=float) * cfg['coord_scale']
        return a
    raise ValueError(f'No silver file holds section {section}.')


def recorded_gene_networks(calls):
    """Inside: every gene network the authors' preprocess builds is appended to calls
    (its cluster's expression and genes, the keyword arguments, the edges)."""
    original = heist_preprocess.build_gene_network_gpu

    def build(cell_data, **kwargs):
        edges, weights, genes = original(cell_data, **kwargs)
        calls.append(dict(X=np.array(cell_data.X), genes=genes, kwargs=kwargs, edges=edges))
        return edges, weights, genes

    return patched(heist_preprocess, 'build_gene_network_gpu', build)


def check_section(model, vocab, cache_root):
    adata = load_benchmark_dataset(DATASET)
    section = adata.obs[SECTION_KEY].value_counts().idxmin()
    rows = np.where((adata.obs[SECTION_KEY] == section).values)[0]
    xy = adata.obsm['spatial'][rows]
    print(f'benchmark: {section}, {len(rows)} cells, {adata.n_vars} genes', flush=True)

    # run_heist path (its gene networks recorded for (d))
    ours_dir = Path(tempfile.mkdtemp(dir=cache_root))
    calls = []
    with recorded_gene_networks(calls):
        ours, tiles = embed_section(model, vocab, section_adata(adata, rows), ours_dir, 'ours')

    # Authors' path: the silver file as read, embed_adata, same seed and gene
    # networks, own cache
    raw = silver_section(section)
    ref_dir = Path(tempfile.mkdtemp(dir=cache_root))
    seed_everything(SEED)
    with reproducible_gene_networks():
        ref = embed_adata(raw, model, vocab, str(ref_dir), 'authors')
    same_order = list(raw.obs['cell_id'].astype(str)) == list(adata.obs_names[rows])
    report('(b) same cells in the same order', same_order)
    differing = compare_graphs(ours_dir, 'ours', ref_dir, 'authors', xy)
    report('(b) identical graphs (cell graph, Leiden, HVGs, gene inputs, gene networks)',
           not differing, ', '.join(differing))
    order = pd.Index(raw.obs['cell_id'].astype(str)).get_indexer(adata.obs_names[rows])
    if (order < 0).any():
        report('(b) run_heist vs. authors\' embed_adata on the silver section', False,
               f'{(order < 0).sum()} benchmark cells are not in the silver section')
    else:
        diff = float(np.abs(ours - ref[order]).max())
        report('(b) run_heist vs. authors\' embed_adata on the silver section',
               diff <= TOL, f'max |diff| = {diff:.2e} (tol {TOL:.0e}), shape {ours.shape}')

    diag = [d for _, _, d, _, _ in tiles]
    shared = max(d['frac_gene_inputs_shared_across_expression'] for d in diag)
    report('(c) only cells with identical HVG expression share a gene input', shared == 0,
           f'max fraction of cells sharing one with other expression = {shared:.4f}; '
           f'sharing one at all: {max(d["frac_duplicate_gene_inputs"] for d in diag):.4f}; '
           f'no counts in the HVGs: {max(d["frac_zero_hvg_cells"] for d in diag):.4f}')
    for (tile, *_), d in zip(tiles, diag):
        print(f'    tile {tile or "-"}: {d["n_cells"]} cells, {d["n_genes"]} genes, '
              f'Leiden res {d["leiden_resolution"]}, clusters {d["cluster_sizes"]}, '
              f'gene-network edges {d["gene_network_edges"]}, median degree '
              f'{d["median_degree"]:.0f}, isolated {d["n_isolated"]}')

    check_gene_networks(calls)

    # (e) another seed
    other, _ = embed_section(model, vocab, section_adata(adata, rows),
                             Path(tempfile.mkdtemp(dir=cache_root)), 'seed1', seed=SEED + 1)
    info(f'(e) seed {SEED + 1} vs. seed {SEED}', ours, other)


def tile_graphs(cache_dir, name, xy):
    """(tile label, cached graphs) for every tile of a section, as embed_adata names them."""
    for tile in sorted(set(spatial_chunks(np.asarray(xy)))):
        yield tile, torch.load(Path(cache_dir) / tile_file(name, tile), weights_only=False)


def compare_graphs(dir_a, name_a, dir_b, name_b, xy):
    """The parts of the cached graphs that differ between two runs on one section."""
    out = []
    for (tile, ga), (_, gb) in zip(tile_graphs(dir_a, name_a, xy),
                                   tile_graphs(dir_b, name_b, xy)):
        tile = tile or '-'
        ca, cb = ga[0], gb[0]
        parts = dict(cell_graph=torch.equal(ca.edge_index, cb.edge_index),
                     cell_features=torch.equal(ca.X, cb.X),
                     leiden=torch.equal(ca.cell_type, cb.cell_type),
                     hvgs=ca.gene_names == cb.gene_names,
                     gene_inputs=all(torch.equal(a.X, b.X) for a, b in zip(ga[1:], gb[1:])))
        out += [f'tile {tile}: {k}' for k, same in parts.items() if not same]
        n = sum(not torch.equal(a.edge_index, b.edge_index) for a, b in zip(ga[1:], gb[1:]))
        if n:
            out.append(f'tile {tile}: gene networks of {n} cells')
    return out


def edge_set(edges):
    return {tuple(e) for e in np.asarray(edges).tolist()}


def differing_edges(a, b):
    """Fraction of the edges of two networks that only one of them has (1 - Jaccard)."""
    return len(a ^ b) / max(len(a | b), 1)


def check_gene_networks(calls):
    authors = heist_networks.build_gene_network_gpu   # unpatched: batches from free memory
    d_ours, d_authors = [], []
    for k, c in enumerate(calls):
        cell_data = ad.AnnData(X=c['X'], var=pd.DataFrame(index=c['genes']))
        kwargs = {key: v for key, v in c['kwargs'].items() if key != 'ksg_batch'}
        ours = edge_set(c['edges'])
        a1, a2 = (edge_set(authors(cell_data, **kwargs)[0]) for _ in range(2))
        d_ours.append((differing_edges(ours, a1) + differing_edges(ours, a2)) / 2)
        d_authors.append(differing_edges(a1, a2))
        print(f'    cluster {k}: {len(c["X"])} cells, {len(ours)} edges (authors {len(a1)}, '
              f'{len(a2)}); differing edges: ours vs. authors {d_ours[-1]:.4f}, '
              f'authors vs. authors {d_authors[-1]:.4f}', flush=True)
    ours_mean, authors_mean = float(np.mean(d_ours)), float(np.mean(d_authors))
    report('(d) gene networks: no further from the authors\' unpatched estimator than '
           'its runs are from each other', ours_mean <= 2 * authors_mean + 0.01,
           f'mean fraction of differing edges {ours_mean:.4f} vs. {authors_mean:.4f} '
           f'(max {max(d_ours):.4f} vs. {max(d_authors):.4f}; {len(calls)} clusters)')


def info(name, a, b):
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    print(f'INFO  {name}: max |diff| = {np.abs(a - b).max():.3f}, per-cell cosine '
          f'similarity mean {cos.mean():.4f}, min {cos.min():.4f}', flush=True)


if __name__ == '__main__':
    check_snapshot()
    check_gpu_memory()
    model, vocab, _ = load_model('cuda')
    with tempfile.TemporaryDirectory() as cache_root:
        check_section(model, vocab, cache_root)
    if failures:
        sys.exit(f'{len(failures)} check(s) failed: {failures}')
    print('All HEIST checks passed.')
