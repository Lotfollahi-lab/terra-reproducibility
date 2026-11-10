"""Niche identification at a common grid of cluster numbers (default k = 5, 10, ..., 30) for all methods and datasets.

Instead of matching each method to the ground-truth niche number (scoring.py), every method is clustered at the same
cluster numbers, and NMI/ARI are averaged over the grid. Same steps as scoring.cluster_and_score otherwise: 15-NN
graph on the method's embedding (run_info emb_key), Leiden (igraph, 2 iterations, seed 0), global NMI/ARI.

Resolution search per k, with the rules of scoring.scan_resolutions:
1. one upward scan on the 0.01 grid until the cluster count has been above max(k) for SCAN_PATIENCE consecutive
   steps (or SCAN_MAX_RES);
2. for every k without an exact hit, the first interval where the count crosses k is refined on a 0.001 grid; if the
   lowest grid resolution already overshoots k, resolutions down to 1e-6 are tried;
3. per k, the lowest resolution giving exactly k clusters is selected; if k is never reached, the run with the
   closest cluster number (lowest resolution on ties) is used and flagged ``matched = False``.
Methods clustered with k-means in the main protocol (cell-type composition) use k-means at each k (seed 0).

Writes ``<FINAL_OUTPUT_FOLDER_PATH>/<method>/<dataset>_<method>_k_grid_scores.csv`` (one row per k) and
``..._k_grid_scan.csv`` (every Leiden run). Run in the reference (banksy) environment (score_k_grid.lsf):
    python score_k_grid.py --index 1          # 1-based index into pairs()
    python score_k_grid.py --method terra --dataset xhs49-1b_1p
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.append(str(Path(__file__).parent))
from bench_utils import FINAL_OUTPUT_FOLDER_PATH, METHODS, NOT_APPLICABLE  # noqa: E402
import scoring  # noqa: E402

DATASETS = ['xhs1000-39b_1p', 'xhs49-1b_1p', 'chl59-8b_1p']   # main-figure datasets without the ISS tract
K_GRID = [5, 10, 15, 20, 25, 30]
RESULTS = Path(FINAL_OUTPUT_FOLDER_PATH)


def pairs():
    """(method, dataset) pairs with a finished result, in METHODS x DATASETS order."""
    return [(m, d) for m in METHODS for d in DATASETS
            if (m, d) not in NOT_APPLICABLE
            and (RESULTS / m / f'{d}_{m}_embedded_clustered.h5ad').exists()]


def leiden_scan(pool, k_grid):
    """All Leiden runs needed to select a resolution for every k (rules in the module docstring)."""
    grid = np.round(np.arange(scoring.GRID_STEP, scoring.SCAN_MAX_RES + scoring.GRID_STEP / 2, scoring.GRID_STEP), 6)
    chunk = pool._processes * 2
    rows = []
    for i in range(0, len(grid), chunk):
        rows += [dict(r, grid='coarse') for r in scoring._run_resolutions(pool, grid[i:i + chunk])]
        counts = [r['n_niches'] for r in rows]
        if len(counts) >= scoring.SCAN_PATIENCE and min(counts[-scoring.SCAN_PATIENCE:]) > max(k_grid):
            break
    df = pd.DataFrame(rows)
    refine, below_grid = set(), False
    for k in k_grid:
        if (df['n_niches'] == k).any():
            continue
        below = df['n_niches'] < k
        cross = np.flatnonzero(below.values[:-1] & ~below.values[1:])
        if len(cross):
            lo = df['leiden_resolution'].iloc[cross[0]]
            refine.update(np.round(np.arange(lo + scoring.FINE_STEP, lo + scoring.GRID_STEP, scoring.FINE_STEP), 6))
        elif df['n_niches'].iloc[0] > k:
            below_grid = True
    refine -= set(df['leiden_resolution'].round(6))
    if refine:
        rows += [dict(r, grid='fine') for r in scoring._run_resolutions(pool, sorted(refine))]
    if below_grid:
        rows += [dict(r, grid='below_grid') for r in scoring._run_resolutions(pool, [1e-3, 1e-4, 1e-5, 1e-6])]
    return pd.DataFrame(rows).sort_values('leiden_resolution').reset_index(drop=True)


def select(scan, k_grid):
    """Per k: lowest resolution with exactly k clusters, else the closest cluster number (flagged)."""
    rows = []
    for k in k_grid:
        hit = scan[scan['n_niches'] == k]
        if len(hit):
            run = hit.loc[hit['leiden_resolution'].idxmin()]
        else:
            dist = (scan['n_niches'] - k).abs()
            run = scan[dist == dist.min()].sort_values('leiden_resolution').iloc[0]
        rows.append(dict(k=k, leiden_resolution=run['leiden_resolution'], n_niches=int(run['n_niches']),
                         NMI=run['NMI'], ARI=run['ARI'], matched=bool(run['n_niches'] == k), grid=run['grid']))
    return pd.DataFrame(rows)


def kmeans_grid(adata, emb_key, k_grid):
    from sklearn.cluster import KMeans
    X = np.asarray(adata.obsm[emb_key], dtype=np.float32)
    rows = []
    for k in k_grid:
        labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)
        rows.append(dict(k=k, leiden_resolution=np.nan, n_niches=int(len(np.unique(labels))),
                         **scoring.niche_metrics(adata.obs['niche_gt'], labels), matched=True, grid='kmeans'))
    return pd.DataFrame(rows)


def main(args):
    scoring.check_environment()
    if args.index is not None:
        todo = pairs()
        if args.index > len(todo):
            print(f'Index {args.index} > {len(todo)} pairs: nothing to do.')
            return
        method, dataset = todo[args.index - 1]
    else:
        method, dataset = args.method, args.dataset
    k_grid = args.k_grid or K_GRID
    prefix = f'{dataset}_{method}'
    run_info = json.load(open(RESULTS / method / f'{prefix}_run_info.json'))
    emb_key = run_info['emb_key']
    clustering = run_info.get('clustering', 'leiden')
    full = sc.read_h5ad(RESULTS / method / f'{prefix}_embedded_clustered.h5ad')
    adata = sc.AnnData(obs=full.obs[['niche_gt']].copy(), obsm={emb_key: np.asarray(full.obsm[emb_key])})
    del full
    print(f'{method} / {dataset}: {adata.n_obs} cells, {emb_key} {adata.obsm[emb_key].shape}, {clustering}', flush=True)

    if clustering == 'kmeans':
        selected = kmeans_grid(adata, emb_key, k_grid)
    else:
        neighbors_key = f'{emb_key}_neighbors'
        sc.pp.neighbors(adata, n_neighbors=scoring.N_LATENT_NEIGHBORS, use_rep=emb_key, key_added=neighbors_key,
                        random_state=0)
        n_workers = max(1, int(os.environ.get('LSB_DJOB_NUMPROC', os.cpu_count() or 1)))
        scoring._SCAN_STATE.update(adata=adata, neighbors_key=neighbors_key)
        try:
            with mp.get_context('fork').Pool(processes=n_workers) as pool:
                scan = leiden_scan(pool, k_grid)
        finally:
            scoring._SCAN_STATE.clear()
        scan.insert(0, 'method', method)
        scan.insert(1, 'dataset', dataset)
        scan.to_csv(RESULTS / method / f'{prefix}_k_grid_scan.csv', index=False)
        selected = select(scan, k_grid)

    selected.insert(0, 'method', method)
    selected.insert(1, 'dataset', dataset)
    selected['n_target'] = run_info.get('n_target_clusters')
    selected.to_csv(RESULTS / method / f'{prefix}_k_grid_scores.csv', index=False)
    print(selected.round(3).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clustering at a common grid of cluster numbers.')
    parser.add_argument('--index', type=int, default=None, help='1-based index into pairs() (array jobs).')
    parser.add_argument('--method')
    parser.add_argument('--dataset')
    parser.add_argument('--k_grid', type=int, nargs='+', default=None, help=f'Cluster numbers (default {K_GRID}).')
    args = parser.parse_args()
    if args.index is None and not (args.method and args.dataset):
        parser.error('use --index or --method and --dataset')
    main(args)
