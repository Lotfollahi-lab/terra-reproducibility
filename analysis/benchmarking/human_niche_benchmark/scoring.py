"""Clustering and scoring for the human niche benchmark (single implementation).

Every method is clustered and scored here and only here (via score.py), in
one reference environment (the banksy environment), so all results share the
same igraph / leidenalg / scanpy / scikit-learn / scib-metrics versions.
Importing this module in any other environment raises an error; method
scripts only compute embeddings (bench_utils.save_embedding).
"""
import itertools
import os
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.metrics as skm
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from bench_utils import N_LATENT_NEIGHBORS, SECTION_KEY

# Package versions of the reference (banksy) environment, see envs/banksy.txt
REFERENCE_VERSIONS = {
    'igraph': '1.0.0',
    'leidenalg': '0.11.0',
    'scanpy': '1.11.5',
    'scikit-learn': '1.8.0',
    'scib-metrics': '0.5.9',
    'pynndescent': '0.6.0',
}


def environment_versions() -> dict:
    return {pkg: version(pkg) for pkg in REFERENCE_VERSIONS}


def check_environment() -> None:
    found = environment_versions()
    wrong = {pkg: (found[pkg], ref) for pkg, ref in REFERENCE_VERSIONS.items()
             if found[pkg] != ref}
    if wrong:
        raise RuntimeError(
            'Scoring must run in the reference (banksy) environment; version '
            'mismatches (found, expected): '
            + ', '.join(f'{p} {f} != {r}' for p, (f, r) in wrong.items()))


check_environment()

# ---------------------------------------------------------------------------
# Leiden resolution scan
# ---------------------------------------------------------------------------
GRID_STEP = 0.01          # one resolution grid for headline, sweep and matching
FINE_STEP = 0.001         # only used if no grid point hits the target
SCAN_PATIENCE = 5         # stop after this many consecutive steps above target
SCAN_MAX_RES = 5.0
N_SWEEP_STEPS = 4         # sweep = +/- 4 grid steps around the matching range

_SCAN_STATE = {}


def _leiden(adata, neighbors_key, resolution, key_added):
    sc.tl.leiden(adata,
                 neighbors_key=neighbors_key,
                 key_added=key_added,
                 resolution=resolution,
                 flavor='igraph',
                 n_iterations=2,
                 random_state=0)
    return adata.obs[key_added].nunique()


def _scan_worker(resolution):
    """Leiden at one resolution; returns cluster count and NMI/ARI (forked)."""
    adata = _SCAN_STATE['adata']
    key = f'_scan_{resolution}'
    n = _leiden(adata, _SCAN_STATE['neighbors_key'], resolution, key)
    scores = niche_metrics(adata.obs['niche_gt'], adata.obs[key])
    del adata.obs[key]
    return dict(leiden_resolution=float(resolution), n_niches=int(n), **scores)


def _run_resolutions(pool, resolutions):
    return pool.map(_scan_worker, [round(float(r), 6) for r in resolutions])


def scan_resolutions(adata, neighbors_key, n_target, n_workers=None):
    """Single upward Leiden scan on a 0.01 grid; every run is kept and scored.

    - Resolutions 0.01, 0.02, ... are clustered (in parallel, in order) until
      the cluster count has been above the target for ``SCAN_PATIENCE``
      consecutive steps (counts are not monotonic in the resolution, so one
      overshoot is not enough) or ``SCAN_MAX_RES`` is reached.
    - If no grid point gives exactly the target, the first interval where the
      count crosses the target is refined on a 0.001 grid.
    - If the target is still not reached and even the lowest grid resolution
      overshoots (graph split into more components than the target), lower
      resolutions down to 1e-6 are tried (best reachable).
    The ±``N_SWEEP_STEPS`` grid points around the median matching resolution
    are always included. Returns a DataFrame with one row per run.
    """
    import multiprocessing as mp

    if n_workers is None:
        n_workers = int(os.environ.get('LSB_DJOB_NUMPROC', os.cpu_count() or 1))
    n_workers = max(1, n_workers)
    chunk = n_workers * 2
    _SCAN_STATE.update(adata=adata, neighbors_key=neighbors_key)
    rows = []
    try:
        with mp.get_context('fork').Pool(processes=n_workers) as pool:
            # 1. Upward scan on the grid
            grid = np.round(np.arange(GRID_STEP, SCAN_MAX_RES + GRID_STEP / 2,
                                      GRID_STEP), 6)
            for i in range(0, len(grid), chunk):
                rows += [dict(r, grid='coarse')
                         for r in _run_resolutions(pool, grid[i:i + chunk])]
                counts = [r['n_niches'] for r in rows]
                if (len(counts) >= SCAN_PATIENCE
                        and min(counts[-SCAN_PATIENCE:]) > n_target):
                    break
            df = pd.DataFrame(rows)

            # 2. Refine the first crossing if the grid never hits the target
            if not (df['n_niches'] == n_target).any():
                below = df['n_niches'] < n_target
                cross = np.flatnonzero(below.values[:-1] & ~below.values[1:])
                if len(cross):
                    lo = df['leiden_resolution'].iloc[cross[0]]
                    fine = np.round(np.arange(lo + FINE_STEP,
                                              lo + GRID_STEP, FINE_STEP), 6)
                    rows += [dict(r, grid='fine')
                             for r in _run_resolutions(pool, fine)]
                elif df['n_niches'].iloc[0] > n_target:
                    # 3. Best reachable below the grid (e.g. disconnected graph)
                    rows += [dict(r, grid='below_grid') for r in
                             _run_resolutions(pool, [1e-3, 1e-4, 1e-5, 1e-6])]
                df = pd.DataFrame(rows)

            # 4. Make sure the full sweep window exists on the grid
            matched = df[df['n_niches'] == n_target]['leiden_resolution']
            centre = (matched.median() if len(matched)
                      else df.loc[(df['n_niches'] - n_target).abs().idxmin(),
                                  'leiden_resolution'])
            centre = max(GRID_STEP, round(centre / GRID_STEP) * GRID_STEP)
            window = [round(centre + k * GRID_STEP, 6)
                      for k in range(-N_SWEEP_STEPS, N_SWEEP_STEPS + 1)
                      if centre + k * GRID_STEP > GRID_STEP / 2]
            missing = [r for r in window
                       if not np.isclose(df['leiden_resolution'], r).any()]
            if missing:
                rows += [dict(r, grid='coarse')
                         for r in _run_resolutions(pool, missing)]
                df = pd.DataFrame(rows)
    finally:
        _SCAN_STATE.clear()

    df = df.sort_values('leiden_resolution').reset_index(drop=True)
    df['n_target'] = n_target
    df['matched'] = df['n_niches'] == n_target
    df['in_sweep'] = df['leiden_resolution'].apply(
        lambda r: any(np.isclose(r, w) for w in window))
    if df['matched'].any():
        head_idx = df[df['matched']]['leiden_resolution'].idxmin()
    else:
        dist = (df['n_niches'] - n_target).abs()
        head_idx = df[dist == dist.min()]['leiden_resolution'].idxmin()
        print(f'Target {n_target} not reachable; best reachable: '
              f'{df.loc[head_idx, "n_niches"]} clusters at resolution '
              f'{df.loc[head_idx, "leiden_resolution"]}.')
    df['headline'] = False
    df.loc[head_idx, 'headline'] = True
    return df


def cluster_and_score(adata: ad.AnnData,
                      method: str,
                      emb_key: str,
                      out_dir: Path) -> pd.DataFrame:
    """Cluster an embedding at the common cluster number and score it.

    1. kNN graph (15) on the embedding.
    2. One upward Leiden scan on a 0.01 grid (``scan_resolutions``); every run
       is scored (NMI/ARI) and kept.
    3. From that single table:
       - headline = lowest grid resolution giving exactly the common cluster
         number (``obs['predicted_niche']``);
       - matched = all resolutions giving exactly the common cluster number;
       - sweep = +/- 4 grid steps (0.01) around the median matching
         resolution (centred on the matching range, as the mouse benchmark's
         sweep was centred on a typical matching resolution).
    Writes ``{dataset}_{method}_metric_scores.csv`` with flags ``headline``,
    ``matched``, ``in_sweep`` and returns it.
    """
    dataset_name = adata.uns['benchmark']['dataset_name']
    n_target = adata.uns['benchmark']['n_target_clusters']
    neighbors_key = f'{emb_key}_neighbors'

    sc.pp.neighbors(adata, n_neighbors=N_LATENT_NEIGHBORS, use_rep=emb_key,
                    key_added=neighbors_key, random_state=0)

    metrics_df = scan_resolutions(adata, neighbors_key, n_target)
    head = metrics_df[metrics_df['headline']].iloc[0]
    _leiden(adata, neighbors_key, head['leiden_resolution'], 'predicted_niche')

    metrics_df.insert(0, 'method', method)
    metrics_df.insert(1, 'dataset', dataset_name)
    metrics_df.to_csv(out_dir / f'{dataset_name}_{method}_metric_scores.csv',
                      index=False)
    return metrics_df


N_KMEANS_SEEDS = 10       # k-means runs (seeds) replace the resolution sweep


def kmeans_and_score(adata: ad.AnnData,
                     method: str,
                     emb_key: str,
                     out_dir: Path) -> pd.DataFrame:
    """k-means at the common cluster number instead of the Leiden scan.

    For methods whose canonical clustering is k-means (cellular-neighbourhood
    composition, Schuerch et al. 2020): ``N_KMEANS_SEEDS`` runs with
    k = common cluster number (scikit-learn defaults, random_state = seed).
    Every run is "matched" and "in_sweep"; seed 0 is the headline
    (``obs['predicted_niche']``). Writes the same table as
    ``cluster_and_score`` (``leiden_resolution`` = NaN, extra column ``seed``).
    """
    from sklearn.cluster import KMeans

    dataset_name = adata.uns['benchmark']['dataset_name']
    n_target = adata.uns['benchmark']['n_target_clusters']
    X = np.asarray(adata.obsm[emb_key], dtype=np.float32)
    rows = []
    for seed in range(N_KMEANS_SEEDS):
        labels = KMeans(n_clusters=n_target, random_state=seed).fit_predict(X)
        if seed == 0:
            adata.obs['predicted_niche'] = pd.Categorical(labels.astype(str))
        rows.append(dict(leiden_resolution=np.nan, seed=seed,
                         n_niches=int(len(np.unique(labels))),
                         **niche_metrics(adata.obs['niche_gt'], labels)))
    df = pd.DataFrame(rows)
    df['grid'] = 'kmeans'
    df['n_target'] = n_target
    df['matched'] = df['n_niches'] == n_target
    df['in_sweep'] = True
    df['headline'] = df['seed'] == 0
    df.insert(0, 'method', method)
    df.insert(1, 'dataset', dataset_name)
    df.to_csv(out_dir / f'{dataset_name}_{method}_metric_scores.csv', index=False)
    return df


CLUSTERING = {'leiden': cluster_and_score, 'kmeans': kmeans_and_score}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def niche_metrics(labels_true, labels_pred) -> dict:
    return dict(
        NMI=skm.normalized_mutual_info_score(labels_true, labels_pred,
                                             average_method='arithmetic'),
        ARI=skm.adjusted_rand_score(labels_true, labels_pred))


def _mmd(X_a, X_b, sigma, n_sub=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = X_a[rng.choice(len(X_a), min(n_sub, len(X_a)), replace=False)]
    b = X_b[rng.choice(len(X_b), min(n_sub, len(X_b)), replace=False)]

    def rbf(A, B):
        # Pairwise squared distances without materialising an n x m x dim array
        # (the latter took > 10 h for 512-dim embeddings on 55 section pairs).
        d2 = cdist(A, B, metric='sqeuclidean')
        return np.exp(-d2 / (2 * sigma ** 2))

    return rbf(a, a).mean() + rbf(b, b).mean() - 2 * rbf(a, b).mean()


def batch_integration_metrics(adata, emb_key, seed=0) -> pd.DataFrame:
    """iLISI, batch ASW and MMD as in the mouse notebooks.

    The mouse MMD compared exactly two batches; with more sections it is
    averaged over all section pairs.
    """
    from scib_metrics import ilisi_knn
    from scib_metrics.nearest_neighbors import pynndescent

    X = np.asarray(adata.obsm[emb_key], dtype=np.float32)
    batch = adata.obs[SECTION_KEY].values
    results = []

    knn = pynndescent(X, n_neighbors=90)
    results.append(dict(metric='iLISI',
                        score=float(np.mean(ilisi_knn(knn, batch)))))
    results.append(dict(metric='ASW', score=float(silhouette_score(
        X, batch, metric='euclidean',
        sample_size=min(10000, len(X)), random_state=seed))))

    X_std = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(seed)
    sub = X_std[rng.choice(len(X_std), min(1000, len(X_std)), replace=False)]
    dists = cdist(sub, sub)
    sigma = np.median(dists[dists > 0])
    sections = adata.obs[SECTION_KEY].cat.categories
    mmds = [_mmd(X_std[batch == a], X_std[batch == b], sigma, seed=seed)
            for a, b in itertools.combinations(sections, 2)]
    results.append(dict(metric='MMD', score=float(np.mean(mmds))))
    return pd.DataFrame(results)
