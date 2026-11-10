"""Check scoring.scan_resolutions / cluster_and_score on a toy embedding.

For a small synthetic dataset the 0.01 grid is clustered by brute force with
the same Leiden settings, and the scan's results are checked against it:
- every scanned run has the same cluster count and NMI/ARI as brute force;
- headline = lowest grid resolution giving exactly the target;
- matched = exactly the grid runs (in the scanned range) giving the target;
- sweep window = +/- 4 grid steps around the median matching resolution;
- obs['predicted_niche'] is the headline clustering;
- an unreachable target (every point its own component) falls back to the
  best reachable number below the grid.
Exits non-zero on any failure.

    python checks/test_resolution_scan.py     (banksy env)
"""
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import scoring as B  # noqa: E402

failures = []


def check(cond, message):
    print(('PASS ' if cond else 'FAIL ') + message)
    if not cond:
        failures.append(message)


# Toy data: 6 Gaussian "niches" of different sizes and spreads in 10 dims
rng = np.random.default_rng(1)
sizes, centres = [400, 300, 250, 200, 150, 120], rng.normal(scale=4, size=(6, 10))
X = np.vstack([rng.normal(c, 1 + 0.3 * i, size=(n, 10))
               for i, (c, n) in enumerate(zip(centres, sizes))]).astype(np.float32)
labels = np.repeat([f'n{i}' for i in range(6)], sizes)
adata = ad.AnnData(np.zeros((len(X), 1)), obs=pd.DataFrame(
    {'niche_gt': pd.Categorical(labels)}, index=[f'c{i}' for i in range(len(X))]))
adata.obsm['emb'] = X
target = 6
adata.uns['benchmark'] = dict(dataset_name='toy', n_target_clusters=target)

with tempfile.TemporaryDirectory() as out_dir:
    scan = B.cluster_and_score(adata, 'toy', 'emb', Path(out_dir))

# Brute force on the same graph for every scanned grid resolution
coarse = scan[scan['grid'] == 'coarse'].copy()
brute = []
for res in coarse['leiden_resolution']:
    n = B._leiden(adata, 'emb_neighbors', res, '_bf')
    brute.append(dict(leiden_resolution=res, n_bf=n,
                      **{f'{k}_bf': v for k, v in
                         B.niche_metrics(adata.obs['niche_gt'], adata.obs['_bf']).items()}))
brute = pd.DataFrame(brute)
merged = coarse.merge(brute, on='leiden_resolution')
check((merged['n_niches'] == merged['n_bf']).all(),
      'cluster counts of every scanned run equal brute force')
check(np.allclose(merged['NMI'], merged['NMI_bf']) and np.allclose(merged['ARI'], merged['ARI_bf']),
      'NMI/ARI of every scanned run equal brute force')

hits = merged[merged['n_bf'] == target]['leiden_resolution']
check(len(hits) > 0, f'toy target ({target}) is reached on the grid')
head = scan[scan['headline']]
check(len(head) == 1 and np.isclose(head['leiden_resolution'].iloc[0], hits.min()),
      'headline = lowest grid resolution giving the target')
check(sorted(scan[scan['matched']]['leiden_resolution']) == sorted(hits),
      'matched = all scanned runs giving the target')
centre = max(B.GRID_STEP, round(hits.median() / B.GRID_STEP) * B.GRID_STEP)
expected = [round(centre + k * B.GRID_STEP, 6) for k in range(-4, 5)
            if centre + k * B.GRID_STEP > B.GRID_STEP / 2]
check(np.allclose(sorted(scan[scan['in_sweep']]['leiden_resolution']), expected),
      'sweep = +/- 4 grid steps around the median matching resolution')
check(adata.obs['predicted_niche'].nunique() == target,
      'obs["predicted_niche"] is the headline clustering')

# Unreachable target: duplicated points form many disconnected components
X2 = np.repeat(rng.normal(size=(40, 5)), 20, axis=0).astype(np.float32)
adata2 = ad.AnnData(np.zeros((len(X2), 1)), obs=pd.DataFrame(
    {'niche_gt': pd.Categorical(np.repeat(['a', 'b'], len(X2) // 2))},
    index=[f'd{i}' for i in range(len(X2))]))
adata2.obsm['emb'] = X2
adata2.uns['benchmark'] = dict(dataset_name='toy_unreachable', n_target_clusters=2)
with tempfile.TemporaryDirectory() as out_dir:
    scan2 = B.cluster_and_score(adata2, 'toy', 'emb', Path(out_dir))
head2 = scan2[scan2['headline']].iloc[0]
check(not scan2['matched'].any() and head2['n_niches'] > 2,
      'unreachable target: no match, headline = best reachable run')
check((scan2['grid'] == 'below_grid').any(),
      'unreachable target: resolutions below the grid were tried')

print('ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED')
sys.exit(0 if not failures else 1)
