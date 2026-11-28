"""Neighbourhood-expression PCA baseline for the human niche benchmark.

Mirrors analysis/benchmarking/mouse_benchmarking/neigh_expr_pca_benchmarking.ipynb:
10-NN spatial graph per section (with self loop), summed neighbourhood counts,
normalize_total(1e4) + log1p, PCA to 20 dims. No integration.

Also the template for new methods (README.md, "Adding a method"): load the
benchmark cells with ``load_benchmark_dataset``, put one embedding per cell in
``adata.obsm``, finish with ``save_embedding``. Clustering and scoring follow
automatically (job_wrapper.sh -> score.py). Run in the banksy environment:
    SCRIPT=run_neigh_expr_pca.py PYTHON_ENV=PYTHON_BANKSY DATASET=xhs1000-39b_1p \\
        bsub -q normal -n 8 -M 64GB -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]" \\
        -W 12:00 < run_method.lsf
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import squidpy as sq

sys.path.append(str(Path(__file__).parent))
from bench_utils import (N_NEIGHS, SECTION_KEY, Timer,  # noqa: E402
                         load_benchmark_dataset, save_embedding)

METHOD = 'neigh_expr_pca'   # key in bench_utils.METHODS = result folder name
EMB_KEY = 'X_pca'           # obsm key of the embedding that is clustered
N_PCS = 20


def main(args):
    # Benchmark cells: raw counts in X, obs['section'], obs['niche_gt'] (never
    # used by the method), obsm['spatial'] in um, obs_names = cell_id
    adata = load_benchmark_dataset(args.dataset, subsample=args.subsample)
    dataset = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    adata.uns['benchmark']['dataset_name'] = dataset
    print(adata)

    with Timer() as t_embed:
        # Spatial graph per section (library_key): no cross-section neighbours
        sq.gr.spatial_neighbors(adata, library_key=SECTION_KEY,
                                coord_type='generic', spatial_key='spatial',
                                n_neighs=N_NEIGHS, set_diag=True)
        adata.layers['X_neigh'] = (
            adata.obsp['spatial_connectivities'].T @ adata.X).astype(np.float32)
        sc.pp.normalize_total(adata, target_sum=1e4, layer='X_neigh')
        sc.pp.log1p(adata, layer='X_neigh')
        sc.pp.pca(adata, layer='X_neigh', n_comps=N_PCS, random_state=0)

    # Writes <method>/<dataset>_<method>_embedded.h5ad and _run_info.json; the
    # dict is stored in run_info (parameters, timings; add 'clustering': 'kmeans'
    # to cluster with k-means instead of the Leiden scan)
    save_embedding(adata, METHOD, EMB_KEY, dict(
        embedding_seconds=t_embed.seconds,
        params=dict(n_neighs=N_NEIGHS, target_sum=1e4, n_pcs=N_PCS,
                    integration='none')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--subsample', type=int, default=None,
                        help='Random subset of cells (smoke tests only).')
    main(parser.parse_args())
