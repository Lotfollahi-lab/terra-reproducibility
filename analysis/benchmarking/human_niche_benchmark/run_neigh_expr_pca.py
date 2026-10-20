"""Neighbourhood-expression PCA baseline for the human niche benchmark.

Mirrors analysis/benchmarking/mouse_benchmarking/neigh_expr_pca_benchmarking.ipynb:
10-NN spatial graph per section (with self loop), summed neighbourhood counts,
normalize_total(1e4) + log1p, PCA to 20 dims. No integration.

Writes the embedding only; job_wrapper.sh then clusters and scores it with
score.py. Run in the banksy environment (run_cpu_methods.lsf); any
scanpy/squidpy environment works.
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

METHOD = 'neigh_expr_pca'
EMB_KEY = 'X_pca'
N_PCS = 20


def main(args):
    adata = load_benchmark_dataset(args.dataset, subsample=args.subsample)
    dataset = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    adata.uns['benchmark']['dataset_name'] = dataset
    print(adata)

    with Timer() as t_embed:
        # Per-section kNN graph (library_key keeps sections separate).
        sq.gr.spatial_neighbors(adata, library_key=SECTION_KEY,
                                coord_type='generic', spatial_key='spatial',
                                n_neighs=N_NEIGHS, set_diag=True)
        adata.layers['X_neigh'] = (
            adata.obsp['spatial_connectivities'].T @ adata.X).astype(np.float32)
        sc.pp.normalize_total(adata, target_sum=1e4, layer='X_neigh')
        sc.pp.log1p(adata, layer='X_neigh')
        sc.pp.pca(adata, layer='X_neigh', n_comps=N_PCS, random_state=0)

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
