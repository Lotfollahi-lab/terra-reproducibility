"""Neighbourhood cell-type composition baseline for the human niche benchmark.

For every cell, the fraction of each annotated cell type among the cell and
its 10 nearest spatial neighbours (per-section kNN graph with self loop, the
same graph as the neighbourhood-expression PCA baseline). The composition
vector is the embedding; clustering and scoring are the shared protocol.

Cell types come from ``DATASETS[...]['cell_type_levels']`` in bench_utils
(``--level fine`` = finest annotation, the main method
``cell_type_composition``; coarser levels -> ``cell_type_composition_<level>``,
sensitivity check). Cells without an annotation count as their own
'unlabelled' type. Datasets without cell-type annotations (ISS reproductive
tract) are not applicable.

Note: unlike the other methods, this baseline uses expert cell-type labels,
and annotated niches are often defined from such compositions, so it can
reproduce part of the ground truth by construction.

``--clustering kmeans`` (method ``..._kmeans``) clusters the composition
vectors with k-means at the common cluster number, the canonical clustering
of cellular neighbourhoods (Schuerch et al., Cell 2020); the default is the
shared Leiden scan. Composition vectors are discrete, so many cells share
identical vectors and the Leiden kNN graph splits into cliques.

Writes the embedding only; job_wrapper.sh then clusters and scores it with
score.py. Run in the banksy environment (run_method.lsf, PYTHON_ENV=PYTHON_BANKSY).
"""
import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import squidpy as sq

sys.path.append(str(Path(__file__).parent))
from bench_utils import (DATASETS, N_NEIGHS, SECTION_KEY, Timer,  # noqa: E402
                         load_benchmark_dataset, save_embedding)

EMB_KEY = 'cell_type_composition'
UNLABELLED = 'unlabelled'


def cell_type_labels(adata, dataset, level):
    """Cell-type label per benchmark cell at the requested annotation level."""
    cfg = DATASETS[dataset]
    levels = cfg.get('cell_type_levels')
    if not levels:
        raise ValueError(f'{dataset} has no cell-type annotation: baseline not applicable.')
    if level not in levels:
        raise ValueError(f'{dataset}: level {level!r} not available ({list(levels)}).')
    key = levels[level]
    if 'cell_type_source' in cfg:
        source = ad.read_h5ad(cfg['cell_type_source'], backed='r')
        labels = pd.Series(source.obs[key].astype(str).values,
                           index=source.obs['cell_id'].astype(str).values)
        source.file.close()
        labels = labels.reindex(adata.obs['original_cell_id'].astype(str).values)
        if labels.isna().any():
            raise ValueError(f'{labels.isna().sum()} benchmark cells not found in '
                             f'{cfg["cell_type_source"]}.')
        labels = labels.values
    else:
        labels = adata.obs[key].astype(object).values
    labels = pd.Series(labels).fillna(UNLABELLED).astype(str)
    labels[labels.isin(['nan', 'None', ''])] = UNLABELLED
    return key, labels.values


def main(args):
    method = EMB_KEY if args.level == 'fine' else f'{EMB_KEY}_{args.level}'
    if args.clustering != 'leiden':
        method += f'_{args.clustering}'
    adata = load_benchmark_dataset(args.dataset, subsample=args.subsample)
    dataset = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    adata.uns['benchmark']['dataset_name'] = dataset
    print(adata)

    key, labels = cell_type_labels(adata, args.dataset, args.level)
    types = sorted(set(labels))
    print(f'Cell types ({key}, level {args.level}): {len(types)}; '
          f'unlabelled cells: {np.mean(labels == UNLABELLED):.1%}')

    with Timer() as t_embed:
        sq.gr.spatial_neighbors(adata, library_key=SECTION_KEY,
                                coord_type='generic', spatial_key='spatial',
                                n_neighs=N_NEIGHS, set_diag=True)
        W = (adata.obsp['spatial_connectivities'] > 0).astype(np.float32)
        W = sp.diags(1 / np.asarray(W.sum(1)).ravel()) @ W
        codes = pd.Categorical(labels, categories=types).codes
        one_hot = sp.csr_matrix((np.ones(len(codes), dtype=np.float32),
                                 (np.arange(len(codes)), codes)),
                                shape=(len(codes), len(types)))
        adata.obsm[EMB_KEY] = np.asarray((W @ one_hot).todense(), dtype=np.float32)
        adata.obs['cell_type_label'] = pd.Categorical(labels)
        adata.obsp = {}

    save_embedding(adata, method, EMB_KEY, dict(
        clustering=args.clustering,
        embedding_seconds=t_embed.seconds,
        n_cell_types=len(types), cell_types=types,
        frac_unlabelled=float(np.mean(labels == UNLABELLED)),
        params=dict(cell_type_key=key, level=args.level, n_neighs=N_NEIGHS,
                    self_loop=True, per_section=True,
                    source=DATASETS[args.dataset].get('cell_type_source', 'silver'),
                    integration='none')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--level', default='fine',
                        help="annotation level in DATASETS[...]['cell_type_levels']")
    parser.add_argument('--clustering', choices=['leiden', 'kmeans'], default='leiden',
                        help='clustering used by score.py (kmeans = canonical for this baseline)')
    parser.add_argument('--subsample', type=int, default=None,
                        help='Random subset of cells (smoke tests only).')
    main(parser.parse_args())
