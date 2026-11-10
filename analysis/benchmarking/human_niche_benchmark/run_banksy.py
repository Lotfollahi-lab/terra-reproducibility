"""BANKSY (+ Harmony) niche embeddings for the human niche benchmark.

Settings mirror analysis/benchmarking/mouse_benchmarking/banksy_benchmarking.ipynb
(raw counts, 10 spatial neighbours, scaled-gaussian decay, max_m=1, PCA to 20
dims, Harmony on the section), except lambda: pyBANKSY has no built-in default,
so we use BANKSY's recommended value for domain/niche segmentation (0.8; 0.2 is
the recommendation for cell typing). The mouse run used 1.0, which gives the
cell's own expression zero weight. One deliberate difference: the BANKSY
neighbour graph is built per tissue section (block-diagonal), so no cell gets
neighbours from another section. The BANKSY matrix itself (z-scoring, lambda
weighting, PCA) is computed jointly over all sections, as in the mouse run.

``--n_top_genes`` (whole-transcriptome datasets, e.g. Visium HD): keep only the
highly variable genes (scanpy ``seurat`` on log-normalised counts, sections as
``batch_key``) before BANKSY, as in BANKSY's own preprocessing; the dense
BANKSY matrix of all ~18,000 genes does not fit in memory for > 1M bins.

Writes the embedding only; job_wrapper.sh then clusters and scores it with
score.py. Run in the banksy environment (run_method.lsf, PYTHON_ENV=PYTHON_BANKSY).
"""
import argparse
import sys
from pathlib import Path

import harmonypy
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from banksy.embed_banksy import generate_banksy_matrix
from banksy.initialize_banksy import initialize_banksy
from banksy_utils.umap_pca import pca_umap

sys.path.append(str(Path(__file__).parent))
from bench_utils import (N_NEIGHS, SECTION_KEY, Timer,  # noqa: E402
                         load_benchmark_dataset, save_embedding)

METHOD = 'banksy'
EMB_KEY = 'banksy_latent'
MAX_M = 1
NBR_WEIGHT_DECAY = 'scaled_gaussian'
LAMBDA = 0.8   # BANKSY recommendation for domain segmentation
PCA_DIMS = 20
COORD_KEYS = ('spatial_x', 'spatial_y', 'spatial')


def per_section_weights(adata):
    """BANKSY weight graphs built per section, assembled block-diagonally."""
    sections = adata.obs[SECTION_KEY].values
    # Rows of each section must be contiguous for a block-diagonal assembly.
    boundaries = np.flatnonzero(sections[1:] != sections[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    if len(starts) != adata.obs[SECTION_KEY].nunique():
        raise ValueError('Sections are not stored contiguously.')

    weights = {m: [] for m in range(MAX_M + 1)}
    for start, end in zip(starts, np.append(starts[1:], adata.n_obs)):
        adata_section = adata[start:end]
        print(f'Building BANKSY graph for section '
              f'{sections[start]} ({end - start} cells)...')
        section_dict = initialize_banksy(
            adata_section, COORD_KEYS, N_NEIGHS,
            nbr_weight_decay=NBR_WEIGHT_DECAY, max_m=MAX_M,
            plt_edge_hist=False, plt_nbr_weights=False,
            plt_agf_angles=False, plt_theta=False)
        for m in range(MAX_M + 1):
            weights[m].append(section_dict[NBR_WEIGHT_DECAY]['weights'][m])

    return {NBR_WEIGHT_DECAY: {'weights': {
        m: sp.block_diag(weights[m], format='csr')
        for m in range(MAX_M + 1)}}}


def main(args):
    adata = load_benchmark_dataset(args.dataset, subsample=args.subsample)
    dataset = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    adata.uns['benchmark']['dataset_name'] = dataset
    print(adata)

    n_genes_input = adata.n_vars
    if args.n_top_genes and adata.n_vars > args.n_top_genes:
        # 'seurat' flavour on log-normalised counts (seurat_v3 needs scikit-misc, absent in the banksy env);
        # the selected genes keep their raw counts
        lognorm = sc.AnnData(X=adata.X.copy(), obs=adata.obs[[SECTION_KEY]].copy(), var=adata.var[[]].copy())
        sc.pp.normalize_total(lognorm, target_sum=1e4)
        sc.pp.log1p(lognorm)
        sc.pp.highly_variable_genes(
            lognorm, flavor='seurat', n_top_genes=args.n_top_genes,
            batch_key=SECTION_KEY if adata.obs[SECTION_KEY].nunique() > 1 else None)
        adata = adata[:, lognorm.var['highly_variable'].values].copy()
        del lognorm
        print(f'HVG: kept {adata.n_vars} of {n_genes_input} genes.')

    adata.X = adata.X.astype(np.float32)   # raw counts, as in the mouse run
    adata.obs['spatial_x'] = adata.obsm['spatial'][:, 0]
    adata.obs['spatial_y'] = adata.obsm['spatial'][:, 1]

    with Timer() as t_embed:
        banksy_dict = per_section_weights(adata)
        banksy_dict, _ = generate_banksy_matrix(
            adata, banksy_dict, [LAMBDA], MAX_M, verbose=False)
        pca_umap(banksy_dict, pca_dims=[PCA_DIMS], add_umap=False,
                 plt_remaining_var=False)
        adata.obsm[EMB_KEY] = np.asarray(
            banksy_dict[NBR_WEIGHT_DECAY][LAMBDA]['adata'].obsm[
                f'reduced_pc_{PCA_DIMS}'])
        del banksy_dict

        emb_key = EMB_KEY
        if adata.obs[SECTION_KEY].nunique() > 1:
            harmony_out = harmonypy.run_harmony(
                adata.obsm[EMB_KEY].astype('float64'), adata.obs, SECTION_KEY,
                random_state=0)
            emb_key = f'{EMB_KEY}_harmony'
            adata.obsm[emb_key] = np.asarray(harmony_out.Z_corr)
            if adata.obsm[emb_key].shape[0] != adata.n_obs:  # harmonypy<0.0.10
                adata.obsm[emb_key] = adata.obsm[emb_key].T

    save_embedding(adata, METHOD, emb_key, dict(
        embedding_seconds=t_embed.seconds,
        params=dict(n_neighs=N_NEIGHS, max_m=MAX_M,
                    nbr_weight_decay=NBR_WEIGHT_DECAY, lambda_param=LAMBDA,
                    pca_dims=PCA_DIMS, input='raw counts',
                    n_top_genes=args.n_top_genes, n_genes_used=adata.n_vars,
                    integration='harmony (post-hoc, forced)')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--n_top_genes', type=int, default=None,
                        help='Highly variable genes before BANKSY (default: all genes).')
    parser.add_argument('--subsample', type=int, default=None,
                        help='Random subset of cells (smoke tests only).')
    main(parser.parse_args())
