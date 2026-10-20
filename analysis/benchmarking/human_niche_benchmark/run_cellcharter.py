"""CellCharter niche embeddings for the human niche benchmark.

Mirrors analysis/benchmarking/mouse_benchmarking/cellcharter_benchmarking.ipynb:
scVI (default settings, section as batch covariate, early stopping) on raw
counts, 10-NN spatial graph per section (with self loop), then
``cc.gr.aggregate_neighbors`` with 3 layers on the scVI latent. Integration is
native (scVI batch covariate), no post-hoc alignment.

Writes the embedding only; job_wrapper.sh then clusters and scores it with
score.py. Run in the cellcharter environment on a GPU node (run_gpu_method.lsf).
"""
import argparse
import sys
from pathlib import Path

import cellcharter as cc
import numpy as np
import scanpy as sc
import scvi
import squidpy as sq
from lightning.pytorch import seed_everything

sys.path.append(str(Path(__file__).parent))
from bench_utils import (N_NEIGHS, SECTION_KEY, Timer,  # noqa: E402
                         load_benchmark_dataset, save_embedding)

METHOD = 'cellcharter'
EMB_KEY = 'X_cellcharter'
N_LAYERS = 3
SEED = 12345   # as in the mouse notebook


def main(args):
    seed_everything(SEED)
    scvi.settings.seed = SEED

    adata = load_benchmark_dataset(args.dataset, subsample=args.subsample)
    dataset = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    adata.uns['benchmark']['dataset_name'] = dataset
    print(adata)

    with Timer() as t_embed:
        adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata)

        scvi.model.SCVI.setup_anndata(adata, layer='counts',
                                      batch_key=SECTION_KEY)
        model = scvi.model.SCVI(adata)
        model.train(early_stopping=True, enable_progress_bar=False)
        adata.obsm['X_scVI'] = model.get_latent_representation(
            adata).astype(np.float32)

        sq.gr.spatial_neighbors(adata, library_key=SECTION_KEY,
                                coord_type='generic', spatial_key='spatial',
                                n_neighs=N_NEIGHS, set_diag=True)
        cc.gr.aggregate_neighbors(adata, n_layers=N_LAYERS, use_rep='X_scVI',
                                  out_key=EMB_KEY, sample_key=SECTION_KEY)
        adata.obsm[EMB_KEY] = np.asarray(adata.obsm[EMB_KEY], dtype=np.float32)

    save_embedding(adata, METHOD, EMB_KEY, dict(
        embedding_seconds=t_embed.seconds,
        params=dict(n_neighs=N_NEIGHS, n_layers=N_LAYERS, seed=SEED,
                    scvi='defaults, batch_key=section, early_stopping=True',
                    scvi_epochs_trained=int(model.history['elbo_train'].shape[0]),
                    integration='native (scVI batch covariate)')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--subsample', type=int, default=None,
                        help='Random subset of cells (smoke tests only).')
    main(parser.parse_args())
