"""scGPT-spatial zero-shot cell embeddings for the human niche benchmark.

The embedding step of human_benchmarking/scgpt-spatial_benchmarking.ipynb as a
script, so that it runs on any benchmark dataset: checkpoint
``scGPT_spatial_v1``, ``scgpt_spatial.tasks.embed_data`` on the dense raw
counts with gene symbols as ``feature_name``, batch size 64. The embedding is
per cell; there is no spatial context.

Environment: the scgpt-spatial venv does not import scgpt_spatial (its
flash-attn 1.0.4 is built for CUDA 11, its torch for CUDA 12); the scgpt venv
has the combination of envs/scgpt-spatial.yaml (torch 2.0.1+cu117,
flash-attn 1.0.4) and is used instead (PYTHON_SCGPT_SPATIAL in config.sh).

Writes the embedding only; job_wrapper.sh then clusters and scores it with
score.py. Run on a GPU (run_gpu_method.lsf).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

SCGPT_SPATIAL_FOLDER_PATH = '/nfs/team361/sb75/terra-reproducibility/analysis/benchmarking/scGPT-spatial'
sys.path.insert(0, SCGPT_SPATIAL_FOLDER_PATH)
import scgpt_spatial  # noqa: E402

sys.path.append(str(Path(__file__).parent))
from bench_utils import SECTION_KEY, Timer, load_benchmark_dataset, save_embedding  # noqa: E402

MODEL_FOLDER_PATH = f'{SCGPT_SPATIAL_FOLDER_PATH}/checkpoints/scGPT_spatial_v1'
GENE_COL = 'feature_name'
EMB_KEY = 'X_scgpt_spatial'


def main(args):
    adata = load_benchmark_dataset(args.dataset, subsample=args.subsample)
    dataset = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    adata.uns['benchmark']['dataset_name'] = dataset
    print(adata)

    with Timer() as t_embed:
        # Cells are embedded independently, so sections are embedded one at a
        # time (the dense count matrix of a whole Visium HD dataset does not
        # fit in memory)
        parts = []
        for section in adata.obs[SECTION_KEY].cat.categories:
            mask = (adata.obs[SECTION_KEY] == section).values
            adata_in = adata[mask].copy()
            adata_in.var[GENE_COL] = adata_in.var_names.astype(str)
            if sp.issparse(adata_in.X):
                adata_in.X = adata_in.X.astype(np.float32).toarray()
            embedded = scgpt_spatial.tasks.embed_data(
                adata_in, MODEL_FOLDER_PATH, gene_col=GENE_COL, obs_to_save=[],
                batch_size=args.batch_size, return_new_adata=True)
            if embedded.n_obs != adata_in.n_obs:
                raise ValueError(f'scGPT-spatial embedded {embedded.n_obs} of {adata_in.n_obs} cells.')
            parts.append((np.where(mask)[0], np.asarray(embedded.X, dtype=np.float32)))
            print(f'{section}: {mask.sum()} cells embedded.', flush=True)
            del adata_in, embedded
        emb = np.zeros((adata.n_obs, parts[0][1].shape[1]), dtype=np.float32)
        for idx, part in parts:
            emb[idx] = part
        adata.obsm[EMB_KEY] = emb

    save_embedding(adata, args.method, EMB_KEY, dict(
        embedding_seconds=t_embed.seconds,
        params=dict(model_folder=MODEL_FOLDER_PATH, gene_col=GENE_COL, batch_size=args.batch_size,
                    spatial_context='none', integration='none')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--method', default='scgpt-spatial',
                        help="Result name; e.g. 'scgpt-spatial_rerun' to compare with the imported embeddings.")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--subsample', type=int, default=None,
                        help='Random subset of cells (smoke tests only).')
    main(parser.parse_args())
