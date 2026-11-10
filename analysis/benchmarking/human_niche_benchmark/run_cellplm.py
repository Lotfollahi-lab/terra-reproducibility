"""CellPLM zero-shot cell embeddings for the human niche benchmark.

The embedding step of human_benchmarking/cellplm_benchmarking.ipynb as a
script, so that it runs on any benchmark dataset: pretrained checkpoint
``20231027_85M``, ``CellEmbeddingPipeline.predict`` on the raw counts, seed 42.
The embedding is per cell; there is no spatial context. ``--batch_size`` splits
very large sections (Visium HD) into chunks of cells for the forward pass.

Writes the embedding only; job_wrapper.sh then clusters and scores it with
score.py. Run in the cellplm environment on a GPU (run_method.lsf, PYTHON_ENV=PYTHON_CELLPLM).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
from CellPLM.utils import set_seed

sys.path.append(str(Path(__file__).parent))
from bench_utils import SECTION_KEY, Timer, load_benchmark_dataset, save_embedding  # noqa: E402

PRETRAIN_VERSION = '20231027_85M'
CHECKPOINT_FOLDER_PATH = '/nfs/team361/sb75/terra-reproducibility/analysis/benchmarking/CellPLM/ckpt'
EMB_KEY = 'emb'
SEED = 42


def main(args):
    adata = load_benchmark_dataset(args.dataset, subsample=args.subsample)
    dataset = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    adata.uns['benchmark']['dataset_name'] = dataset
    print(adata)

    with Timer() as t_embed:
        set_seed(SEED)
        pipeline = CellEmbeddingPipeline(pretrain_prefix=PRETRAIN_VERSION,
                                         pretrain_directory=CHECKPOINT_FOLDER_PATH)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # Cells are embedded independently; one section at a time bounds memory
        emb = None
        for section in adata.obs[SECTION_KEY].cat.categories:
            mask = (adata.obs[SECTION_KEY] == section).values
            part = np.asarray(pipeline.predict(adata[mask].copy(), device=device,
                                               inference_config=dict(batch_size=args.batch_size)).cpu().numpy(),
                              dtype=np.float32)
            if emb is None:
                emb = np.zeros((adata.n_obs, part.shape[1]), dtype=np.float32)
            emb[mask] = part
            print(f'{section}: {mask.sum()} cells embedded.', flush=True)
        adata.obsm[EMB_KEY] = emb

    save_embedding(adata, args.method, EMB_KEY, dict(
        embedding_seconds=t_embed.seconds,
        params=dict(pretrain_version=PRETRAIN_VERSION, checkpoint_folder=CHECKPOINT_FOLDER_PATH,
                    seed=SEED, device=device, batch_size=args.batch_size, spatial_context='none',
                    integration='none')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--method', default='cellplm',
                        help="Result name; e.g. 'cellplm_rerun' to compare with the imported embeddings.")
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Cells per forward pass (default 0 = whole section; set for > ~300k-cell sections, '
                             'e.g. Visium HD, whose dense input does not fit on the GPU).')
    parser.add_argument('--subsample', type=int, default=None,
                        help='Random subset of cells (smoke tests only).')
    main(parser.parse_args())
