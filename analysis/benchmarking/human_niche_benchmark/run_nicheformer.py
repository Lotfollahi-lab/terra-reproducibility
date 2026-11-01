"""Nicheformer zero-shot cell embeddings for the human niche benchmark.

Follows Nicheformer's own recipe (theislab/nicheformer):
notebooks/experiments/get_embeddings.ipynb for inference and
notebooks/tokenization/xenium_human_*.ipynb for the technology mean.
It corrects two deviations of the earlier embeddings
(human_benchmarking/nicheformer_benchmarking.ipynb, imported by import_fm.py
as method ``nicheformer``):

- Technology mean: undefined (NaN) for genes outside that technology's
  pretraining panels (e.g. 1,735 of 20,310 genes defined for Xenium).
  NicheformerDataset divides by it as is, so every unmeasured gene becomes NaN,
  counts as expressed and fills each cell's 1,500-token sequence with the same
  tail of genes. Here NaN -> 0 -> 1 and values are rounded, as in the authors'
  tokenization notebooks.
- Context tokens: modality, species and assay are passed to the model
  (``metadata_fields``); without them get_embeddings drops the first three
  gene tokens instead of the context tokens.

The embedding is the mean over the last layer's gene-token outputs of a single
cell; there is no spatial context.

Writes the embedding only; job_wrapper.sh then clusters and scores it with
score.py. Run in the nicheformer environment on a GPU (run_gpu_method.lsf).
"""
import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nicheformer.data import NicheformerDataset
from nicheformer.models import Nicheformer

sys.path.append(str(Path(__file__).parent))
from bench_utils import (DATASETS, SILVER_FOLDER_PATH, Timer,  # noqa: E402
                         load_benchmark_dataset, save_embedding)

METHOD = 'nicheformer_rerun'
EMB_KEY = 'X_nicheformer'
NICHEFORMER_DATA_PATH = ('/nfs/team361/sb75/terra-reproducibility/analysis/'
                         'benchmarking/nicheformer/data')
MAX_SEQ_LEN = 1500
AUX_TOKENS = 30
MODALITY_SPATIAL = 4
SPECIE_HUMAN = 5
# Assay tokens and technology means from the authors' tokenization notebooks
# ("cosmx": 8, "Xenium": 9). ISS has neither; Xenium (closest targeted
# imaging panel) is used, as in the earlier embeddings.
TECHNOLOGY = {
    'xhs1000-39b_1p': ('xenium', 9),
    'xhs49-1b_1p': ('xenium', 9),
    'chr78-11b_1p': ('xenium', 9),
    'chl59-8b_1p': ('cosmx', 8),
}


def technology_mean(technology):
    """Technology mean as prepared in the authors' tokenization notebooks."""
    mean = np.nan_to_num(np.load(f'{NICHEFORMER_DATA_PATH}/model_means/'
                                 f'{technology}_mean_script.npy'))
    rounded = np.where((mean % 1) >= 0.5, np.ceil(mean), np.floor(mean))
    return np.where(mean == 0, 1, rounded)


def ensembl_ids(dataset_name, gene_names):
    """Ensembl IDs of the panel genes, from the silver files' var.

    load_benchmark_dataset concatenates sections with ad.concat, which drops
    var columns, so they are read from the source files here.
    """
    folder = Path(SILVER_FOLDER_PATH) / dataset_name
    mapping = {}
    for file in sorted(folder.glob('*.h5ad')):
        if not any(p in file.name for p in DATASETS[dataset_name]['file_patterns']):
            continue
        var = ad.read_h5ad(file, backed='r').var
        for name, ensembl in zip(var.index.astype(str), var['ensembl_id'].astype(str)):
            if mapping.setdefault(name, ensembl) != ensembl:
                raise ValueError(f'{name} has two Ensembl IDs across sections.')
    return np.array([mapping[g] for g in gene_names])


def to_model_genes(adata, ensembl, model_genes):
    """Counts in the model's gene order (Ensembl IDs); unmeasured genes are 0."""
    pos = {g: i for i, g in enumerate(model_genes)}
    keep = np.array([g in pos for g in ensembl])
    cols = np.array([pos[g] for g in ensembl[keep]])
    if len(set(cols)) != len(cols):
        raise ValueError('Duplicate Ensembl IDs in the panel.')
    X = sp.csr_matrix(adata.X)[:, np.where(keep)[0]].tocoo()
    X = sp.csr_matrix((X.data, (X.row, cols[X.col])),
                      shape=(adata.n_obs, len(model_genes)), dtype=np.float32)
    return X, int(keep.sum())


def main(args):
    adata = load_benchmark_dataset(args.dataset, subsample=args.subsample)
    dataset = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    adata.uns['benchmark']['dataset_name'] = dataset
    print(adata)
    technology, assay_token = TECHNOLOGY[args.dataset]

    with Timer() as t_embed:
        model_genes = ad.read_h5ad(f'{NICHEFORMER_DATA_PATH}/model_means/model.h5ad').var_names
        ensembl = ensembl_ids(args.dataset, adata.var_names.astype(str))
        X, n_genes_used = to_model_genes(adata, ensembl, model_genes)
        print(f'{n_genes_used} of {adata.n_vars} panel genes in the Nicheformer vocabulary.')
        tokens_in = ad.AnnData(X=X)
        tokens_in.obs['modality'] = MODALITY_SPATIAL
        tokens_in.obs['specie'] = SPECIE_HUMAN
        tokens_in.obs['assay'] = assay_token
        tokens_in.obs['nicheformer_split'] = 'train'
        dataset_nf = NicheformerDataset(
            adata=tokens_in, technology_mean=technology_mean(technology),
            split='train', max_seq_len=MAX_SEQ_LEN, aux_tokens=AUX_TOKENS,
            chunk_size=1000,
            metadata_fields={'obs': ['modality', 'specie', 'assay']})
        loader = DataLoader(dataset_nf, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

        torch.manual_seed(0)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Nicheformer.load_from_checkpoint(
            checkpoint_path=f'{NICHEFORMER_DATA_PATH}/nicheformer.ckpt',
            strict=False).to(device).eval()
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                embeddings.append(model.get_embeddings(batch=batch, layer=-1).cpu().numpy())
        adata.obsm[EMB_KEY] = np.concatenate(embeddings).astype(np.float32)

    save_embedding(adata, METHOD, EMB_KEY, dict(
        embedding_seconds=t_embed.seconds,
        n_genes_in_vocabulary=n_genes_used,
        params=dict(technology_mean=technology, assay_token=assay_token,
                    modality_token=MODALITY_SPATIAL, specie_token=SPECIE_HUMAN,
                    max_seq_len=MAX_SEQ_LEN, embedding_layer=-1,
                    technology_mean_nan='-> 1 (authors tokenization notebooks)',
                    context_tokens=True, spatial_context='none',
                    integration='none')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--subsample', type=int, default=None,
                        help='Random subset of cells (smoke tests only).')
    main(parser.parse_args())
