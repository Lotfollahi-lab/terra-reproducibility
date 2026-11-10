"""TERRA zero-shot niche embeddings for the human niche benchmark.

Embeds the benchmark cells with a TERRA model bundle via
``harmonize_tokenize_embed_pipeline`` (as in
human_benchmarking/terra_benchmarking.ipynb, Section 2) and evaluates
``neighborhood_emb`` with the shared clustering/scoring protocol, so TERRA is
clustered exactly like every baseline (common cluster number, lowest-resolution
tie rule). Each section is tokenized separately (sample_key), so spatial
neighbourhoods never cross sections.

``--preprocess hvg`` (method ``terra_hvg``) keeps only the ``--n_top_genes``
highly variable genes (scanpy ``seurat_v3`` on raw counts, sections as
``batch_key``), chosen among the genes TERRA tokenizes (mapped to an Ensembl
ID and seen in pretraining), before tokenization; panels with at most ``--n_top_genes``
genes are left unchanged. Motivation (diagnostics/token_diagnostic.py): on the
Xenium 5k panels about half of the cells express more genes than TERRA's
per-cell budget of 256 tokens, and the cut-off falls among count-1 genes, so
which of them are kept is arbitrary.

``--model`` selects the model bundle (TERRA_MODELS): ``96m``
(TERRA-96M, default; methods ``terra`` / ``terra_hvg``) or ``112m``
(TERRA-112M; methods ``terra_112m`` / ``terra_112m_hvg``) or ``noneigh``
(TERRA-96M retrained with the neighbour cells masked; method ``terra_noneigh``).

``--random_init`` (method ``terra_random``) skips loading the checkpoint: the
same architecture is built by TERRA's own ``init_model`` (seeded) and used
untrained, as a capacity control.

Writes the embedding only; job_wrapper.sh then clusters and scores it with
score.py. Run in the terra environment on a GPU node (run_method.lsf, PYTHON_ENV=PYTHON_TERRA).
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import terra
from terra import harmonize_tokenize_embed_pipeline

# Checkout the terra package is imported from (recorded with its commit)
TERRA_SOURCE = Path(terra.__file__).resolve().parents[2]

sys.path.append(str(Path(__file__).parent))
from bench_utils import (CACHE_FOLDER_PATH, SECTION_KEY,  # noqa: E402
                         Timer, git_commit,
                         load_benchmark_dataset, save_embedding)

# Published TERRA models (Hugging Face Lotfollahi-lab/TERRA-96M / TERRA-112M;
# the local bundles are byte-identical, checked by SHA-256). TERRA-96M was
# trained with held-out data and is the benchmark's main TERRA model;
# TERRA-112M was trained on the full corpus (no held-out split) and has a
# 'batch' meta token (padded at zero-shot inference).
TERRA_MODELS = {
    '96m': '/nfs/team361/sb75/terra-reproducibility/artifacts/models/01112025_081731_614',
    '112m': '/nfs/team361/sb75/terra-reproducibility/artifacts/models/11012026_190057_836',
    # TERRA-96M retrained with all neighbour segments masked (epoch-2 checkpoint,
    # same budget as the 96m bundle); see README.txt in the folder.
    'noneigh': '/nfs/team361/me13/terra_models/96m_no_neighbours_ep2',
}

EMB_KEY = 'neighborhood_emb'


def method_name(model, preprocess, random_init=False):
    name = 'terra' if model == '96m' else f'terra_{model}'
    if random_init:
        name += '_random'
    return name if preprocess == 'none' else f'{name}_{preprocess}'


def skip_checkpoint_loading():
    """Make the TERRA pipeline keep the randomly initialized target encoder."""
    import terra.inference.embed as embed_module

    def load_no_checkpoint(**kwargs):
        print('random_init: checkpoint NOT loaded (untrained target encoder).')
        return None, None, kwargs['target_encoder'], None, None, 0, None

    embed_module.load_checkpoint = load_no_checkpoint


# Minimum number of pretraining occurrences for a gene to be tokenized
# (harmonize_tokenize_embed_pipeline's gene_occurrence_count_filter_value)
MIN_PRETRAINING_OCCURRENCES = 10


def terra_vocabulary_genes(var_names, model_folder_path):
    """Genes TERRA keeps when harmonizing (terra.inference.harmonize.harmonize_adata).

    A gene is kept if it maps to an Ensembl ID with the bundle's gene mapping
    and occurred more than MIN_PRETRAINING_OCCURRENCES times in pretraining
    (bundle's gene_count_dictionary.pkl); all other genes are dropped before
    tokenization.
    """
    folder = Path(model_folder_path)
    with open(folder / 'ensembl_dictionary.pkl', 'rb') as f:
        name_to_ensembl = pickle.load(f)
    with open(folder / 'gene_count_dictionary.pkl', 'rb') as f:
        occurrences = pickle.load(f)
    ensembl_ids = set(name_to_ensembl.values())
    keep = []
    for gene in var_names:
        ensembl = name_to_ensembl.get(gene, gene if gene in ensembl_ids else None)
        keep.append(ensembl is not None
                    and occurrences.get(ensembl, 0) > MIN_PRETRAINING_OCCURRENCES)
    return np.array(keep)


def select_hvg(adata, n_top_genes, batch_key):
    """Subset to highly variable genes (seurat_v3 on raw counts)."""
    if adata.n_vars <= n_top_genes:
        print(f'Panel has {adata.n_vars} <= {n_top_genes} genes: no selection.')
        return adata, dict(n_genes_before=adata.n_vars, n_genes_after=adata.n_vars)
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top_genes,
                                batch_key=batch_key)
    selected = adata.var_names[adata.var['highly_variable']]
    print(f'HVG: kept {len(selected)} of {adata.n_vars} genes.')
    info = dict(n_genes_before=adata.n_vars, n_genes_after=len(selected))
    adata = adata[:, selected].copy()
    adata.var = adata.var.drop(columns=[c for c in adata.var.columns
                                        if c.startswith(('highly_variable', 'means',
                                                         'variances', 'dispersions'))])
    return adata, info


def main(args):
    method = method_name(args.model, args.preprocess, args.random_init)
    if args.random_init:
        skip_checkpoint_loading()
    model_folder_path = TERRA_MODELS[args.model]
    print(f'Method {method}: model {model_folder_path}')
    adata = load_benchmark_dataset(args.dataset, subsample=args.subsample)
    dataset = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    adata.uns['benchmark']['dataset_name'] = dataset
    benchmark_uns = dict(adata.uns['benchmark'])
    cell_ids = adata.obs_names.copy()
    print(adata)
    print(f'CUDA available: {torch.cuda.is_available()}')

    multi_section = adata.obs[SECTION_KEY].nunique() > 1
    preprocess_info = {}
    if args.preprocess == 'hvg':
        # HVGs only among the genes TERRA tokenizes; otherwise the
        # harmonization drops part of the selection afterwards (e.g. 1,039 of
        # 2,000 HVGs on Visium HD LUAD)
        in_vocabulary = terra_vocabulary_genes(adata.var_names.astype(str), model_folder_path)
        print(f'{in_vocabulary.sum()} of {adata.n_vars} genes in the TERRA vocabulary.')
        n_panel = adata.n_vars
        adata = adata[:, in_vocabulary].copy()
        adata, preprocess_info = select_hvg(
            adata, args.n_top_genes, SECTION_KEY if multi_section else None)
        preprocess_info.update(n_genes_panel=n_panel, hvg_among='TERRA vocabulary genes')
        preprocess_info.update(flavor='seurat_v3', n_top_genes=args.n_top_genes,
                               batch_key=SECTION_KEY if multi_section else None)
    cache_dir = Path(CACHE_FOLDER_PATH) / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)

    with Timer() as t_embed:
        adata_emb = harmonize_tokenize_embed_pipeline(
            adata=adata.copy(),
            sample_key=SECTION_KEY if multi_section else None,
            batch_key='terra_batch',
            model_folder_path=model_folder_path,
            cache_directory_path=str(cache_dir),
            min_cells_per_gene=0,
            batch_size=args.batch_size)

    # The pipeline regroups cells by section; put them back in benchmark order
    # and check that no cell was dropped.
    if set(adata_emb.obs_names) != set(cell_ids):
        raise ValueError(f'TERRA pipeline changed the cell set: '
                         f'{adata_emb.n_obs} vs {len(cell_ids)} cells.')
    adata_emb = adata_emb[cell_ids]
    for key in ['cell_emb', 'neighborhood_emb', 'spatial_cell_emb']:
        if key in adata_emb.obsm:
            adata.obsm[key] = np.asarray(adata_emb.obsm[key])
    adata.uns['benchmark'] = benchmark_uns
    del adata_emb

    save_embedding(adata, method, EMB_KEY, dict(
        embedding_seconds=t_embed.seconds,
        params=dict(model=args.model, model_folder_path=model_folder_path,
                    random_init=args.random_init,
                    terra_source=f'{TERRA_SOURCE} @ {git_commit(TERRA_SOURCE)}',
                    sample_key=SECTION_KEY if multi_section else None,
                    batch_size=args.batch_size, integration='none (zero-shot)',
                    preprocess=args.preprocess, **preprocess_info)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--subsample', type=int, default=None,
                        help='Random subset of cells (smoke tests only).')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model', choices=sorted(TERRA_MODELS), default='96m',
                        help='TERRA model bundle (TERRA_MODELS).')
    parser.add_argument('--preprocess', choices=['none', 'hvg'], default='none',
                        help="'hvg': highly variable genes before tokenization.")
    parser.add_argument('--n_top_genes', type=int, default=2000)
    parser.add_argument('--random_init', action='store_true',
                        help='untrained model of the same architecture (method terra_random)')
    main(parser.parse_args())
