"""Novae zero-shot niche embeddings for the human niche benchmark.

The embedding step of human_benchmarking/novae_benchmarking.ipynb as a script,
so that it runs on any benchmark dataset: pretrained ``MICS-Lab/novae-human-0``,
Novae's default spatial graph per section (slide), ``compute_representations``
in zero-shot mode and the native domains at the common cluster number
(``assign_domains(level=n_target)``). As for the imported embeddings
(import_fm.py, method ``novae``), the embedding that is clustered is
batch-corrected with Novae's correction based on these domains (paper protocol,
mouse_benchmarking/novae_benchmarking.ipynb), and the native domains are scored
as well (``<dataset>_novae_native_metric_scores.csv``).

The notebook was run with an earlier Novae version (``slide_key`` passed to
``compute_representations``); Novae 1.0.3 takes it in ``spatial_neighbors``.

Writes the embedding only; job_wrapper.sh then clusters and scores it with
score.py. Run in the novae environment (run_method.lsf, PYTHON_ENV=PYTHON_NOVAE). The default
accelerator is the CPU, as in the notebook: the environment's torch needs a
newer NVIDIA driver than the farm and tiger GPU nodes have.
"""
import argparse
import sys
from pathlib import Path

import novae
import numpy as np
import pandas as pd

# pandas 3 (novae environment) would store text as its string dtype, which this
# anndata cannot write; keep text as object dtype as in earlier pandas versions
pd.set_option('future.infer_string', False)
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

sys.path.append(str(Path(__file__).parent))
from bench_utils import (SECTION_KEY, Timer, load_benchmark_dataset,  # noqa: E402
                         method_output_dir, save_embedding)
from import_fm import novae_batch_effect_correction  # noqa: E402

MODEL = 'MICS-Lab/novae-human-0'
EMB_KEY = 'fm_emb'


def object_strings(df):
    """Text columns and index as object dtype.

    pandas 3 (novae environment) stores text as its string dtype, which this
    anndata cannot write and older anndata versions (scoring environment)
    cannot read.
    """
    df = df.copy()
    df.index = pd.Index(np.asarray(df.index, dtype=object), dtype=object)
    for column in df.columns:
        values = df[column]
        if isinstance(values.dtype, pd.CategoricalDtype):
            categories = pd.Index(np.asarray(values.cat.categories, dtype=object), dtype=object)
            df[column] = pd.Categorical(np.asarray(values, dtype=object), categories=categories)
        elif pd.api.types.is_string_dtype(values.dtype):
            df[column] = np.asarray(values, dtype=object)
    return df


def main(args):
    adata = load_benchmark_dataset(args.dataset, subsample=args.subsample)
    dataset = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    adata.uns['benchmark']['dataset_name'] = dataset
    n_target = adata.uns['benchmark']['n_target_clusters']
    multi_section = adata.obs[SECTION_KEY].nunique() > 1
    print(adata)

    with Timer() as t_embed:
        novae.spatial_neighbors(adata, slide_key=SECTION_KEY if multi_section else None)
        model = novae.Novae.from_pretrained(MODEL)
        accelerator = args.accelerator
        model.compute_representations(adata, zero_shot=True, accelerator=accelerator)
        domain_key = model.assign_domains(adata, level=n_target)
    latent = np.asarray(adata.obsm['novae_latent'], dtype=np.float32)

    # Native domains at the target level, scored on all benchmark cells
    domains = adata.obs[domain_key].astype(str).replace({'nan': '-1'}).values
    native = dict(n_niches=int(pd.Series(domains).nunique()),
                  NMI=normalized_mutual_info_score(adata.obs['niche_gt'], domains,
                                                   average_method='arithmetic'),
                  ARI=adjusted_rand_score(adata.obs['niche_gt'], domains))
    pd.DataFrame([dict(method='novae_native', dataset=dataset, n_target=n_target, **native)]).to_csv(
        method_output_dir(args.method) / f'{dataset}_novae_native_metric_scores.csv', index=False)
    print(f'Novae native domains ({domain_key}): {native}')

    adata.obsm['novae_latent'] = latent
    adata.obsm[EMB_KEY] = novae_batch_effect_correction(
        latent, adata.obs[SECTION_KEY].astype(str).values, domains)
    adata.obs['novae_native_domain'] = pd.Categorical(domains)
    shift = np.linalg.norm(adata.obsm[EMB_KEY] - latent, axis=1)
    print(f'Batch-effect correction: {np.mean(shift > 0):.1%} of cells shifted (mean shift {shift.mean():.3f}).')
    # Drop Novae's graphs and per-cell columns that are not needed for scoring
    adata.obsp = {}
    adata.obs = adata.obs.drop(columns=[c for c in adata.obs.columns
                                        if c.startswith('novae_') and c != 'novae_native_domain'])

    adata.obs = object_strings(adata.obs)
    adata.var = object_strings(adata.var)
    save_embedding(adata, args.method, EMB_KEY, dict(
        embedding_seconds=t_embed.seconds, native_novae=native,
        params=dict(model=MODEL, novae_version=novae.__version__, zero_shot=True,
                    spatial_neighbors='novae defaults, per section', domain_level=n_target,
                    accelerator=accelerator,
                    integration=('Novae batch_effect_correction with its native domains at the '
                                 'target level; Leiden on the corrected embedding'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--method', default='novae',
                        help="Result name; e.g. 'novae_rerun' to compare with the imported embeddings.")
    parser.add_argument('--accelerator', default='cpu', help="'cpu' (default) or 'cuda'.")
    parser.add_argument('--subsample', type=int, default=None,
                        help='Random subset of cells (smoke tests only).')
    main(parser.parse_args())
