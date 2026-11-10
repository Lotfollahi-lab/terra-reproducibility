"""Import existing foundation-model embeddings for scoring.

The embeddings were computed earlier by the human_benchmarking notebooks
(zero-shot, versions/checkpoints as in the paper) and are only read here:
    <FM_EMBEDDINGS_FOLDER_PATH>/<Method>/<dataset>_embedded.h5ad
Cells are matched to the benchmark cells by ``cell_id``; coverage is recorded.
Writes the embedding only; job_wrapper.sh then clusters and scores it with
score.py, like every other method. For Novae, its own
domain assignment at the target number (``novae_domains_<n>`` in the
``_embedded_clustered.h5ad`` written by novae_benchmarking.ipynb) is scored as
well. The Novae embedding that is clustered is batch-corrected with Novae's
own correction based on these domains, exactly as in the paper's mouse
benchmark (mouse_benchmarking/novae_benchmarking.ipynb).

    python import_fm.py --method novae --dataset xhs1000-39b_1p   (run_method.lsf, PYTHON_ENV=PYTHON_BANKSY, EXTRA_ARGS="--method novae")
"""
import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from bench_utils import (SECTION_KEY, load_benchmark_dataset,  # noqa: E402
                         method_output_dir, save_embedding)

# Existing zero-shot embeddings: <Method>/<dataset>_embedded.h5ad
FM_EMBEDDINGS_FOLDER_PATH = '/nfs/team361/sb75/terra-reproducibility/artifacts/benchmarking'

FM = {
    'novae': dict(folder='Novae', emb_key='novae_latent'),
    'scgpt-spatial': dict(folder='scGPT-spatial', emb_key='X'),
    'nicheformer': dict(folder='nicheformer', emb_key='X_niche_embeddings'),
    'cellplm': dict(folder='CellPLM', emb_key='emb'),
}
EMB_KEY = 'fm_emb'


def match_keys(fm, adata):
    """Keys to match foundation-model cells to benchmark cells.

    Preferably ``cell_id``. Some outputs were saved without it and keep the
    original silver obs_names instead; these are only unique within a section,
    so the section (``batch`` like '78batch0' -> '78_batch0') is prepended when
    the file has several sections.
    """
    if 'cell_id' in fm.obs:
        return (fm.obs['cell_id'].astype(str).values,
                adata.obs_names.astype(str).values, 'cell_id')
    fm_names = fm.obs_names.astype(str).values
    our_names = adata.obs['orig_obs_name'].astype(str).values
    if 'batch' in fm.obs and adata.obs[SECTION_KEY].nunique() > 1:
        fm_section = fm.obs['batch'].astype(str).str.replace(
            r'^(\d+)batch', r'\1_batch', regex=True).values
        our_section = adata.obs[SECTION_KEY].astype(str).values
        return (np.char.add(np.char.add(fm_section.astype(str), '|'), fm_names),
                np.char.add(np.char.add(our_section.astype(str), '|'), our_names),
                'section + original obs_name')
    return fm_names, our_names, 'original obs_name'


def novae_batch_effect_correction(emb, sections, domains):
    """Novae's batch-effect correction as patched in the paper's notebook.

    Re-implements ``_patched_batch_effect_correction`` from
    ``mouse_benchmarking/novae_benchmarking.ipynb`` (sections = Novae slides):
    for each domain, the reference section is the one with the most cells of
    that domain (first in section order on ties, as ``idxmax``); in every other
    section the domain's cells are shifted so that their centroid matches the
    reference centroid. Cells without a domain are left unchanged. With a single
    section the embedding is returned unchanged.
    """
    corrected = emb.copy()
    section_order = list(dict.fromkeys(sections))
    valid = pd.notna(domains) & (domains != '-1')
    counts = (pd.crosstab(sections[valid], domains[valid])
              .reindex(section_order).fillna(0))
    counts = counts.loc[:, (counts > 0).any(axis=0)]
    ref_sections = counts.idxmax(axis=0)
    for domain, ref_section in ref_sections.items():
        ref_mask = (sections == ref_section) & (domains == domain)
        centroid_ref = emb[ref_mask].mean(0)
        for section in section_order:
            if section == ref_section:
                continue
            mask = (sections == section) & (domains == domain)
            if mask.sum() == 0:
                continue
            corrected[mask] += centroid_ref - emb[mask].mean(0)
    return corrected


def novae_native_and_correction(adata, fm_path, dataset, out_dir):
    """Score Novae's native domains and replace the embedding by the corrected one."""
    # Imported here so checks/test_novae_correction.py can import this module
    # in the novae environment (scoring only loads in the reference env).
    from scoring import niche_metrics
    n_target = adata.uns['benchmark']['n_target_clusters']
    clustered_path = fm_path.with_name(f'{dataset}_embedded_clustered.h5ad')
    key = f'novae_domains_{n_target}'
    fm_cl = ad.read_h5ad(clustered_path, backed='r')
    if key not in fm_cl.obs:
        raise ValueError(f'No {key} in {clustered_path}: cannot run the '
                         'paper protocol (correction needs native domains).')
    domains = pd.Series(fm_cl.obs[key].astype(str).values,
                        index=fm_cl.obs['cell_id'].astype(str).values)
    domains = domains.replace({'nan': '-1'})
    adata.obs['novae_native_domain'] = pd.Categorical(
        domains.reindex(adata.obs['fm_key'].values).fillna('-1').values)
    native = dict(n_niches=adata.obs['novae_native_domain'].nunique(),
                  **niche_metrics(adata.obs['niche_gt'],
                                  adata.obs['novae_native_domain']))
    pd.DataFrame([dict(method='novae_native', dataset=dataset,
                       n_target=n_target, **native)]).to_csv(
        out_dir / f'{dataset}_novae_native_metric_scores.csv', index=False)
    print(f'Novae native domains ({key}): {native}')

    adata.obsm['novae_latent'] = adata.obsm[EMB_KEY].copy()
    adata.obsm[EMB_KEY] = novae_batch_effect_correction(
        adata.obsm[EMB_KEY],
        adata.obs[SECTION_KEY].astype(str).values,
        adata.obs['novae_native_domain'].astype(str).values)
    shift = np.linalg.norm(adata.obsm[EMB_KEY] - adata.obsm['novae_latent'], axis=1)
    print(f'Batch-effect correction: {np.mean(shift > 0):.1%} of cells shifted '
          f'(mean shift {shift.mean():.3f}).')
    return native


def main(args):
    cfg = FM[args.method]
    out_dir = method_output_dir(args.method)
    adata = load_benchmark_dataset(args.dataset)
    n_benchmark = adata.n_obs

    fm_path = Path(FM_EMBEDDINGS_FOLDER_PATH) / cfg['folder'] / f'{args.dataset}_embedded.h5ad'
    fm = ad.read_h5ad(fm_path, backed='r')
    fm_keys, our_keys, match_on = match_keys(fm, adata)
    if not pd.Index(fm_keys).is_unique:
        raise ValueError(f'Match keys ({match_on}) not unique in {fm_path}.')
    emb = pd.DataFrame(np.asarray(fm.obsm[cfg['emb_key']], dtype=np.float32),
                       index=fm_keys)
    print(f'Matching cells on: {match_on}')

    adata.obs['fm_key'] = our_keys
    matched = adata.obs['fm_key'].isin(emb.index).values
    coverage = float(matched.mean())
    print(f'{args.method} {args.dataset}: {matched.sum()} / {n_benchmark} '
          f'benchmark cells have an embedding ({coverage:.2%}).')
    adata = adata[matched].copy()
    adata.obsm[EMB_KEY] = emb.loc[adata.obs['fm_key'].values].values
    # The common cluster number stays the one of the full benchmark cell set.

    native = None
    if args.method == 'novae':
        native = novae_native_and_correction(adata, fm_path, args.dataset, out_dir)

    save_embedding(adata, args.method, EMB_KEY, dict(
        source_file=str(fm_path), source_emb_key=cfg['emb_key'],
        n_benchmark_cells=n_benchmark, coverage=coverage, native_novae=native,
        params=dict(integration=(
            'Novae batch_effect_correction with its native domains at the target '
            'level (as in mouse_benchmarking/novae_benchmarking.ipynb); Leiden on '
            'the corrected embedding' if args.method == 'novae'
            else 'as in the original zero-shot run'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=list(FM))
    parser.add_argument('--dataset', required=True)
    main(parser.parse_args())
