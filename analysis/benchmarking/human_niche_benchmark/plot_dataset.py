"""Figures for one dataset of the human niche benchmark, from the per-method results only.

Reads ``<FINAL_OUTPUT_FOLDER_PATH>/<method>/<dataset>_<method>_*`` for every method with results on the dataset
(the summary tables are not read or changed) and writes to ``<FINAL_OUTPUT_FOLDER_PATH>/summary/figures/<dataset>/``:

- ``niche_{NMI,ARI}_comparison_<dataset>.{svg,pdf,png}``: paper-style bars over the resolution sweep (as in
  benchmark_comparison.ipynb); methods that never reach the target number of niches are marked with a double dagger.
- ``spatial_maps_<dataset>_selected.png``: ground truth and every method's headline clustering for the sections with
  the most ground-truth niches; ``spatial_maps_<dataset>_part<k>.png``: all sections.

Usage (terra environment, via LSF):
    python plot_dataset.py --dataset xhl3001-22b_1p
"""
import argparse
from pathlib import Path

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata.io import read_elem

import plotting as P
from bench_utils import DATASET_LABELS, FINAL_OUTPUT_FOLDER_PATH, METHODS, NOT_APPLICABLE

RESULTS = Path(FINAL_OUTPUT_FOLDER_PATH)


def read_obs_and_coords(path):
    """Only obs and obsm['spatial'] of a result file (the embeddings are not needed)."""
    with h5py.File(path, 'r') as f:
        obs = read_elem(f['obs'])
        coords = read_elem(f['obsm/spatial'])
    if 'sample' in obs:  # sample names as section titles
        obs['section'] = obs['sample'].astype(str).astype('category')
    adata = ad.AnnData(obs=obs)
    adata.obsm['spatial'] = coords
    return adata


def spatial_figure(adatas, sections, gt_colors, dataset_label, path, max_cells):
    n_cols = len(adatas) + 1
    fig, axes = plt.subplots(len(sections), n_cols, figsize=(1.9 * n_cols, 2.0 * len(sections)), squeeze=False)
    for row, section in zip(axes, sections):
        P.plot_spatial_panels(adatas, section, row, gt_colors, max_cells=max_cells)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=c, label=n) for n, c in gt_colors.items()]
    handles.append(plt.Line2D([], [], marker='o', linestyle='', color=P.UNMATCHED, label='unmatched cluster'))
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False,
               title=f'{dataset_label}: ground-truth niches', title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {path}', flush=True)


def main(args):
    dataset, label = args.dataset, DATASET_LABELS[args.dataset]
    out = RESULTS / 'summary' / 'figures' / dataset
    out.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHODS if (RESULTS / m / f'{dataset}_{m}_metric_scores.csv').exists()
               and (m, dataset) not in NOT_APPLICABLE]
    scans = pd.concat([pd.read_csv(RESULTS / m / f'{dataset}_{m}_metric_scores.csv') for m in methods])
    sweep = scans[scans['in_sweep']]
    head = scans[scans['headline']].set_index('method')
    unreachable = {(m, dataset): '‡' for m in methods
                   if int(head.loc[m, 'n_niches']) != int(head.loc[m, 'n_target'])}
    table = pd.DataFrame({
        'method': [METHODS[m]['label'] for m in methods],
        'sweep_NMI': [sweep.loc[sweep.method == m, 'NMI'].mean() for m in methods],
        'sweep_ARI': [sweep.loc[sweep.method == m, 'ARI'].mean() for m in methods],
        'headline_NMI': head.loc[methods, 'NMI'].values, 'headline_ARI': head.loc[methods, 'ARI'].values,
        'headline_n_clusters': head.loc[methods, 'n_niches'].values,
        'target': head.loc[methods, 'n_target'].values})
    table.to_csv(out / f'scores_{dataset}.csv', index=False)
    print(table.round(3).to_string(index=False), flush=True)
    print('Target not reached:', [m for m, _ in unreachable], flush=True)

    with plt.rc_context(P.PAPER_RC):
        for metric in ['NMI', 'ARI']:
            fig = P.plot_paper_metric_comparison(sweep, metric, {dataset: label}, method_order=methods,
                                                 figsize_per_panel=(0.38 * len(methods) + 1.2, 2.6),
                                                 annotations=unreachable)
            for ext in ['svg', 'pdf', 'png']:
                fig.savefig(out / f'niche_{metric}_comparison_{dataset}.{ext}')
            plt.close(fig)

    adatas = {METHODS[m]['label'] + (' ‡' if (m, dataset) in unreachable else ''):
              read_obs_and_coords(RESULTS / m / f'{dataset}_{m}_embedded_clustered.h5ad') for m in methods}
    ref = next(iter(adatas.values()))
    gt = ref.obs['niche_gt'].astype(str)
    gt_colors = P.niche_colors(gt.value_counts().index.tolist())
    sections = ref.obs['section'].astype(str)
    # Sections with the most ground-truth niches (> 100 cells in the section) first
    n_niches = (pd.crosstab(sections, gt) > 100).sum(1).sort_values(ascending=False)
    print('Ground-truth niches per section:\n', n_niches.to_string(), flush=True)
    selected = n_niches.index[:args.n_selected].tolist()
    spatial_figure(adatas, selected, gt_colors, label, out / f'spatial_maps_{dataset}_selected.png', args.max_cells)
    ordered = sorted(sections.unique())
    for k, start in enumerate(range(0, len(ordered), args.sections_per_page)):
        spatial_figure(adatas, ordered[start:start + args.sections_per_page], gt_colors, label,
                       out / f'spatial_maps_{dataset}_part{k + 1}.png', args.max_cells)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bars and spatial maps for one dataset.')
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASET_LABELS))
    parser.add_argument('--n_selected', type=int, default=4, help='Sections in the selected spatial-map figure.')
    parser.add_argument('--sections_per_page', type=int, default=8)
    parser.add_argument('--max_cells', type=int, default=60000, help='Plotting-only subsample per section.')
    main(parser.parse_args())
