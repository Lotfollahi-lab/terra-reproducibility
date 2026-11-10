"""Figures for niche identification at a fixed range of Leiden resolutions (score_fixed_resolutions.py).

Reads ``<method>/<dataset>_<method>_fixed_resolution_scores.csv`` for the main-figure datasets except the ISS
reproductive tract and writes to
``<FINAL_OUTPUT_FOLDER_PATH>/summary/``:
- ``fixed_resolution.csv``: every run (method, dataset, resolution, number of clusters, NMI, ARI);
- ``figures/fixed_resolution/niche_{NMI,ARI}_fixed_resolution.{svg,pdf,png}``: paper-style bars as
  niche_{NMI,ARI}_comparison_human (bar = mean over the fixed resolutions, error bar = 1 SD, dots = resolutions);
- ``figures/fixed_resolution/{NMI,ARI,n_clusters}_vs_resolution.{svg,png}``: one line per method and dataset; the
  dashed line in the cluster-number panels is the ground-truth niche number.

Usage (terra environment):
    python plot_fixed_resolutions.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).parent))
import plot_results as P  # noqa: E402
from bench_utils import DATASET_LABELS, FINAL_OUTPUT_FOLDER_PATH, METHODS  # noqa: E402

RESULTS = Path(FINAL_OUTPUT_FOLDER_PATH)
# Main-figure datasets of score_fixed_resolutions.py without the ISS reproductive tract (left out here)
DATASETS = ['xhs1000-39b_1p', 'xhs49-1b_1p', 'chl59-8b_1p']
OUT = RESULTS / 'summary' / 'figures' / 'fixed_resolution'
# Clustered with Leiden here, not k-means
P.METHODS['cell_type_composition_kmeans'] = dict(P.METHODS['cell_type_composition_kmeans'],
                                                 label='Neigh. Cell-Type Comp.')


def collect():
    frames = [pd.read_csv(path) for m in METHODS for d in DATASETS
              for path in [RESULTS / m / f'{d}_{m}_fixed_resolution_scores.csv'] if path.exists()]
    return pd.concat(frames, ignore_index=True)


def curves(df, metric, methods, labels, path):
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 3.2), sharey=metric != 'n_niches')
    for ax, (dataset, label) in zip(axes, labels.items()):
        sub = df[df['dataset'] == dataset]
        for m in methods:
            run = sub[sub['method'] == m].sort_values('leiden_resolution')
            if run.empty:
                continue
            is_terra = m == 'terra'
            ax.plot(run['leiden_resolution'], run[metric], marker='o', markersize=3,
                    linewidth=2.2 if is_terra else 1, color=METHODS[m]['color'],
                    label=P.METHODS[m]['label'], zorder=3 if is_terra else 2)
        if metric == 'n_niches' and len(sub):
            ax.axhline(sub['n_target'].iloc[0], color='0.4', linestyle='--', linewidth=0.8)
            ax.set_yscale('log')
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Leiden resolution')
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel({'n_niches': 'Number of clusters'}.get(metric, metric))
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ['svg', 'png']:
        fig.savefig(path.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(RESULTS / 'summary' / 'fixed_resolution.csv', index=False)
    labels = {d: DATASET_LABELS[d] for d in DATASETS}
    methods = [m for m in METHODS if m in set(df['method'])]
    summary = (df.groupby(['method', 'dataset'])[['NMI', 'ARI']].mean().unstack('dataset')
               .reindex(methods).round(3))
    summary[('NMI', 'mean')] = df.groupby('method')['NMI'].mean().reindex(methods).round(3)
    summary[('ARI', 'mean')] = df.groupby('method')['ARI'].mean().reindex(methods).round(3)
    print(summary.to_string())

    with plt.rc_context(P.PAPER_RC):
        for metric in ['NMI', 'ARI']:
            fig = P.plot_paper_metric_comparison(df, metric, labels, method_order=methods,
                                                 figsize_per_panel=(0.24 * len(methods) + 0.6, 2.6))
            for ext in ['svg', 'pdf', 'png']:
                fig.savefig(OUT / f'niche_{metric}_fixed_resolution.{ext}')
            plt.close(fig)
    for metric in ['NMI', 'ARI', 'n_niches']:
        curves(df, metric, methods, labels, OUT / f'{"n_clusters" if metric == "n_niches" else metric}_vs_resolution')
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
