"""Figures for niche identification at a common grid of cluster numbers (score_k_grid.py).

Reads ``<method>/<dataset>_<method>_k_grid_scores.csv`` and writes to ``<FINAL_OUTPUT_FOLDER_PATH>/summary/``:
- ``k_grid.csv``: one row per method, dataset and k (selected resolution, cluster number, NMI, ARI, matched);
- ``figures/k_grid/niche_{NMI,ARI}_k_grid.{svg,pdf,png}``: paper-style bars as niche_{NMI,ARI}_comparison_human
  (bar = mean over the k-grid, error bar = 1 SD, dots = k values; hollow dots = k not reached exactly, closest
  cluster number used);
- ``figures/k_grid/{NMI,ARI}_vs_k.{svg,png}``: one line per method and dataset; dashed line = ground-truth niche
  number.

Usage (terra environment):
    python plot_k_grid.py
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
OUT = RESULTS / 'summary' / 'figures' / 'k_grid'
DATASETS = ['xhs1000-39b_1p', 'xhs49-1b_1p', 'chl59-8b_1p']   # as score_k_grid.DATASETS


def collect():
    frames = [pd.read_csv(path) for m in METHODS for d in DATASETS
              for path in [RESULTS / m / f'{d}_{m}_k_grid_scores.csv'] if path.exists()]
    return pd.concat(frames, ignore_index=True)


def mark_unmatched(fig, df, metric, labels, methods):
    """Hollow white dots over the k values that were not reached exactly."""
    for ax, dataset in zip(fig.axes, labels):
        sub = df[(df['dataset'] == dataset) & ~df['matched']]
        for _, row in sub.iterrows():
            ax.scatter(methods.index(row['method']), row[metric], s=12, facecolor='white', edgecolor='0.15',
                       linewidth=0.6, zorder=5)


def curves(df, metric, methods, labels, path):
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 3.2), sharey=True)
    for ax, (dataset, label) in zip(axes, labels.items()):
        sub = df[df['dataset'] == dataset]
        for m in methods:
            run = sub[sub['method'] == m].sort_values('k')
            if run.empty:
                continue
            is_terra = m == 'terra'
            ax.plot(run['k'], run[metric], marker='o', markersize=3, linewidth=2.2 if is_terra else 1,
                    color=METHODS[m]['color'], label=METHODS[m]['label'], zorder=3 if is_terra else 2)
        if len(sub):
            ax.axvline(sub['n_target'].iloc[0], color='0.4', linestyle='--', linewidth=0.8)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Number of clusters (k)')
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(metric)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ['svg', 'png']:
        fig.savefig(path.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(RESULTS / 'summary' / 'k_grid.csv', index=False)
    labels = {d: DATASET_LABELS[d] for d in DATASETS if d in set(df['dataset'])}
    methods = [m for m in METHODS if m in set(df['method'])]
    summary = df.groupby(['method', 'dataset'])[['NMI', 'ARI']].mean().unstack('dataset').reindex(methods)
    summary[('NMI', 'mean')] = df.groupby('method')['NMI'].mean().reindex(methods)
    summary[('ARI', 'mean')] = df.groupby('method')['ARI'].mean().reindex(methods)
    print(summary.round(3).to_string())
    unmatched = df[~df['matched']]
    if len(unmatched):
        print('k not reached exactly (closest cluster number used):')
        print(unmatched[['method', 'dataset', 'k', 'n_niches']].to_string(index=False))

    with plt.rc_context(P.PAPER_RC):
        for metric in ['NMI', 'ARI']:
            fig = P.plot_paper_metric_comparison(df, metric, labels, method_order=methods,
                                                 figsize_per_panel=(0.24 * len(methods) + 0.6, 2.6))
            mark_unmatched(fig, df, metric, list(labels), methods)
            for ext in ['svg', 'pdf', 'png']:
                fig.savefig(OUT / f'niche_{metric}_k_grid.{ext}')
            plt.close(fig)
    for metric in ['NMI', 'ARI']:
        curves(df, metric, methods, labels, OUT / f'{metric}_vs_k')
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
