"""Figures for the human niche benchmark (used by benchmark_comparison.ipynb).

- Method colours, labels and order come from ``bench_utils.METHODS`` (as in the
  paper's mouse figure, mouse_benchmarking/benchmarking_comparison.ipynb).
- ``plot_paper_metric_comparison`` is the drawing code of that notebook's
  ``plot_metric_comparison``, with one panel per dataset.
- Spatial maps use the repo's ``custom_palette`` (utils/analysis_utils.py);
  predicted clusters take the colour of the ground-truth niche they overlap
  most (Hungarian matching), unmatched clusters are grey.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / 'utils'))
from analysis_utils import custom_palette  # noqa: E402
from bench_utils import METHODS  # noqa: E402

METHOD_ORDER = list(METHODS)
INK, INK_2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
UNMATCHED = '#c3c2b7'
SEQ_BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
            '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281',
            '#0d366b']

# rcParams of the paper figure (mouse notebook, cell "Run Notebook Setup")
PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def plot_paper_metric_comparison(rows, metric, dataset_labels,
                                 method_order=METHOD_ORDER,
                                 figsize_per_panel=(3.0, 2.4),
                                 annotations=None):
    """Paper-style bar figure, one panel per dataset.

    ``rows``: one row per Leiden run (e.g. ``sweep.csv`` or ``matched.csv``).
    Bar = mean of the metric over the runs of a method, error bar = 1 SD,
    dots = individual runs, dashed line = TERRA mean.
    ``annotations``: {(method, dataset): text} printed above a bar.
    """
    n_panels = len(dataset_labels)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
        sharey=True, constrained_layout=True)
    if n_panels == 1:
        axes = [axes]
    n_methods = len(method_order)
    colors = [METHODS[m]['color'] for m in method_order]

    for ax, (dataset, label) in zip(axes, dataset_labels.items()):
        sub = rows[rows['dataset'] == dataset]
        means, sds, all_vals = [], [], []
        for m in method_order:
            vals = sub.loc[sub['method'] == m, metric].values
            means.append(np.mean(vals) if len(vals) > 0 else 0)
            sds.append(np.std(vals) if len(vals) > 0 else 0)
            all_vals.append(vals)
        x = np.arange(n_methods)

        ax.bar(x, means, width=0.6, color=colors, edgecolor='white',
               linewidth=0.6, alpha=0.75, zorder=2)
        ax.errorbar(x, means, yerr=sds, fmt='none', ecolor='0.25',
                    elinewidth=0.8, capsize=2.5, capthick=0.8, zorder=3)
        rng = np.random.default_rng(42)
        for i, vals in enumerate(all_vals):
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=12, color='0.15',
                       edgecolor='white', linewidth=0.3, zorder=4, alpha=0.65)
        if 'terra' in method_order:
            ax.axhline(y=means[method_order.index('terra')], color='0.3',
                       linestyle='--', linewidth=0.8, zorder=1, alpha=0.7)
        for (m, d), text in (annotations or {}).items():
            if d == dataset and m in method_order:
                i = method_order.index(m)
                ax.text(i, means[i] + sds[i] + 0.02, text, ha='center',
                        va='bottom', fontsize=7)

        ax.set_title(label, fontweight='medium')
        ax.set_xticks(x)
        ax.set_xticklabels([METHODS[m]['label'] for m in method_order],
                           rotation=45, ha='right')
        ax.tick_params(axis='x', pad=1)
        ax.set_ylabel(f'{metric} (↑)' if ax is axes[0] else '')
        ax.set_xlim(-0.5, n_methods - 0.5)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.grid(axis='y', linewidth=0.3, alpha=0.5)
    return fig


def _style_axes(ax, grid_axis='x'):
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#c3c2b7')
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_heatmap(headline, metric, dataset_labels, ax,
                 method_order=METHOD_ORDER):
    """Method x dataset heatmap of the headline metric (sequential blue)."""
    pivot = (headline.pivot(index='method', columns='dataset', values=metric)
             .reindex(index=method_order, columns=list(dataset_labels)))
    cmap = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUE)
    im = ax.imshow(pivot.values.astype(float), cmap=cmap, vmin=0, vmax=0.7,
                   aspect='auto')
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.isna(v):
                ax.text(j, i, '–', ha='center', va='center', color=MUTED, fontsize=8)
                continue
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if v > 0.45 else INK)
    ax.set_yticks(range(len(method_order)))
    ax.set_yticklabels([METHODS[m]['label'] for m in method_order], fontsize=8)
    ax.set_xticks(range(len(dataset_labels)))
    ax.set_xticklabels([d.replace(' (', '\n(') for d in dataset_labels.values()],
                       fontsize=8)
    ax.set_title(metric, fontsize=10, color=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    return im


def plot_sweep(sweep, metric, dataset_labels, axes):
    """Metric vs number of clusters over each method's sweep, per dataset."""
    for ax, (dataset, label) in zip(axes, dataset_labels.items()):
        df = sweep[sweep['dataset'] == dataset]
        for method, mdf in df.groupby('method'):
            mdf = mdf.sort_values('leiden_resolution')
            is_terra = method == 'terra'
            ax.plot(mdf['n_niches'], mdf[metric], marker='o', markersize=4,
                    linewidth=2.2 if is_terra else 1,
                    alpha=1 if is_terra else 0.7,
                    color=METHODS[method]['color'], zorder=3 if is_terra else 2)
            last = mdf.iloc[-1]
            ax.annotate(METHODS[method]['label'], (last['n_niches'], last[metric]),
                        xytext=(3, 0), textcoords='offset points', fontsize=6,
                        color=INK_2, va='center')
        if len(df):
            ax.axvline(df['n_target'].iloc[0], color=MUTED, linestyle='--',
                       linewidth=0.8)
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xlabel('Number of clusters', fontsize=8, color=INK_2)
        _style_axes(ax, grid_axis='y')
    axes[0].set_ylabel(metric, fontsize=9, color=INK)


def match_clusters_to_niches(niche_gt, predicted):
    """Map each predicted cluster to the GT niche it overlaps most (Hungarian)."""
    table = pd.crosstab(predicted, niche_gt)
    rows, cols = linear_sum_assignment(-table.values)
    return {table.index[r]: table.columns[c] for r, c in zip(rows, cols)}


def niche_colors(categories):
    palette = list(custom_palette.values())
    return {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}


def plot_spatial_panels(adatas, section, axes, gt_colors, max_cells=None,
                        seed=0):
    """Ground truth + one panel per method for one section.

    ``adatas``: {panel title: AnnData with obs 'niche_gt', 'section',
    'predicted_niche' and obsm 'spatial'}; all on the same cells. Large
    sections are randomly subsampled to ``max_cells`` for plotting only.
    """
    ref = next(iter(adatas.values()))
    idx = np.flatnonzero((ref.obs['section'] == section).values)
    if max_cells and len(idx) > max_cells:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_cells,
                                                         replace=False))
    coords = ref.obsm['spatial'][idx]
    point_size = float(np.clip(40000 / max(len(idx), 1), 0.3, 6.0))

    def scatter(ax, colors, title):
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=point_size,
                   linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

    gt = ref.obs['niche_gt'].astype(str).values[idx]
    scatter(axes[0], [gt_colors.get(g, UNMATCHED) for g in gt],
            f'{section}: ground truth')
    for ax, (name, adata) in zip(axes[1:], adatas.items()):
        pred_all = adata.obs['predicted_niche'].astype(str)
        mapping = match_clusters_to_niches(adata.obs['niche_gt'].astype(str),
                                           pred_all)
        pred = pred_all.reindex(ref.obs_names[idx]).values
        scatter(ax, [gt_colors.get(mapping.get(p), UNMATCHED) for p in pred], name)
