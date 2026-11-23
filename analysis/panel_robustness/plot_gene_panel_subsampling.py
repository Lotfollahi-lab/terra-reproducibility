"""
Figures of the gene panel subsampling experiment, from results/metrics_long.tsv and panel_statistics.tsv
(gene_panel_subsampling.py summarize calls plot_results; run this file to redraw them without recomputing):

    python plot_gene_panel_subsampling.py --output_folder_path <run folder>

- biology_recovery: NMI/ARI of Leiden clusters against the annotations and leave-one-section-out label transfer, per
  panel size, for TERRA and the PCA baselines;
- consistency_with_full_panel: kNN overlap and cluster agreement with the full panel (TERRA and PCA), and label
  transfer from full-panel to subset-panel embeddings (TERRA, which embeds every panel in one space); the point at the
  full panel is the full panel embedded a second time (for clusters: the seed-to-seed ceiling);
- disjoint_panels: two panels of the same size without a shared gene (TERRA);
- panel_statistics: detected genes per cell per panel size.

Points are means over the random panels of a size (error bars: standard deviation); the full panel is one point.
Colors: the reference categorical palette of the dataviz guidance (slot 1 blue = TERRA, slot 2 orange = PCA,
slot 3 aqua = TERRA transfer across panels), a one-hue ramp for panel size, and neutral gray for data statistics.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TERRA = '#2a78d6'
PCA = '#eb6834'
AQUA = '#1baf7a'
BLUE_LIGHT = '#86b6ef'
BLUE_DARK = '#1c5cab'
NEUTRAL = '#52514e'
NEUTRAL_LIGHT = '#b5b4ae'
TEXT = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e6e5e1'
AXIS = '#b5b4ae'

LABELS = {'niche': 'Niches', 'cell_type': 'Cell types'}
PAIRS = {'niche': ('terra_neighborhood', 'pca_neighborhood'), 'cell_type': ('terra_cell', 'pca_cell')}


def _style():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 6, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': TEXT_SECONDARY,
        'axes.titlecolor': TEXT, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'text.color': TEXT,
        'xtick.color': TEXT_SECONDARY, 'ytick.color': TEXT_SECONDARY,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'axes.grid.axis': 'y',
        'grid.color': GRID, 'grid.linewidth': 0.5, 'grid.linestyle': '-', 'legend.frameon': False,
        'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
        'figure.constrained_layout.use': True,
    })
    return plt


def _aggregate(metrics, comparison, representation, metric, kinds, sizes=None):
    """Mean, standard deviation and count over replicates per gene number (overall values, section 'all')."""
    data = metrics[(metrics['comparison'] == comparison) & (metrics['representation'] == representation)
                   & (metrics['metric'] == metric) & (metrics['section'] == 'all') & metrics['kind'].isin(kinds)]
    if sizes is not None:
        data = data[data['n_genes'].isin(sizes)]
    return (data.groupby('n_genes')['value'].agg(['mean', 'std', 'count']).reset_index()
            .sort_values('n_genes'))


def _line(ax, table, color, label, marker='o', linestyle='-', filled=True):
    if table.empty:
        return
    ax.errorbar(table['n_genes'], table['mean'], yerr=table['std'].fillna(0.0), color=color, label=label,
                marker=marker, markersize=4, linewidth=1.4, linestyle=linestyle, capsize=1.8, elinewidth=0.7,
                markerfacecolor=color if filled else 'white', markeredgecolor='white' if filled else color,
                markeredgewidth=0.8 if filled else 1.0, zorder=3)


def _size_label(n):
    return f'{n / 1000:g}k' if n >= 1000 and n % 100 == 0 else f'{n:,}'


def _size_axis(ax, sizes, n_panel_genes):
    ticks = sorted(sizes) + [n_panel_genes]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_size_label(t) for t in sorted(sizes)] + [f'{n_panel_genes:,}\n(full)'])
    pad = 0.06 * (ticks[-1] - ticks[0])
    ax.set_xlim(ticks[0] - pad, ticks[-1] + pad)
    ax.set_xlabel('Genes in panel')


def _legend(fig, handles, labels, ncol):
    """Legend below the panels (space reserved by constrained layout)."""
    try:
        fig.legend(handles, labels, loc='outside lower center', ncol=ncol)
    except ValueError:  # matplotlib < 3.7
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=ncol)


def _save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(folder / f'{name}.png', dpi=300, bbox_inches='tight')


def plot_biology_recovery(plt, metrics, sizes, n_panel_genes, folder):
    panels = [('ari_kmatched', 'ARI with annotation'), ('nmi_kmatched', 'NMI with annotation'),
              ('transfer_macro_f1', 'Label transfer to held-out\nsection (kNN, macro F1)')]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for row, label in enumerate(('niche', 'cell_type')):
        terra, pca = PAIRS[label]
        for column, (metric, ylabel) in enumerate(panels):
            ax = axes[row, column]
            _line(ax, _aggregate(metrics, 'intrinsic', terra, metric, ['random', 'full']), TERRA, 'TERRA')
            _line(ax, _aggregate(metrics, 'intrinsic', pca, metric, ['random', 'full']), PCA, 'PCA', marker='s')
            _size_axis(ax, sizes, n_panel_genes)
            ax.set_ylabel(ylabel)
            ax.set_title(LABELS[label])
    handles, _ = axes[0, 0].get_legend_handles_labels()
    fig.suptitle('Biology recovered with random subsets of the panel', fontsize=8, fontweight='bold')
    _legend(fig, handles, ['TERRA (niches: neighborhood embedding; cell types: cell embedding)',
                           'PCA (niches: of neighborhood-averaged expression; cell types: of expression)'], ncol=2)
    _save(fig, folder, 'biology_recovery')
    plt.close(fig)


def plot_consistency(plt, metrics, sizes, n_panel_genes, folder):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    for row, label in enumerate(('niche', 'cell_type')):
        terra, pca = PAIRS[label]
        for representation, color, name, marker in ((terra, TERRA, 'TERRA', 'o'), (pca, PCA, 'PCA', 's')):
            _line(axes[row, 0], _aggregate(metrics, 'vs_full', representation, 'knn_overlap',
                                           ['random', 'full_repeat']), color, name, marker=marker)
            _line(axes[row, 1], _aggregate(metrics, 'vs_full', representation, 'cluster_ari_kmatched',
                                           ['random', 'full_repeat']), color, name, marker=marker)
        axes[row, 0].set_ylabel('Shared nearest neighbors\nwith full panel (k = 15)')
        axes[row, 0].set_title(f'{LABELS[label]}: neighbor preservation')
        axes[row, 1].set_ylabel('ARI of Leiden clusters with\nfull-panel clusters (other seeds)')
        axes[row, 1].set_title(f'{LABELS[label]}: cluster agreement')

        ax = axes[row, 2]
        _line(ax, _aggregate(metrics, 'intrinsic', terra, 'transfer_macro_f1', ['random', 'full']), TERRA, 'TERRA')
        _line(ax, _aggregate(metrics, 'vs_full', terra, 'cross_panel_transfer_macro_f1', ['random', 'full_repeat']),
              AQUA, 'TERRA, reference on full panel, query on subset', marker='s')
        _line(ax, _aggregate(metrics, 'vs_full', terra, 'cross_panel_transfer_macro_f1_centered',
                             ['random', 'full_repeat']),
              AQUA, 'Same, each panel centered on its mean', marker='s', linestyle='--', filled=False)
        ax.set_ylabel('Label transfer to held-out\nsection (kNN, macro F1)')
        ax.set_title(f'{LABELS[label]}: transfer across panels')
        for ax in axes[row]:
            _size_axis(ax, sizes, n_panel_genes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    transfer_handles, transfer_labels = axes[0, 2].get_legend_handles_labels()
    fig.suptitle('Agreement of subset-panel embeddings with full-panel embeddings of the same cells', fontsize=8,
                 fontweight='bold')
    _legend(fig, handles + transfer_handles[1:], labels + transfer_labels[1:], ncol=4)
    _save(fig, folder, 'consistency_with_full_panel')
    plt.close(fig)


def plot_disjoint(plt, metrics, folder):
    disjoint_sizes = sorted(int(s) for s in metrics.loc[metrics['kind'] == 'disjoint', 'n_genes'].dropna().unique())
    if not disjoint_sizes:
        return
    comparisons = [('Random\nvs full', 'vs_full', 'random'),
                   ('Disjoint\nvs full', 'vs_full', 'disjoint'),
                   ('Disjoint\nvs random', 'vs_partner', 'disjoint')]
    panels = [('knn_overlap', 'Shared nearest neighbors (k = 15)'),
              ('cluster_ari_kmatched', 'ARI of Leiden clusters'),
              ('cross_panel_transfer_macro_f1', 'Label transfer from one panel\nto the other (macro F1)')]
    if len(disjoint_sizes) == 2:
        colors = dict(zip(disjoint_sizes, [BLUE_LIGHT, BLUE_DARK]))
    else:
        colors = dict(zip(disjoint_sizes, plt.cm.Blues(np.linspace(0.45, 0.9, len(disjoint_sizes)))))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    offsets = np.linspace(-0.12, 0.12, len(disjoint_sizes))
    for row, (label, representation) in enumerate((('niche', 'terra_neighborhood'), ('cell_type', 'terra_cell'))):
        for column, (metric, ylabel) in enumerate(panels):
            ax = axes[row, column]
            for offset, size in zip(offsets, disjoint_sizes):
                means, stds = [], []
                for _, comparison, kind in comparisons:
                    table = _aggregate(metrics, comparison, representation, metric, [kind], sizes=[size])
                    means.append(table['mean'].iloc[0] if len(table) else np.nan)
                    stds.append(table['std'].fillna(0.0).iloc[0] if len(table) else np.nan)
                ax.errorbar(np.arange(len(comparisons)) + offset, means, yerr=stds, color=colors[size],
                            marker='o', markersize=4.5, linestyle='none', capsize=1.8, elinewidth=0.7,
                            markeredgecolor='white', markeredgewidth=0.8, label=f'{size:,} genes', zorder=3)
            ax.set_xticks(np.arange(len(comparisons)))
            ax.set_xticklabels([c[0] for c in comparisons])
            ax.set_xlim(-0.5, len(comparisons) - 0.5)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{LABELS[label]} (TERRA)')
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle('Two random panels of the same size that share no gene ("disjoint" vs "random")', fontsize=8,
                 fontweight='bold')
    _legend(fig, handles, legend_labels, ncol=len(disjoint_sizes))
    _save(fig, folder, 'disjoint_panels')
    plt.close(fig)


def plot_panel_statistics(plt, statistics, sizes, n_panel_genes, folder):
    above = [c for c in statistics.columns if c.startswith('pct_cells_above_')]
    panels = [('median_detected_genes', 'Median detected genes per cell'),
              ('pct_cells_below_10_genes', 'Cells with < 10 detected genes (%)')]
    if above:
        panels.append((above[0], f'Cells with > {above[0].split("_")[3]} detected genes (%),\n'
                                 'truncated to their top-ranked genes'))
    data = statistics[statistics['kind'].isin(['random', 'full'])].copy()
    data['n_genes'] = data['n_panel_genes']
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.4))
    for ax, (column, ylabel) in zip(np.atleast_1d(axes), panels):
        sections = data[data['section'] != 'all'].groupby(['section', 'n_genes'])[column].mean().reset_index()
        for _, section_data in sections.groupby('section'):
            ax.plot(section_data['n_genes'], section_data[column], color=NEUTRAL_LIGHT, linewidth=0.8, zorder=2)
        overall = data[data['section'] == 'all'].groupby('n_genes')[column].mean().reset_index()
        ax.plot(overall['n_genes'], overall[column], color=NEUTRAL, linewidth=1.4, marker='o', markersize=4,
                markeredgecolor='white', markeredgewidth=0.8, zorder=3)
        _size_axis(ax, sizes, n_panel_genes)
        ax.set_ylabel(ylabel)
    fig.suptitle('Detected genes per cell (dark: all cells; light: each section)', fontsize=8, fontweight='bold')
    _save(fig, folder, 'panel_statistics')
    plt.close(fig)


def plot_results(metrics: pd.DataFrame, statistics: pd.DataFrame, folder, n_panel_genes: int):
    plt = _style()
    folder = Path(folder)
    sizes = sorted(int(s) for s in metrics.loc[metrics['kind'] == 'random', 'n_genes'].dropna().unique())
    plot_biology_recovery(plt, metrics, sizes, n_panel_genes, folder)
    plot_consistency(plt, metrics, sizes, n_panel_genes, folder)
    plot_disjoint(plt, metrics, folder)
    plot_panel_statistics(plt, statistics, sizes, n_panel_genes, folder)


def main():
    parser = argparse.ArgumentParser(description='Redraw the figures of a gene panel subsampling run.')
    parser.add_argument('--output_folder_path', required=True)
    args = parser.parse_args()
    import json

    out = Path(args.output_folder_path)
    metrics = pd.read_csv(out / 'results' / 'metrics_long.tsv', sep='\t')
    statistics = pd.read_csv(out / 'panel_statistics.tsv', sep='\t')
    with open(out / 'panels.json') as f:
        n_panel_genes = json.load(f)['n_panel_genes']
    plot_results(metrics, statistics, out / 'results' / 'figures', n_panel_genes)


if __name__ == '__main__':
    main()
