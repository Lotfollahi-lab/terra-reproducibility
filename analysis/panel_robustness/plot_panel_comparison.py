"""
Figures of compare_panel_runs.py (it calls plot_comparison; run this file to redraw them):
    python plot_panel_comparison.py --output_folder_path <comparison folder>
- similarity_to_full_panel: similarity of a panel's embedding to a full-panel embedding against the number of genes,
  for every arm: a model applied to random panels compared with its own full panel (zero-shot, one embedding space);
  models trained on a panel compared with a full-panel model; a full-panel training run applied to random panels
  compared with another full-panel training run (zero-shot, two models). The dashed line is the training-seed ceiling
  (two full-panel training runs on the full panel); dotted lines are each run's Leiden seed-to-seed agreement.
- biology_by_arm: annotation recovery (K-matched Leiden ARI, leave-one-section-out kNN transfer) against the number
  of genes for each run (zero-shot on random panels) and for the panel-trained models (each on its own panel, with
  the full-panel training runs as the full-panel point), with the PCA baseline.
Points are means over random panel draws (error bars: standard deviation).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_gene_panel_subsampling import (NEUTRAL, NEUTRAL_LIGHT, PCA, _legend, _line, _save,  # noqa: E402
                                         _size_label, _style)

PALETTE = ['#2a78d6', '#1baf7a', '#8a63d2', '#d23f6b', '#c9a227', '#3fb0c9', '#6b7a8f', '#1c5cab']
SIMILARITY_METRICS = [('knn_overlap', 'kNN overlap'), ('linear_cka', 'Linear CKA'),
                      ('cluster_ari_kmatched', 'Cluster agreement (ARI)'),
                      ('procrustes_cosine', 'Matched cosine, aligned'),
                      ('procrustes_foscttm', 'FOSCTTM, aligned (lower = better)')]
REPRESENTATION_TITLES = {'terra_neighborhood': 'Neighborhoods', 'terra_cell': 'Cells',
                         'terra_spatial_cell': 'Spatial cells'}
BIOLOGY_PANELS = [('terra_neighborhood', 'pca_neighborhood', 'ari_kmatched', 'Niches: ARI'),
                  ('terra_cell', 'pca_cell', 'ari_kmatched', 'Cell types: ARI'),
                  ('terra_neighborhood', 'pca_neighborhood', 'transfer_macro_f1', 'Niches: transfer macro F1'),
                  ('terra_cell', 'pca_cell', 'transfer_macro_f1', 'Cell types: transfer macro F1')]


def _aggregate(data):
    return (data.groupby('n_genes')['value'].agg(['mean', 'std', 'count']).reset_index().sort_values('n_genes'))


def _gene_axis(ax, n_genes):
    ticks = sorted(set(int(n) for n in n_genes))
    ax.set_xticks(ticks)
    ax.set_xticklabels([_size_label(t) for t in ticks])
    ax.set_xlabel('Genes in panel')


def similarity_lines(overall: pd.DataFrame, representation: str, metric: str) -> list:
    """(label, table) per line of the similarity figure."""
    data = overall[(overall['representation'] == representation) & (overall['metric'] == metric)]
    data = data.rename(columns={'n_genes_a': 'n_genes'})
    lines = []
    zero_shot = data[(data['pair_type'] == 'same_model_panel_vs_full') & (data['panel_kind_a'] == 'random')]
    for run in zero_shot['run_a'].unique():
        lines.append((f'{run}: panel vs own full panel', _aggregate(zero_shot[zero_shot['run_a'] == run])))
    # Models trained here with the same seed share their initialization: compare with the other seed, as the
    # training-seed ceiling does
    trained = data[(data['pair_type'] == 'trained_panel_vs_full') & ~data['same_seed'].fillna(False).astype(bool)]
    for run in trained['run_b'].unique():
        lines.append((f'trained on panel vs {run} full panel', _aggregate(trained[trained['run_b'] == run])))
    cross = data[(data['pair_type'] == 'panel_vs_other_model_full') & (data['panel_kind_a'] == 'random')]
    for (run_a, run_b), group in cross.groupby(['run_a', 'run_b']):
        lines.append((f'{run_a} panel vs {run_b} full panel', _aggregate(group)))
    return lines


def plot_similarity(overall: pd.DataFrame, folder: Path):
    plt = _style()
    representations = [r for r in REPRESENTATION_TITLES if r in set(overall['representation'])]
    metrics = [(m, t) for m, t in SIMILARITY_METRICS if m in set(overall['metric'])]
    if not representations or not metrics:
        return
    fig, axes = plt.subplots(len(metrics), len(representations), figsize=(3.4 * len(representations),
                                                                          2.1 * len(metrics)), squeeze=False)
    colors, handles = {}, {}
    ceiling = overall[(overall['pair_type'] == 'same_panel_across_models') & (overall['condition_a'] == 'full')
                      & (overall['kind_a'] == 'trained_full') & (overall['kind_b'] == 'trained_full')]
    seed_ceiling = overall[overall['pair_type'] == 'leiden_seed_ceiling']
    for i, (metric, title) in enumerate(metrics):
        for j, representation in enumerate(representations):
            ax = axes[i, j]
            genes = set()
            for label, table in similarity_lines(overall, representation, metric):
                color = colors.setdefault(label, PALETTE[len(colors) % len(PALETTE)])
                _line(ax, table, color, label)
                genes |= set(table['n_genes'])
                handles[label] = ax.get_legend_handles_labels()[0][-1] if ax.get_legend_handles_labels()[0] else None
            values = ceiling[(ceiling['representation'] == representation) & (ceiling['metric'] == metric)]['value']
            if len(values):
                line = ax.axhline(values.mean(), color=NEUTRAL, linestyle='--', linewidth=0.9, zorder=2)
                handles['training-seed ceiling (two full-panel models)'] = line
            values = seed_ceiling[(seed_ceiling['representation'] == representation)
                                  & (seed_ceiling['metric'] == metric)]['value']
            if len(values):
                line = ax.axhline(values.mean(), color=NEUTRAL_LIGHT, linestyle=':', linewidth=0.9, zorder=2)
                handles['Leiden seed-to-seed agreement (one embedding)'] = line
            if genes:
                _gene_axis(ax, genes)
            ax.set_title(f'{REPRESENTATION_TITLES[representation]}: {title}')
    items = [(label, handle) for label, handle in handles.items() if handle is not None]
    if items:
        _legend(fig, [h for _, h in items], [label for label, _ in items], ncol=2)
    _save(fig, folder, 'similarity_to_full_panel')
    plt.close(fig)


def plot_biology(biology: pd.DataFrame, folder: Path):
    plt = _style()
    data = biology[biology['section'] == 'all']
    fig, axes = plt.subplots(1, len(BIOLOGY_PANELS), figsize=(3.0 * len(BIOLOGY_PANELS), 2.4), squeeze=False)
    colors, handles = {}, {}
    for ax, (terra_rep, pca_rep, metric, title) in zip(axes[0], BIOLOGY_PANELS):
        genes = set()
        rows = data[(data['metric'] == metric)]
        zero_shot = rows[(rows['representation'] == terra_rep) & rows['panel_kind'].isin(['random', 'full'])
                         & (rows['kind'] != 'trained_panel')]
        for run in zero_shot['run'].unique():
            label = f'{run} (zero-shot on random panels)'
            color = colors.setdefault(label, PALETTE[len(colors) % len(PALETTE)])
            table = _aggregate(zero_shot[zero_shot['run'] == run])
            _line(ax, table, color, label)
            handles[label] = ax.get_legend_handles_labels()[0][-1]
            genes |= set(table['n_genes'])
        trained = rows[(rows['representation'] == terra_rep)
                       & (((rows['kind'] == 'trained_panel') & (rows['panel_kind'] != 'full_repeat'))
                          | ((rows['kind'] == 'trained_full') & (rows['panel_kind'] == 'full')))]
        trained = trained[(trained['kind'] == 'trained_full')
                          | (trained['panel_kind'].isin(['random', 'disjoint'])
                             & (trained['condition'] == trained['trained_panel']))]
        if not trained.empty:
            label = 'trained on the panel (own panel)'
            color = colors.setdefault(label, '#d23f6b')
            table = _aggregate(trained)
            _line(ax, table, color, label, marker='s')
            handles[label] = ax.get_legend_handles_labels()[0][-1]
            genes |= set(table['n_genes'])
        first_run = data['run'].iloc[0]
        pca = rows[(rows['representation'] == pca_rep) & (rows['run'] == first_run)
                   & rows['panel_kind'].isin(['random', 'full'])]
        if not pca.empty:
            label = 'PCA baseline'
            _line(ax, _aggregate(pca), PCA, label, marker='D', filled=False)
            handles[label] = ax.get_legend_handles_labels()[0][-1]
        if genes:
            _gene_axis(ax, genes)
        ax.set_title(title)
    _legend(fig, list(handles.values()), list(handles), ncol=2)
    _save(fig, folder, 'biology_by_arm')
    plt.close(fig)


def plot_comparison(folder):
    folder = Path(folder)
    pairs = pd.read_csv(folder / 'pairs.tsv', sep='\t')
    long = pd.read_csv(folder / 'similarity_long.tsv', sep='\t')
    overall = long[long['section'] == 'all'].merge(pairs, on='pair_id')
    figures = folder / 'figures'
    plot_similarity(overall, figures)
    biology_file = folder / 'biology_long.tsv'
    if biology_file.is_file():
        plot_biology(pd.read_csv(biology_file, sep='\t', dtype={'trained_panel': object}), figures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output_folder_path', required=True)
    plot_comparison(parser.parse_args().output_folder_path)
    print(f'Figures in {Path(parser.parse_args().output_folder_path) / "figures"}')
