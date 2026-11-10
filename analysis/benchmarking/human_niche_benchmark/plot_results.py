"""Figures of the human niche benchmark.

Collects the results first (collect_results.py -> summary/*.csv), then:

- ``python plot_results.py``: main figure, one panel per dataset of
  ``bench_utils.MAIN_DATASETS``, written to ``summary/figures/``:
  ``niche_{NMI,ARI}_comparison_human.{svg,pdf,png}`` (bar = mean over the
  resolution sweep, error bar = 1 SD, dots = sweep runs, dashed line = TERRA
  mean; paper convention, mouse_benchmarking/benchmarking_comparison.ipynb) and
  ``niche_{NMI,ARI}_comparison_human_matched.*`` (only the runs giving exactly
  the ground-truth number of niches).
- ``python plot_results.py --dataset vhb3003-3b_1p``: the same bars for one
  dataset plus ``scores_<dataset>.csv``, in ``summary/figures/<dataset>/``.
- ``--maps`` (with ``--dataset``): spatial maps of the ground truth and every
  method's headline clustering; ``spatial_maps_<dataset>_selected.png`` (the
  sections with the most ground-truth niches) and ``..._part<k>.png`` (all
  sections). Predicted clusters take the colour of the ground-truth niche they
  overlap most (Hungarian matching); unmatched clusters are grey. Reads every
  method's clustered result: up to ~200 GB for Visium HD, so run it via LSF.

Methods: every method in ``bench_utils.METHODS`` with results, in that order
(colours and labels from there); ``NOT_APPLICABLE`` pairs show "n/a"; methods
that never reach the target number of niches are marked with a double dagger.
Runs in the terra or banksy environment.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import anndata as ad  # noqa: E402
import h5py  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from anndata.io import read_elem  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[3] / 'utils'))
import collect_results  # noqa: E402
from analysis_utils import custom_palette  # noqa: E402
from bench_utils import DATASET_LABELS, MAIN_DATASETS, METHODS  # noqa: E402

UNMATCHED = '#c3c2b7'

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


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------
def plot_paper_metric_comparison(rows, metric, dataset_labels, method_order,
                                 figsize_per_panel=(3.0, 2.4), annotations=None):
    """Paper-style bar figure, one panel per dataset (drawing code of the mouse
    notebook's ``plot_metric_comparison``).

    ``rows``: one row per clustering run (e.g. ``sweep.csv`` or ``matched.csv``).
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
                ax.text(i, 0.01, 'n/a', ha='center', va='bottom', fontsize=6,
                        color='0.4', rotation=90)
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


def bar_figures(summary, datasets, out, name, figsize_per_panel):
    """Sweep and matched-only bars for ``datasets``; returns the plotted methods."""
    headline = pd.read_csv(summary / 'headline.csv')
    done = headline[(headline['status'] == 'done') & headline['dataset'].isin(datasets)]
    methods = [m for m in METHODS if m in set(done['method'])]
    labels = {d: DATASET_LABELS[d] for d in datasets}
    unreachable = {(r.method, r.dataset): '‡' for r in done.itertuples() if r.n_clusters != r.n_target}
    print(done.pivot_table(index='method_name', columns='dataset_label', values=['NMI', 'ARI'])
          .reindex([METHODS[m]['label'] for m in methods]).round(3).to_string())
    if unreachable:
        print('Target not reached (‡):', sorted(unreachable))
    with plt.rc_context(PAPER_RC):
        for rows_file, suffix in [('sweep.csv', ''), ('matched.csv', '_matched')]:
            rows = pd.read_csv(summary / rows_file)
            for metric in ['NMI', 'ARI']:
                fig = plot_paper_metric_comparison(rows, metric, labels, methods,
                                                   figsize_per_panel(len(methods)), unreachable)
                for ext in ['svg', 'pdf', 'png']:
                    fig.savefig(out / f'niche_{metric}_comparison_{name}{suffix}.{ext}')
                plt.close(fig)
    return methods, done, unreachable


def dataset_table(summary, dataset, methods, done):
    """Sweep means and headline scores of one dataset (scores_<dataset>.csv)."""
    sweep = pd.read_csv(summary / 'sweep.csv')
    sweep = sweep[sweep['dataset'] == dataset]
    head = done.set_index('method')
    return pd.DataFrame({
        'method': [METHODS[m]['label'] for m in methods],
        'sweep_NMI': [sweep.loc[sweep['method'] == m, 'NMI'].mean() for m in methods],
        'sweep_ARI': [sweep.loc[sweep['method'] == m, 'ARI'].mean() for m in methods],
        'headline_NMI': head.loc[methods, 'NMI'].values,
        'headline_ARI': head.loc[methods, 'ARI'].values,
        'headline_n_clusters': head.loc[methods, 'n_clusters'].values,
        'target': head.loc[methods, 'n_target'].values})


# ---------------------------------------------------------------------------
# Spatial maps
# ---------------------------------------------------------------------------
def match_clusters_to_niches(niche_gt, predicted):
    """Map each predicted cluster to the GT niche it overlaps most (Hungarian)."""
    table = pd.crosstab(predicted, niche_gt)
    rows, cols = linear_sum_assignment(-table.values)
    return {table.index[r]: table.columns[c] for r, c in zip(rows, cols)}


def niche_colors(categories):
    """Ground-truth niche colours from the repo's custom_palette (utils/analysis_utils.py)."""
    palette = list(custom_palette.values())
    return {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}


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


def plot_spatial_panels(adatas, section, axes, gt_colors, max_cells=None, seed=0):
    """Ground truth + one panel per method for one section.

    ``adatas``: {panel title: AnnData with obs 'niche_gt', 'section',
    'predicted_niche' and obsm 'spatial'}; all on the same cells. Large
    sections are randomly subsampled to ``max_cells`` for plotting only.
    """
    ref = next(iter(adatas.values()))
    idx = np.flatnonzero((ref.obs['section'] == section).values)
    if max_cells and len(idx) > max_cells:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_cells, replace=False))
    coords = ref.obsm['spatial'][idx]
    point_size = float(np.clip(40000 / max(len(idx), 1), 0.3, 6.0))

    def scatter(ax, colors, title):
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=point_size,
                   linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=8)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

    gt = ref.obs['niche_gt'].astype(str).values[idx]
    scatter(axes[0], [gt_colors.get(g, UNMATCHED) for g in gt], f'{section}: ground truth')
    for ax, (name, adata) in zip(axes[1:], adatas.items()):
        pred_all = adata.obs['predicted_niche'].astype(str)
        mapping = match_clusters_to_niches(adata.obs['niche_gt'].astype(str), pred_all)
        pred = pred_all.reindex(ref.obs_names[idx]).values
        scatter(ax, [gt_colors.get(mapping.get(p), UNMATCHED) for p in pred], name)


def spatial_figure(adatas, sections, gt_colors, dataset_label, path, max_cells):
    n_cols = len(adatas) + 1
    fig, axes = plt.subplots(len(sections), n_cols, figsize=(1.9 * n_cols, 2.0 * len(sections)),
                             squeeze=False)
    for row, section in zip(axes, sections):
        plot_spatial_panels(adatas, section, row, gt_colors, max_cells=max_cells)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=c, label=n) for n, c in gt_colors.items()]
    handles.append(plt.Line2D([], [], marker='o', linestyle='', color=UNMATCHED, label='unmatched cluster'))
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False,
               title=f'{dataset_label}: ground-truth niches', title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {path}', flush=True)


def spatial_maps(dataset, methods, unreachable, out, args):
    label = DATASET_LABELS[dataset]
    adatas = {METHODS[m]['label'] + (' ‡' if (m, dataset) in unreachable else ''):
              read_obs_and_coords(collect_results.RESULTS / m / f'{dataset}_{m}_embedded_clustered.h5ad')
              for m in methods}
    ref = next(iter(adatas.values()))
    gt = ref.obs['niche_gt'].astype(str)
    gt_colors = niche_colors(gt.value_counts().index.tolist())
    sections = ref.obs['section'].astype(str)
    # Sections with the most ground-truth niches (> 100 cells in the section) first
    n_niches = (pd.crosstab(sections, gt) > 100).sum(1).sort_values(ascending=False)
    print('Ground-truth niches per section:\n', n_niches.to_string(), flush=True)
    selected = n_niches.index[:args.n_selected].tolist()
    spatial_figure(adatas, selected, gt_colors, label, out / f'spatial_maps_{dataset}_selected.png',
                   args.max_cells)
    ordered = sorted(sections.unique())
    for k, start in enumerate(range(0, len(ordered), args.sections_per_page)):
        spatial_figure(adatas, ordered[start:start + args.sections_per_page], gt_colors, label,
                       out / f'spatial_maps_{dataset}_part{k + 1}.png', args.max_cells)


def main(args):
    collect_results.main()
    summary = collect_results.OUT
    if args.dataset is None:
        out = summary / 'figures'
        out.mkdir(parents=True, exist_ok=True)
        bar_figures(summary, MAIN_DATASETS, out, 'human', lambda n: (0.24 * n + 0.6, 2.6))
    else:
        dataset = args.dataset
        out = summary / 'figures' / dataset
        out.mkdir(parents=True, exist_ok=True)
        methods, done, unreachable = bar_figures(summary, [dataset], out, dataset,
                                                 lambda n: (0.38 * n + 1.2, 2.6))
        dataset_table(summary, dataset, methods, done).to_csv(out / f'scores_{dataset}.csv', index=False)
        if args.maps:
            spatial_maps(dataset, methods, unreachable, out, args)
    print(f'Wrote {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark figures (main figure, or one dataset).')
    parser.add_argument('--dataset', choices=list(DATASET_LABELS), default=None,
                        help='One dataset instead of the main figure.')
    parser.add_argument('--maps', action='store_true', help='Spatial maps (with --dataset; high memory).')
    parser.add_argument('--n_selected', type=int, default=4, help='Sections in the selected spatial-map figure.')
    parser.add_argument('--sections_per_page', type=int, default=8)
    parser.add_argument('--max_cells', type=int, default=60000, help='Plotting-only subsample per section.')
    args = parser.parse_args()
    if args.maps and args.dataset is None:
        parser.error('--maps needs --dataset')
    main(args)
