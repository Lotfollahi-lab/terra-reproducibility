"""Per-method report of the human niche benchmark, for the method notebooks.

Each ``human_benchmarking/<method>_benchmarking.ipynb`` has a section that calls
these functions, so every method's results are shown in the same way and in
the place the repo keeps per-method analyses. Nothing is recomputed: results
are read from ``FINAL_OUTPUT_FOLDER_PATH`` (written by the scripts in this
folder, see README.md).

    import sys
    sys.path.append('../human_niche_benchmark')
    from method_report import show_method_metrics, show_method_maps
    show_method_metrics('banksy')
    show_method_maps('banksy')
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / 'utils'))
import plotting as P  # noqa: E402
from bench_utils import DATASET_LABELS, FINAL_OUTPUT_FOLDER_PATH, METHODS  # noqa: E402
from benchmarking_utils import max_overlap_plot  # noqa: E402

RESULTS = Path(FINAL_OUTPUT_FOLDER_PATH)


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except ImportError:
        print(obj)


def _files(method, dataset):
    folder = RESULTS / method
    prefix = f'{dataset}_{method}'
    return dict(scan=folder / f'{prefix}_metric_scores.csv',
                batch=folder / f'{prefix}_batch_integration_metrics.csv',
                run_info=folder / f'{prefix}_run_info.json',
                h5ad=folder / f'{prefix}_embedded_clustered.h5ad')


def show_method_metrics(method):
    """Headline and matched NMI/ARI on all datasets, sweep, batch metrics and
    run parameters of one method."""
    print(f'{METHODS[method]["label"]}: niche identification at the '
          'ground-truth number of niches.')
    rows, sweeps, batches, params = [], [], [], {}
    for dataset, label in DATASET_LABELS.items():
        files = _files(method, dataset)
        if not files['scan'].exists():
            rows.append(dict(dataset=label, note='not available'))
            continue
        scan = pd.read_csv(files['scan'])
        run_info = (json.load(open(files['run_info']))
                    if files['run_info'].exists() else {})
        head = scan[scan['headline']].iloc[0]
        matched = scan[scan['matched']]
        rows.append(dict(
            dataset=label, n_target=head['n_target'], n_clusters=head['n_niches'],
            headline_resolution=head['leiden_resolution'],
            NMI=head['NMI'], ARI=head['ARI'], n_matched=len(matched),
            NMI_matched_mean=matched['NMI'].mean() if len(matched) else None,
            ARI_matched_mean=matched['ARI'].mean() if len(matched) else None,
            n_cells=run_info.get('n_cells'),
            coverage=run_info.get('coverage', 1.0),
            embedding_minutes=(run_info.get('embedding_seconds') or 0) / 60 or None))
        sweeps.append(scan[scan['in_sweep'] | scan['headline']].assign(dataset=label))
        if files['batch'].exists():
            batches.append(pd.read_csv(files['batch']).assign(dataset=label))
        params[label] = run_info.get('params', {})

    print('Headline = lowest 0.01-grid resolution giving the ground-truth number '
          'of niches; matched = all runs giving it:')
    _display(pd.DataFrame(rows).round(3))
    if sweeps:
        print('Sweep: +/- 4 grid steps (0.01) around the median matching resolution:')
        _display(pd.concat(sweeps)[['dataset', 'leiden_resolution', 'n_niches',
                                     'n_target', 'matched', 'headline',
                                     'NMI', 'ARI']].round(3))
    if batches:
        print('Batch integration (multi-section datasets): iLISI higher = more '
              'mixed; ASW near 0 = more mixed; MMD lower = more mixed.')
        _display(pd.concat(batches).pivot_table(index='dataset', columns='metric',
                                                values='score').round(3))
    if params:
        print('Run parameters:')
        _display(pd.DataFrame(params).T)


def show_method_maps(method, datasets=None, max_cells=60000):
    """Cluster-to-niche matching and spatial maps (ground truth vs. method,
    headline clustering) per section."""
    for dataset in datasets or DATASET_LABELS:
        path = _files(method, dataset)['h5ad']
        if not path.exists():
            print(f'{DATASET_LABELS[dataset]}: no results.')
            continue
        adata = sc.read_h5ad(path)
        print(f'===== {DATASET_LABELS[dataset]} ({dataset})')
        max_overlap_plot(adata.obs, 'niche_gt', 'predicted_niche')

        gt_cats = adata.obs['niche_gt'].astype(str).value_counts().index.tolist()
        gt_colors = P.niche_colors(gt_cats)
        sections = adata.obs['section'].cat.categories.tolist()
        fig, axes = plt.subplots(len(sections), 2, figsize=(7, 3.2 * len(sections)),
                                 squeeze=False)
        for row, section in zip(axes, sections):
            P.plot_spatial_panels({METHODS[method]['label']: adata}, section, row,
                                  gt_colors, max_cells=max_cells)
        handles = [plt.Line2D([], [], marker='o', linestyle='', color=c, label=n)
                   for n, c in gt_colors.items()]
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5),
                   fontsize=7, frameon=False, title='Ground-truth niches',
                   title_fontsize=8)
        fig.tight_layout()
        plt.show()
        del adata
