"""Collect all method results into tidy tables for the comparison notebook.

Reads the per-method outputs in ``<FINAL_OUTPUT_FOLDER_PATH>/<method>/`` and
writes to ``<FINAL_OUTPUT_FOLDER_PATH>/summary/``:

- ``headline.csv``  one row per method x dataset: headline NMI/ARI (lowest grid
  resolution giving the common cluster number, or best reachable), number of
  clusters, mean NMI/ARI over all runs that hit the target exactly
  (robustness), runtime, coverage, note.
- ``scan.csv``      every Leiden run of the 0.01-grid scan (flags: headline,
  matched, in_sweep).
- ``sweep.csv``     sweep rows: +/- 4 grid steps around the median matching
  resolution (``in_sweep``).
- ``matched.csv``   all runs with exactly the target number of niches (the
  headline / best-reachable run if the target is never reached).
- ``batch.csv``     batch-integration metrics (multi-section datasets).

Runs in any environment with the bench_utils dependencies:
    python collect_results.py
"""
import json
from pathlib import Path

import pandas as pd

from bench_utils import DATASET_LABELS, FINAL_OUTPUT_FOLDER_PATH, METHODS

RESULTS = Path(FINAL_OUTPUT_FOLDER_PATH)
OUT = RESULTS / 'summary'


def read_run_info(method, dataset):
    path = RESULTS / method / f'{dataset}_{method}_run_info.json'
    return json.load(open(path)) if path.exists() else {}


def matched_rows(scan):
    """Runs with exactly the target number of niches, per method and dataset.

    If the target is never reached (e.g. Nicheformer on chr78), the headline
    (best reachable) run is kept instead so the method still has a bar.
    """
    parts = []
    for _, group in scan.groupby(['method', 'dataset']):
        hit = group[group['matched']]
        parts.append(hit if len(hit) else group[group['headline']])
    return pd.concat(parts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    headline_rows, scan_frames, batch_frames = [], [], []

    for method, spec in METHODS.items():
        for dataset, dataset_label in DATASET_LABELS.items():
            base = dict(method=method, method_name=spec['label'],
                        dataset=dataset, dataset_label=dataset_label,
                        note=spec['note'])
            path = RESULTS / method / f'{dataset}_{method}_metric_scores.csv'
            if not path.exists():
                headline_rows.append(dict(base, status='missing'))
                continue

            scan = pd.read_csv(path)
            scan_frames.append(scan.assign(method_name=spec['label'],
                                           dataset_label=dataset_label))
            head = scan[scan['headline']].iloc[0]
            matched = scan[scan['matched']]
            info = read_run_info(method, dataset)
            if int(head['n_niches']) != int(head['n_target']):
                base['note'] = ('target not reachable; best reachable '
                                f'{int(head["n_niches"])} clusters')
            headline_rows.append(dict(
                base, status='done',
                n_target=int(head['n_target']), n_clusters=int(head['n_niches']),
                resolution=head['leiden_resolution'],
                NMI=head['NMI'], ARI=head['ARI'],
                n_matched_resolutions=len(matched),
                NMI_matched_mean=matched['NMI'].mean() if len(matched) else None,
                ARI_matched_mean=matched['ARI'].mean() if len(matched) else None,
                embedding_minutes=(info.get('embedding_seconds') or 0) / 60 or None,
                coverage=info.get('coverage', 1.0),
                n_cells=info.get('n_cells')))

            batch_path = (RESULTS / method
                          / f'{dataset}_{method}_batch_integration_metrics.csv')
            if batch_path.exists():
                batch_frames.append(pd.read_csv(batch_path).assign(
                    method=method, method_name=spec['label'],
                    dataset=dataset, dataset_label=dataset_label))

    headline = pd.DataFrame(headline_rows)
    headline.to_csv(OUT / 'headline.csv', index=False)
    scan_all = pd.concat(scan_frames)
    scan_all.to_csv(OUT / 'scan.csv', index=False)
    scan_all[scan_all['in_sweep']].to_csv(OUT / 'sweep.csv', index=False)
    matched_rows(scan_all).to_csv(OUT / 'matched.csv', index=False)
    if batch_frames:
        pd.concat(batch_frames).to_csv(OUT / 'batch.csv', index=False)

    missing = headline[headline['status'] == 'missing']
    print(f'Wrote {OUT}: {len(headline)} headline rows '
          f'({len(missing)} missing).')
    if len(missing):
        print(missing[['method', 'dataset']].to_string(index=False))


if __name__ == '__main__':
    main()
