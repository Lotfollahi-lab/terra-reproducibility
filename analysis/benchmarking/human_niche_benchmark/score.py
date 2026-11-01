"""Cluster and score embeddings with the shared protocol (only scoring entry point).

Two modes, both in the reference (banksy) environment:

- ``--pending`` (run automatically by job_wrapper.sh after every method script):
  scores every ``<method>/<dataset>_<method>_embedded.h5ad`` the method script
  wrote into the job's working folder (BENCH_OUTPUT_DIR), then replaces it by
  the clustered result.
- ``--method M --dataset D``: re-scores a finished result from the permanent
  folder (e.g. after a protocol change), without re-embedding:
      python score.py --method banksy --dataset xhs1000-39b_1p

Writes ``{dataset}_{method}_metric_scores.csv``, the batch-integration metrics
(multi-section datasets), ``{dataset}_{method}_embedded_clustered.h5ad`` and
adds clustering time and the scoring environment to ``*_run_info.json``.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import scanpy as sc

sys.path.append(str(Path(__file__).parent))
from bench_utils import (FINAL_OUTPUT_FOLDER_PATH, OUTPUT_FOLDER_PATH,  # noqa: E402
                         SECTION_KEY, Timer, method_output_dir, save_results)
from scoring import (CLUSTERING, batch_integration_metrics,  # noqa: E402
                     environment_versions)

EMBEDDED_SUFFIX = '_embedded.h5ad'


def score(adata, method, dataset, run_info, out_dir):
    emb_key = run_info['emb_key']
    clustering = run_info.setdefault('clustering', 'leiden')   # set by the embedding step
    with Timer() as t_cluster:
        metrics_df = CLUSTERING[clustering](adata, method, emb_key, out_dir)
    print(metrics_df.to_string(index=False))

    batch_path = out_dir / f'{dataset}_{method}_batch_integration_metrics.csv'
    if adata.obs[SECTION_KEY].nunique() > 1:
        batch_integration_metrics(adata, emb_key).to_csv(batch_path, index=False)

    run_info.update(clustering_seconds=t_cluster.seconds,
                    scoring_environment=dict(python=sys.executable,
                                             **environment_versions()),
                    scored_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(out_dir / f'{dataset}_{method}_run_info.json', 'w') as f:
        json.dump(run_info, f, indent=2, default=str)
    save_results(adata, out_dir / f'{dataset}_{method}_embedded_clustered.h5ad')
    print(f'Scored {method} / {dataset} -> {out_dir}')


def score_pending():
    if 'BENCH_OUTPUT_DIR' not in os.environ:
        raise RuntimeError('--pending scores the job working folder; run it via '
                           'job_wrapper.sh (BENCH_OUTPUT_DIR is not set).')
    pending = sorted(Path(OUTPUT_FOLDER_PATH).glob(f'*/*{EMBEDDED_SUFFIX}'))
    if not pending:
        raise RuntimeError(f'No *{EMBEDDED_SUFFIX} in {OUTPUT_FOLDER_PATH}.')
    for path in pending:
        method = path.parent.name
        dataset = path.name[:-len(f'_{method}{EMBEDDED_SUFFIX}')]
        run_info = json.load(open(path.parent / f'{dataset}_{method}_run_info.json'))
        score(sc.read_h5ad(path), method, dataset, run_info, path.parent)
        path.unlink()   # superseded by *_embedded_clustered.h5ad


def score_final(method, dataset):
    in_dir = Path(FINAL_OUTPUT_FOLDER_PATH) / method
    prefix = f'{dataset}_{method}'
    adata = sc.read_h5ad(in_dir / f'{prefix}_embedded_clustered.h5ad')
    run_info = json.load(open(in_dir / f'{prefix}_run_info.json'))
    score(adata, method, dataset, run_info, method_output_dir(method))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pending', action='store_true')
    parser.add_argument('--method')
    parser.add_argument('--dataset')
    args = parser.parse_args()
    if args.pending == bool(args.method or args.dataset):
        parser.error('use either --pending or --method and --dataset')
    if args.pending:
        score_pending()
    else:
        score_final(args.method, args.dataset)
