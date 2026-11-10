"""Niche identification at a fixed range of Leiden resolutions (no matching to the ground-truth niche number).

The main protocol (scoring.py) picks each method's resolutions so that the number of clusters matches the number of
ground-truth niches. Here every method and dataset is instead clustered at the same resolutions (default 0.1, 0.2,
..., 1.0), so methods are compared at equal clustering granularity settings rather than equal cluster numbers.

Same steps as scoring.cluster_and_score otherwise: 15-NN graph on the method's embedding (run_info emb_key, read from
the finished result), Leiden (igraph, 2 iterations, seed 0), NMI/ARI against the ground truth. Methods scored with
k-means in the main protocol (cell-type composition) are clustered with Leiden here like all others.

Writes ``<FINAL_OUTPUT_FOLDER_PATH>/<method>/<dataset>_<method>_fixed_resolution_scores.csv``. Run in the reference
(banksy) environment, one method x dataset per call or array element (score_fixed_resolutions.lsf):
    python score_fixed_resolutions.py --index 1          # 1-based index into pairs()
    python score_fixed_resolutions.py --method terra --dataset xhs49-1b_1p
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.append(str(Path(__file__).parent))
from bench_utils import FINAL_OUTPUT_FOLDER_PATH, METHODS, NOT_APPLICABLE  # noqa: E402
import scoring  # noqa: E402

DATASETS = ['xhs1000-39b_1p', 'xhs49-1b_1p', 'chr78-11b_1p', 'chl59-8b_1p']   # main figure datasets
RESOLUTIONS = [round(0.1 * k, 1) for k in range(1, 11)]
RESULTS = Path(FINAL_OUTPUT_FOLDER_PATH)


def pairs():
    """(method, dataset) pairs with a finished result, in METHODS x DATASETS order."""
    return [(m, d) for m in METHODS for d in DATASETS
            if (m, d) not in NOT_APPLICABLE
            and (RESULTS / m / f'{d}_{m}_embedded_clustered.h5ad').exists()]


def main(args):
    scoring.check_environment()
    if args.index is not None:
        todo = pairs()
        if args.index > len(todo):
            print(f'Index {args.index} > {len(todo)} pairs: nothing to do.')
            return
        method, dataset = todo[args.index - 1]
    else:
        method, dataset = args.method, args.dataset
    resolutions = args.resolutions or RESOLUTIONS
    prefix = f'{dataset}_{method}'
    run_info = json.load(open(RESULTS / method / f'{prefix}_run_info.json'))
    emb_key = run_info['emb_key']
    adata = sc.read_h5ad(RESULTS / method / f'{prefix}_embedded_clustered.h5ad')
    adata = sc.AnnData(obs=adata.obs[['niche_gt']].copy(), obsm={emb_key: np.asarray(adata.obsm[emb_key])})
    print(f'{method} / {dataset}: {adata.n_obs} cells, embedding {emb_key} {adata.obsm[emb_key].shape}', flush=True)

    neighbors_key = f'{emb_key}_neighbors'
    sc.pp.neighbors(adata, n_neighbors=scoring.N_LATENT_NEIGHBORS, use_rep=emb_key, key_added=neighbors_key,
                    random_state=0)
    n_workers = max(1, int(os.environ.get('LSB_DJOB_NUMPROC', os.cpu_count() or 1)))
    scoring._SCAN_STATE.update(adata=adata, neighbors_key=neighbors_key)
    with mp.get_context('fork').Pool(processes=min(n_workers, len(resolutions))) as pool:
        rows = scoring._run_resolutions(pool, resolutions)
    scoring._SCAN_STATE.clear()

    df = pd.DataFrame(rows)
    df.insert(0, 'method', method)
    df.insert(1, 'dataset', dataset)
    df['n_target'] = run_info.get('n_target_clusters')
    df.to_csv(RESULTS / method / f'{prefix}_fixed_resolution_scores.csv', index=False)
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Leiden at fixed resolutions for one method and dataset.')
    parser.add_argument('--index', type=int, default=None, help='1-based index into pairs() (array jobs).')
    parser.add_argument('--method')
    parser.add_argument('--dataset')
    parser.add_argument('--resolutions', type=float, nargs='+', default=None,
                        help=f'Leiden resolutions (default {RESOLUTIONS}).')
    args = parser.parse_args()
    if args.index is None and not (args.method and args.dataset):
        parser.error('use --index or --method and --dataset')
    main(args)
