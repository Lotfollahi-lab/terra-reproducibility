"""Re-cluster and re-score a saved embedding (no re-embedding).

Used when only the clustering protocol changes, e.g. the lowest-resolution tie
rule. Overwrites the method's metric CSV and the clusters in its h5ad.

    python rescore.py --method banksy --dataset xhs1000-39b_1p
"""
import argparse
import json
import sys
from pathlib import Path

import scanpy as sc

sys.path.append(str(Path(__file__).parent))
from bench_utils import (FINAL_OUTPUT_FOLDER_PATH, cluster_and_score,  # noqa: E402
                         method_output_dir, save_results)


def main(args):
    # Read the saved embedding from the permanent (NFS) results; write the
    # updated outputs to the job's working folder (copied back by the wrapper).
    in_dir = Path(FINAL_OUTPUT_FOLDER_PATH) / args.method
    out_dir = method_output_dir(args.method)
    prefix = f'{args.dataset}_{args.method}'
    adata = sc.read_h5ad(in_dir / f'{prefix}_embedded_clustered.h5ad')
    emb_key = json.load(open(in_dir / f'{prefix}_run_info.json'))['emb_key']
    metrics_df = cluster_and_score(adata, args.method, emb_key, out_dir)
    print(metrics_df.to_string(index=False))
    save_results(adata, out_dir / f'{prefix}_embedded_clustered.h5ad')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--dataset', required=True)
    main(parser.parse_args())
