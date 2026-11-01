"""Post-hoc Harmony on a saved embedding (e.g. TERRA).

Reads ``<method>/<dataset>_<method>_embedded_clustered.h5ad`` from the
permanent results, integrates its embedding over sections with Harmony exactly
as run_banksy.py does (harmonypy.run_harmony on the full embedding, section as
batch, random_state 0) and writes the result as method ``<method>_harmony``.
Single-section datasets are left unintegrated, as for BANKSY. This is forced
integration. Writes the embedding only; job_wrapper.sh then clusters and
scores it with score.py.

Run in the banksy environment (run_harmony.lsf).
"""
import argparse
import json
import sys
from pathlib import Path

import harmonypy
import numpy as np
import scanpy as sc

sys.path.append(str(Path(__file__).parent))
from bench_utils import (FINAL_OUTPUT_FOLDER_PATH, SECTION_KEY, Timer,  # noqa: E402
                         save_embedding)


def main(args):
    in_dir = Path(FINAL_OUTPUT_FOLDER_PATH) / args.method
    prefix = f'{args.dataset}_{args.method}'
    base_info = json.load(open(in_dir / f'{prefix}_run_info.json'))
    base_key = base_info['emb_key']
    method = f'{args.method}_harmony'

    adata = sc.read_h5ad(in_dir / f'{prefix}_embedded_clustered.h5ad')
    adata.obsm = {base_key: adata.obsm[base_key]}   # drop other embeddings
    adata.obs = adata.obs.drop(columns=['predicted_niche'],  # base method's clusters
                               errors='ignore')
    multi_section = adata.obs[SECTION_KEY].nunique() > 1

    emb_key = base_key
    with Timer() as t_harmony:
        if multi_section:
            harmony_out = harmonypy.run_harmony(
                np.asarray(adata.obsm[base_key], dtype='float64'), adata.obs,
                SECTION_KEY, random_state=0)
            emb_key = f'{base_key}_harmony'
            adata.obsm[emb_key] = np.asarray(harmony_out.Z_corr)
            if adata.obsm[emb_key].shape[0] != adata.n_obs:  # harmonypy<0.0.10
                adata.obsm[emb_key] = adata.obsm[emb_key].T

    save_embedding(adata, method, emb_key, dict(
        # embedding time of the base method plus the Harmony step
        embedding_seconds=(base_info.get('embedding_seconds') or 0) + t_harmony.seconds,
        harmony_seconds=t_harmony.seconds,
        params=dict(base_params=base_info.get('params'), base_method=args.method,
                    base_emb_key=base_key,
                    integration=('harmony (post-hoc, forced)' if multi_section
                                 else 'none (single section)'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--dataset', required=True)
    main(parser.parse_args())
