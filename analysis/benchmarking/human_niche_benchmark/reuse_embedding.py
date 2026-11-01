"""Score another embedding already saved in a finished result as its own method.

Reads ``<source>/<dataset>_<source>_embedded_clustered.h5ad`` from the
permanent results and registers ``obsm[<emb_key>]`` as method ``<method>``,
e.g. TERRA-96M's cell embedding (encoder pass over the index cell only, no
neighbour tokens) as ``terra_cell_emb``:

    python reuse_embedding.py --source terra --emb_key cell_emb \
        --method terra_cell_emb --dataset xhs1000-39b_1p

With ``--neighbour_average`` the embedding is replaced by its mean over each
cell and its 10 spatial neighbours (per-section kNN graph with self loop, as
for the neighbourhood-expression PCA baseline), i.e. spatial context added
after the fact to a cell-level embedding (method e.g.
``terra_cell_emb_nbavg``).

``--clustering kmeans`` scores the new method with k-means at the common
cluster number instead of the Leiden scan (e.g. the cell-type composition
baseline as ``cell_type_composition_kmeans``).

Writes the embedding only; job_wrapper.sh then clusters and scores it with
score.py. Run in the banksy environment (reuse_embedding.lsf).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import squidpy as sq

sys.path.append(str(Path(__file__).parent))
from bench_utils import (FINAL_OUTPUT_FOLDER_PATH, N_NEIGHS,  # noqa: E402
                         SECTION_KEY, save_embedding)


def neighbour_average(adata, emb_key):
    """Mean of the embedding over each cell and its spatial neighbours."""
    sq.gr.spatial_neighbors(adata, library_key=SECTION_KEY, coord_type='generic',
                            spatial_key='spatial', n_neighs=N_NEIGHS, set_diag=True)
    W = (adata.obsp['spatial_connectivities'] > 0).astype(np.float32)
    W = sp.diags(1 / np.asarray(W.sum(1)).ravel()) @ W
    return np.asarray(W @ np.asarray(adata.obsm[emb_key], dtype=np.float32))


def main(args):
    in_dir = Path(FINAL_OUTPUT_FOLDER_PATH) / args.source
    prefix = f'{args.dataset}_{args.source}'
    source_info = json.load(open(in_dir / f'{prefix}_run_info.json'))

    adata = sc.read_h5ad(in_dir / f'{prefix}_embedded_clustered.h5ad')
    if args.emb_key not in adata.obsm:
        raise KeyError(f'{args.emb_key} not in {in_dir}/{prefix}_embedded_clustered.h5ad '
                       f'(available: {list(adata.obsm)}).')
    # Keep only the requested embedding and coordinates; drop the source
    # method's graphs and clusters.
    adata.obsm = {k: adata.obsm[k] for k in [args.emb_key, 'spatial'] if k in adata.obsm}
    adata.obsp = {}
    adata.uns = {'benchmark': adata.uns['benchmark']}
    adata.obs = adata.obs.drop(columns=['predicted_niche'], errors='ignore')

    emb_key = args.emb_key
    if args.neighbour_average:
        emb_key = f'{args.emb_key}_nbavg'
        adata.obsm[emb_key] = neighbour_average(adata, args.emb_key)
        del adata.obsm[args.emb_key]
        adata.obsp = {}

    save_embedding(adata, args.method, emb_key, dict(
        clustering=args.clustering,
        embedding_seconds=source_info.get('embedding_seconds'),
        params=dict(source_method=args.source, source_emb_key=args.emb_key,
                    neighbour_average=(dict(n_neighs=N_NEIGHS, self_loop=True,
                                            per_section=True)
                                       if args.neighbour_average else None),
                    source_params=source_info.get('params'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='finished method, e.g. terra')
    parser.add_argument('--emb_key', required=True, help='obsm key, e.g. cell_emb')
    parser.add_argument('--method', required=True, help='new method name')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--clustering', choices=['leiden', 'kmeans'], default='leiden',
                        help='clustering used by score.py for the new method')
    parser.add_argument('--neighbour_average', action='store_true',
                        help='average the embedding over the 10 spatial neighbours')
    main(parser.parse_args())
