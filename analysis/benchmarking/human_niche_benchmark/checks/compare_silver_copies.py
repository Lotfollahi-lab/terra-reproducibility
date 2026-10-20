"""Check that the lustre silver copy used by the benchmark (SILVER_FOLDER_PATH)
is identical to the NFS copy for all benchmark files: counts, genes, cell IDs,
coordinates and niche labels. Exits non-zero on any difference.

    python checks/compare_silver_copies.py     (banksy env, ~32 GB RAM)
"""
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

sys.path.append(str(Path(__file__).parent.parent))
from bench_utils import DATASETS, SILVER_FOLDER_PATH  # noqa: E402

NFS = '/nfs/team361/sb75/DATASETS/silver'   # reference copy
LUSTRE = SILVER_FOLDER_PATH

all_ok = True
for name, cfg in DATASETS.items():
    files = sorted(f for f in os.listdir(f'{NFS}/{name}') if f.endswith('.h5ad')
                   and any(p in f for p in cfg['file_patterns']))
    for f in files:
        a, b = ad.read_h5ad(f'{NFS}/{name}/{f}'), ad.read_h5ad(f'{LUSTRE}/{name}/{f}')
        Xa, Xb = sp.csr_matrix(a.X), sp.csr_matrix(b.X)
        checks = {
            'shape': a.shape == b.shape,
            'genes': list(a.var_names) == list(b.var_names),
            'cell_id': (a.obs['cell_id'].astype(str).values
                        == b.obs['cell_id'].astype(str).values).all(),
            'counts': Xa.shape == Xb.shape and (Xa != Xb).nnz == 0,
            'spatial': np.allclose(np.asarray(a.obsm['spatial'], float),
                                   np.asarray(b.obsm['spatial'], float), equal_nan=True),
            'niche': (a.obs[cfg['niche_key']].astype(str).values
                      == b.obs[cfg['niche_key']].astype(str).values).all(),
        }
        ok = all(checks.values())
        all_ok &= ok
        print(f"{'OK  ' if ok else 'DIFF'} {name}/{f} {a.shape} "
              f"{'' if ok else {k: v for k, v in checks.items() if not v}}", flush=True)
print('ALL IDENTICAL' if all_ok else 'DIFFERENCES FOUND')
sys.exit(0 if all_ok else 1)
