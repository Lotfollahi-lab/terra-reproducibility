"""Checks of run_cifm.py against the authors' CIFM code (cifm env, GPU).

(a) The authors' tutorial (test.ipynb in the CIFM snapshot): ``model.embed`` on
    their adata.h5ad reproduces the embedding values printed in the notebook,
    and run_cifm's tiled embedding equals ``model.embed`` on the same data.
(b) A benchmark section (smallest xhs1000 section, ~18k cells, 4,948 genes):
    run_cifm's path (genes written into the model's channel order, tiles with a
    halo) equals the authors' path (``channel_matching`` + whole-graph
    ``model.embed``).

    bsub -G s10396 -q inference -n 4 -M 32GB \\
        -R "select[mem>32GB] rusage[mem=32GB] span[hosts=1]" \\
        -gpu "num=1:mode=exclusive_process" -W 1:00 -o test_cifm_%J.out \\
        "$PYTHON_CIFM checks/test_cifm.py"      (from the benchmark folder, after: source config.sh)
"""
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from bench_utils import SECTION_KEY, load_benchmark_dataset  # noqa: E402
from run_cifm import (CIFM_MODEL_PATH, embed_section, ensembl_ids,  # noqa: E402
                      load_model, normalized_panel, snapshot_revision,
                      to_model_genes)

TOL = 1e-3   # absolute; notebook values are printed with 4 decimals

# Rows 0, 1, 2, -3, -2, -1 of the embedding printed in test.ipynb: first three
# and last three dimensions.
NOTEBOOK = {
    0: ([-0.4326, -0.8625, 0.1121], [0.4980, 0.3855, -0.1965]),
    1: ([-0.6833, -0.9950, 0.1927], [-0.2064, 0.6193, 0.0387]),
    2: ([-0.2099, -0.9877, 0.3462], [0.2102, 0.6807, -0.2155]),
    -3: ([-0.0187, -0.8444, 0.3058], [0.1030, 0.8362, -0.1859]),
    -2: ([-0.5535, -0.8201, 0.7805], [-0.1402, 0.5221, -0.3520]),
    -1: ([-0.9339, -0.8467, 0.0600], [0.0406, 0.3608, 0.3418]),
}

failures = []


def report(name, value, tol=TOL):
    ok = value <= tol
    print(f'{"PASS" if ok else "FAIL"}  {name}: max |diff| = {value:.2e} (tol {tol:.0e})', flush=True)
    if not ok:
        failures.append(name)


def check_tutorial(device):
    model, model_genes, _ = load_model(device)
    adata = sc.read_h5ad(CIFM_MODEL_PATH / 'adata.h5ad')
    # The tutorial's channel_matching is the identity here: same genes, same order.
    assert list(adata.var_names) == list(model_genes), 'tutorial genes != model channels'
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    with torch.no_grad():
        ref = model.embed(adata).float().cpu().numpy()
    print(f'tutorial: {adata.n_obs} cells, embedding {ref.shape}')
    diff = max(max(np.abs(ref[r, :3] - np.array(head)).max(),
                   np.abs(ref[r, -3:] - np.array(tail)).max())
               for r, (head, tail) in NOTEBOOK.items())
    report('(a) tutorial embed() vs. values printed in test.ipynb', diff)

    xy = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    ours, _, n_tiles, _ = embed_section(model, adata.X.tocsr(), xy, device,
                                        max_tile_cells=3000)
    print(f'tutorial: {n_tiles} tiles')
    report('(a) tiled embedding vs. embed() on the tutorial data',
           float(np.abs(ours - ref).max()))


def check_benchmark_section(device):
    adata = load_benchmark_dataset('xhs1000-39b_1p')
    section = adata.obs[SECTION_KEY].value_counts().idxmin()
    adata = adata[(adata.obs[SECTION_KEY] == section).values].copy()
    print(f'benchmark: {section}, {adata.n_obs} cells, {adata.n_vars} genes')

    panel, _ = normalized_panel(adata)
    ensembl = ensembl_ids('xhs1000-39b_1p', panel.var_names.astype(str))
    xy = np.asarray(adata.obsm['spatial'], dtype=np.float64)

    # run_cifm path
    model, model_genes, _ = load_model(device)
    X, matched = to_model_genes(panel.X, ensembl, model_genes)
    ours, _, n_tiles, _ = embed_section(model, X, xy, device, max_tile_cells=2000)
    print(f'benchmark: {matched.sum()} of {len(matched)} genes matched, {n_tiles} tiles')
    del model
    torch.cuda.empty_cache()

    # Authors' path: channel_matching (replaces the input layer), whole graph
    model_ref, _, _ = load_model(device)
    source = torch.load(CIFM_MODEL_PATH / 'models_cifm' / 'channel2ensembl.pt',
                        weights_only=True)
    model_ref.channel_matching([[e] for e in ensembl], source)
    panel.obsm['spatial'] = xy
    with torch.no_grad():
        ref = model_ref.embed(panel).float().cpu().numpy()
    report('(b) run_cifm (gene expansion + tiles) vs. channel_matching + embed()',
           float(np.abs(ours - ref).max()))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device {device}; CIFM revision {snapshot_revision()[0]}')
    check_tutorial(device)
    check_benchmark_section(device)
    if failures:
        sys.exit(f'{len(failures)} check(s) failed: {failures}')
    print('All CIFM checks passed.')
