"""Checks of run_aido_tissue.py against the authors' AIDO.Tissue code (aido_tissue env, GPU).

(a) The authors' embedding recipe: ``mgen predict`` with
    experiments/GB.Tissue/emb.xenium.yaml on their demo file (586 cells), pooled as in
    the README (max over the tokens), for the shipped 3M checkpoint and for 60M (the
    benchmark model), vs. run_aido_tissue's embed on the same file (kNN over the whole
    file, as the authors).
(b) Benchmark sections (smallest section of chr78, xhs1000 and chl59):
    - run_aido_tissue's aligned counts equal align_adata
      (experiments/GB.Cell/cell_utils.py) on 2,000 cells;
    - the section written in the authors' format: the neighbour lists equal
      SpatialDataGenerator's, and the input rows of 2,000 cells equal its
      __getitem__ byte for byte;
    - chr78: ``mgen predict`` (60M) on the whole section equals run_aido_tissue's
      embedding.
    vhb3003 (Visium HD): the genes matched through the Ensembl fallback are listed.
(c) Determinism: embed run twice on the chr78 section gives identical embeddings.
(d) Neighbour-order sensitivity (measurement only; the protocol keeps the silver
    order): the cells of an xhs1000 section and of a Visium HD window are shuffled,
    which changes the tie-breaks among equidistant neighbours; cosine of each cell's
    embedding to the one in silver order, for cells with and without ties.

    bsub -G s10396 -q inference -n 8 -M 64GB \\
        -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]" \\
        -gpu "num=1:mode=exclusive_process" -W 3:00 -o test_aido_tissue_%J.out \\
        "$PYTHON_AIDO_TISSUE checks/test_aido_tissue.py"   (from the benchmark folder, after: source config.sh)
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
import run_aido_tissue as rat  # noqa: E402
from bench_utils import (DATASETS, SECTION_KEY, SILVER_FOLDER_PATH,  # noqa: E402
                         ensembl_ids, load_benchmark_dataset, patched)
from modelgenerator import backbones  # noqa: E402
from modelgenerator.data.data import SpatialDataGenerator  # noqa: E402
from modelgenerator.main import cli_main  # noqa: E402

TOL = 1e-3
DEMO = rat.MODEL_ROOT / 'GB.Tissue_data' / 'processed_fetal_lung_visium_xenium.xenium.convert.h5ad'
CHECKPOINTS = {'gb_tissue_3m': rat.MODEL_ROOT / 'GB.Tissue-3M',
               'gb_tissue_60m': rat.MODEL_PATH}
N_ROWS = 2000
failures = []


def report(name, value, tol=TOL):
    ok = value <= tol
    print(f'{"PASS" if ok else "FAIL"}  {name}: max |diff| = {value:.2e} (tol {tol:.0e})',
          flush=True)
    if not ok:
        failures.append(name)


def cosine(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(axis=1)


def authors_predict(h5ad, backbone, workdir, label='cell_type'):
    """`mgen predict` with the authors' emb.xenium.yaml (data file, backbone and output
    folders replaced; checkpoints from the local snapshots), then the README pooling:
    rows with NaN dropped, max over the tokens."""
    cfg = yaml.safe_load(open(rat.CODE_PATH / 'experiments' / 'GB.Tissue' / 'emb.xenium.yaml'))
    data = cfg['data']['init_args']
    data['path'] = str(Path(h5ad).parent)
    for split in ['train_split_files', 'valid_split_files', 'test_split_files']:
        data[split] = [Path(h5ad).name]
    data['filter_columns'] = [label, 'x', 'y']
    cfg['model']['init_args']['backbone']['class_path'] = f'modelgenerator.backbones.{backbone}'
    cfg['trainer']['default_root_dir'] = str(workdir / 'logs')
    cfg['trainer']['callbacks']['init_args']['output_dir'] = str(workdir / 'pred')
    workdir.mkdir(parents=True, exist_ok=True)
    config = workdir / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    argv = sys.argv
    sys.argv = ['mgen', 'predict', '--config', str(config)]
    try:
        with patched(backbones.gb_tissue_3m, 'model_path', str(CHECKPOINTS['gb_tissue_3m'])), \
                patched(backbones.gb_tissue_60m, 'model_path', str(CHECKPOINTS['gb_tissue_60m'])):
            cli_main()
    finally:
        sys.argv = argv
    files = sorted((workdir / 'pred').glob('predict_predictions_0_*.pt'),
                   key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    out = []
    for i, f in enumerate(files):
        batch = torch.load(f, map_location='cpu', weights_only=False)
        pred = batch['predictions']
        hidden = (pred.last_hidden_state if hasattr(pred, 'last_hidden_state') else pred)[0]
        hidden = hidden[~torch.any(hidden.isnan(), dim=1)]
        out.append(hidden.max(dim=0)[0].float().numpy())
        if 'uid' in batch and int(np.asarray(batch['uid']).ravel()[0]) != i:
            raise RuntimeError(f'{f}: uid {batch["uid"]} is not batch {i}.')
    return np.stack(out)


def write_authors_format(path, X, xy):
    """A section as the authors' data module reads it: X in the model's gene order,
    obs x / y, a numeric dummy label first (a string label with a comma would be
    parsed as numbers)."""
    obs = pd.DataFrame({'label': 0.0, 'x': xy[:, 0], 'y': xy[:, 1]},
                       index=np.arange(len(xy)).astype(str))
    ad.AnnData(X=sp.csr_matrix(X), obs=obs).write_h5ad(path)


def check_demo(device, tmp):
    demo = ad.read_h5ad(DEMO)
    X = sp.csr_matrix(demo.X, dtype=np.float32)
    xy = np.stack([demo.obs.x, demo.obs.y], axis=1)
    rows = rat.nearest_cells(xy)
    print(f'demo: {demo.n_obs} cells, {demo.n_vars} genes')
    for name, path in CHECKPOINTS.items():
        backbone, _, _ = rat.load_backbone(device, getattr(backbones, name), path)
        ours, _ = rat.embed(backbone, X, rows, device, num_workers=0, log_every=0)
        del backbone
        torch.cuda.empty_cache()
        ref = authors_predict(DEMO, name, tmp / f'demo_{name}')
        print(f'  {name}: min cosine {cosine(ours, ref).min():.6f}, '
              f'identical rows {(ours == ref).all(axis=1).mean():.3f}')
        report(f'(a) {name}: run_aido_tissue vs. mgen predict on the authors\' demo',
               float(np.abs(ours - ref).max()))


def load_align_adata():
    spec = importlib.util.spec_from_file_location(
        'gb_cell_utils', rat.CODE_PATH / 'experiments' / 'GB.Cell' / 'cell_utils.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.align_adata


def smallest_section(dataset):
    adata = load_benchmark_dataset(dataset)
    section = adata.obs[SECTION_KEY].value_counts().idxmin()
    return adata[(adata.obs[SECTION_KEY] == section).values].copy(), section


def ours_aligned(dataset, adata, genes, genes_ensembl):
    panel = adata.var_names.astype(str).to_numpy()
    cols, by_ensembl, conflicts = rat.match_genes(panel, ensembl_ids(dataset, panel),
                                                  genes, genes_ensembl)
    return rat.aligned_counts(adata.X, cols, len(genes)), by_ensembl, conflicts


def check_section(dataset, device, backbone, genes, genes_ensembl, tmp, rng, predict=False):
    adata, section = smallest_section(dataset)
    X, by_ensembl, _ = ours_aligned(dataset, adata, genes, genes_ensembl)
    xy = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    print(f'{dataset} {section}: {adata.n_obs} cells, {len(by_ensembl)} genes by Ensembl ID')
    sample = np.sort(rng.choice(adata.n_obs, min(N_ROWS, adata.n_obs), replace=False))

    # gene alignment vs. align_adata. align_adata matches exact symbols only, so the
    # Ensembl-fallback columns (e.g. Xenium UGT1A1 -> UGT1A) are compared separately:
    # each must hold its panel gene's raw counts.
    part = ad.AnnData(X=sp.csr_matrix(adata.X[sample], dtype=np.float32),
                      var=pd.DataFrame(index=adata.var_names.astype(str)))
    ref, _ = load_align_adata()(part)
    ref_X = ref.X.toarray() if sp.issparse(ref.X) else np.asarray(ref.X)
    ours_X = X[sample].toarray()
    gene_pos = {g: i for i, g in enumerate(genes)}
    panel_pos = {g: j for j, g in enumerate(adata.var_names.astype(str))}
    fallback = [gene_pos[t] for t in by_ensembl.values()]
    symbol = np.setdiff1d(np.arange(len(genes)), fallback)
    report(f'(b) {dataset}: aligned counts vs. align_adata (exact-symbol columns)',
           float(np.abs(ours_X[:, symbol] - ref_X[:, symbol]).max()), tol=0)
    if by_ensembl:
        raw = sp.csr_matrix(adata.X[sample]).toarray()
        diff = max(float(np.abs(ours_X[:, gene_pos[t]] - raw[:, panel_pos[s]]).max())
                   for s, t in by_ensembl.items())
        report(f'(b) {dataset}: Ensembl-fallback columns = panel counts {by_ensembl}',
               diff, tol=0)
        if np.abs(ref_X[:, fallback]).max() != 0:
            failures.append(f'(b) {dataset}: align_adata filled a fallback column')

    # input rows vs. SpatialDataGenerator
    path = tmp / f'{dataset}_{section}.h5ad'
    write_authors_format(path, X, xy)
    generator = SpatialDataGenerator(str(path), neighbor_num=rat.NEIGHBOR_NUM,
                                     filter_cols=['label', 'x', 'y'], rename_cols=None)
    rows = rat.nearest_cells(xy)
    same_lists = np.array_equal(generator.neighbor_indices, rows)
    print(f'{"PASS" if same_lists else "FAIL"}  (b) {dataset}: neighbour lists vs. '
          'SpatialDataGenerator', flush=True)
    if not same_lists:
        failures.append(f'(b) {dataset}: neighbour lists')
    ours_rows = rat.NeighbourRows(X, rows)
    differ = [i for i in sample
              if not np.array_equal(generator[i]['sequences'], ours_rows[i].numpy())]
    report(f'(b) {dataset}: input rows vs. SpatialDataGenerator ({len(sample)} cells)',
           float(len(differ)), tol=0)

    if predict:
        ours, _ = rat.embed(backbone, X, rows, device, log_every=0)
        ref = authors_predict(path, 'gb_tissue_60m', tmp / f'{dataset}_predict', label='label')
        print(f'  min cosine {cosine(ours, ref).min():.6f}, identical rows '
              f'{(ours == ref).all(axis=1).mean():.3f}')
        report(f'(b) {dataset}: run_aido_tissue vs. mgen predict (60M, whole section)',
               float(np.abs(ours - ref).max()))
        again, _ = rat.embed(backbone, X, rows, device, log_every=0)
        report('(c) two runs of embed', float(np.abs(again - ours).max()), tol=0)
        cell = rat.embed(backbone, X, np.arange(len(xy))[:, None], device, log_every=0)[0]
        cell_again = rat.embed(backbone, X, np.arange(len(xy))[:, None], device, log_every=0)[0]
        report('(c) two runs of embed, cell alone', float(np.abs(cell - cell_again).max()), tol=0)
    return adata, X, xy


def check_fallback(genes, genes_ensembl):
    cfg = DATASETS['vhb3003-3b_1p']
    folder = Path(cfg.get('silver_folder', SILVER_FOLDER_PATH)) / 'vhb3003-3b_1p'
    var = ad.read_h5ad(next(folder.glob('adata_batch0.h5ad')), backed='r').var
    panel = var.index.astype(str).to_numpy()
    cols, by_ensembl, conflicts = rat.match_genes(panel, var['ensembl_id'].astype(str).to_numpy(),
                                                  genes, genes_ensembl)
    print(f'vhb3003: {(cols >= 0).sum()} of {len(panel)} genes matched, {len(by_ensembl)} by '
          f'Ensembl ID (e.g. {dict(list(by_ensembl.items())[:8])}), conflicts {conflicts}')
    wrong = {g: t for g, t in by_ensembl.items() if g == t}
    if wrong:
        failures.append(f'vhb3003 fallback maps genes to themselves: {wrong}')


def order_sensitivity(name, X, xy, backbone, device, rng):
    """Shuffle the cells: equidistant neighbours are then broken differently."""
    rows = rat.nearest_cells(xy)
    diag_dist = rat.cKDTree(xy).query(xy, k=rat.NEIGHBOR_NUM + 2)[0]
    k = rat.NEIGHBOR_NUM
    tied = ((diag_dist[:, k] == diag_dist[:, k + 1])
            | (np.diff(diag_dist[:, 1:k + 1], axis=1) == 0).any(axis=1))
    base, _ = rat.embed(backbone, X, rows, device, log_every=0)
    perm = rng.permutation(len(xy))
    inverse = np.argsort(perm)
    rows_p = rat.nearest_cells(xy[perm])
    shuffled, _ = rat.embed(backbone, X[perm], rows_p, device, log_every=0)
    shuffled = shuffled[inverse]
    # neighbour lists in original indices
    lists_p = perm[rows_p][inverse]
    changed = (lists_p != rows).any(axis=1)
    cos = cosine(base, shuffled)
    print(f'(d) {name}: {len(xy)} cells, {tied.mean():.2%} with tied neighbours, '
          f'{changed.mean():.2%} with a changed neighbour list')
    for label, mask in [('tied', tied), ('untied', ~tied), ('list changed', changed),
                        ('list unchanged', ~changed)]:
        if mask.any():
            c = cos[mask]
            print(f'    {label}: mean cosine {c.mean():.5f}, p1 {np.quantile(c, 0.01):.5f}, '
                  f'min {c.min():.5f}', flush=True)
    if (~changed).any() and np.abs(base[~changed] - shuffled[~changed]).max() > 0:
        print('    note: cells with unchanged neighbour lists differ (obs order otherwise '
              'enters the embedding)')


def visium_window(n_target=5000):
    """A square window of about n_target bins from the middle of a vhb3003 section."""
    cfg = DATASETS['vhb3003-3b_1p']
    folder = Path(cfg.get('silver_folder', SILVER_FOLDER_PATH)) / 'vhb3003-3b_1p'
    adata = ad.read_h5ad(folder / 'adata_batch0.h5ad', backed='r')
    labelled = ~adata.obs[cfg['niche_key']].isna().values
    xy = np.asarray(adata.obsm['spatial'], dtype=np.float64) * cfg['coord_scale']
    centre = np.median(xy[labelled], axis=0)
    half = np.sqrt(n_target) * 8 / 2   # 8 um bins
    inside = np.where(labelled & np.all(np.abs(xy - centre) <= half, axis=1))[0]
    part = adata[inside].to_memory()
    return part, xy[inside]


def check_cached_gather(X, xy, backbone, device, n=300):
    """rat.embed (cached gather_data) vs. the authors' gather_data, bitwise, on n cells
    with and without neighbours."""
    rows = rat.nearest_cells(xy)[:n]
    for label, r in (('with neighbours', rows), ('cell alone', rows[:, :1])):
        ours, _ = rat.embed(backbone, X, r, device, log_every=0)
        assert rat.mgen_backbones.gather_data is rat.authors_gather_data
        ref = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(len(r)):
                seq = rat.NeighbourRows(X, r)[i][None].to(device)
                hidden = backbone(input_ids=seq).last_hidden_state[0]
                if r.shape[1] == 1:
                    hidden = hidden[:X.indptr[r[i, 0] + 1] - X.indptr[r[i, 0]] + rat.N_DEPTH_TOKENS]
                ref.append(hidden.max(dim=0).values.float().cpu().numpy())
        report(f'(c) cached gather_data vs. authors\' gather_data ({label}, {n} cells)',
               float(np.abs(ours - np.stack(ref)).max()), tol=0)


if __name__ == '__main__':
    skip_sensitivity = '--skip-sensitivity' in sys.argv
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rat.check_gpu(device)
    print(f'ModelGenerator {rat.code_revision()}; GB.Tissue-60M {rat.snapshot_revision()[0]}')
    rat.seed_everything()
    rng = np.random.default_rng(0)
    genes = rat.gene_list()
    genes_ensembl = rat.gene_map(genes)
    with tempfile.TemporaryDirectory(dir=rat.MODEL_ROOT.parent / 'human_niche_benchmark') as tmp:
        tmp = Path(tmp)
        check_demo(device, tmp)
        backbone, _, _ = rat.load_backbone(device)
        check_section('chr78-11b_1p', device, backbone, genes, genes_ensembl, tmp, rng,
                      predict=True)
        check_section('chl59-8b_1p', device, backbone, genes, genes_ensembl, tmp, rng)
        _, X, xy = check_section('xhs1000-39b_1p', device, backbone, genes, genes_ensembl,
                                 tmp, rng)
        check_cached_gather(X, xy, backbone, device)
        check_fallback(genes, genes_ensembl)
        if skip_sensitivity:
            print('(d) skipped (--skip-sensitivity)')
        else:
            order_sensitivity('xhs1000 section', X, xy, backbone, device, rng)
        if not skip_sensitivity:
            part, xy = visium_window()
            panel = part.var_names.astype(str).to_numpy()
            cols, _, _ = rat.match_genes(panel, part.var['ensembl_id'].astype(str).to_numpy(),
                                         genes, genes_ensembl)
            order_sensitivity('Visium HD window', rat.aligned_counts(part.X, cols, len(genes)),
                              xy, backbone, device, rng)
    if failures:
        sys.exit(f'{len(failures)} check(s) failed: {failures}')
    print('All AIDO.Tissue checks passed.')
