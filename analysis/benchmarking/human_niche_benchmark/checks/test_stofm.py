"""Checks of run_stofm.py against the authors' SToFM code (stofm env, GPU).

(a) The authors' demo (seekgene_sample in the checkpoint folder):
    - run_stofm's cell encoder on their tokenized cells (hf.dataset) reproduces their
      ce_emb.npy;
    - their get_embeddings.py loop on their data.h5ad + ce_emb.npy (seed 1, as
      get_embeddings.sh), with exact kNN as run_stofm, reproduces their
      stofm_emb.npy. The same loop with scanpy's approximate kNN for the virtual
      cells (extraction.py's CPU fallback) is reported for comparison.
(b) Tokens: run_stofm.tokenize equals the authors' tokenize_anndata (the class in
    preprocessing/preprocess.py, exec'd from the file) + Geneformer's create_dataset,
    on benchmark sections of Xenium, CosMx and Visium HD.
(c) run_stofm.embed_section equals the authors' get_embeddings.py loop (load_data on
    the same files, zero-initialized embeddings, no torch.no_grad), on the demo and
    on a benchmark section with obs-order chunks and >= 8192 cells. The SE(2) batch
    size (1 vs 4) does not change the embedding.
(d) Seed: seed 1 vs 2 on a section where the virtual cells' PCA is randomized
    (500-2560 cells).
(e) Obs order: split_graph cuts grid squares with >= 2000 cells into chunks in obs
    order. The section of (c) in silver order vs a random order vs sorted by x.

    bsub -G s10396 -q inference -n 8 -M 96GB \\
        -R "select[mem>96GB] rusage[mem=96GB] span[hosts=1]" \\
        -gpu "num=1:mode=exclusive_process" -W 3:00 -o test_stofm_%J.out \\
        "$PYTHON_STOFM checks/test_stofm.py"      (from the benchmark folder, after: source config.sh)
"""
import ast
import contextlib
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from bench_utils import CACHE_FOLDER_PATH, SECTION_KEY, load_benchmark_dataset  # noqa: E402
from run_stofm import (ALPHA, LEIDEN_RES, SPLIT_NUM, STOFM_CODE_PATH,  # noqa: E402
                       STOFM_MODEL_PATH, SToFM_Collator, check_weights, code_revision,
                       embed_section, encode_cells, exact_neighbors, gene_tokens,
                       grid_squares, load_data, load_models, normalized_coordinates,
                       seed_everything, tokenize)

DEMO = STOFM_MODEL_PATH / 'seekgene_sample'
TOL = 1e-5

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


def describe(name, a, b):
    c = cosine(a, b)
    print(f'INFO  {name}: max |diff| {np.abs(a - b).max():.2e}, cosine mean {c.mean():.5f}, '
          f'p1 {np.quantile(c, 0.01):.4f}, min {c.min():.4f}', flush=True)
    return c


def authors_loop(model, data_path, emb_path, device, exact=True, batch_size=4):
    """get_embeddings.py l.89-120 for one data folder, with load_data on its CPU path
    (device -1), inside exact_neighbors() if exact."""
    with exact_neighbors() if exact else contextlib.nullcontext():
        graphs = load_data(data_path=str(data_path), spatial_path=None, emb_path=str(emb_path),
                           new_emb=False, device=-1, filter=False, split_num=SPLIT_NUM,
                           leiden_res=LEIDEN_RES, alpha=ALPHA)
    adata = sc.read_h5ad(data_path)
    data_num = len(adata)
    del adata
    dataloader = DataLoader(graphs, collate_fn=SToFM_Collator(mask=False, mask_pair=False),
                            batch_size=batch_size, shuffle=False)
    embeddings = torch.zeros(data_num, 256)
    for i, graph in enumerate(dataloader):
        indices = graph['indices']
        graph = {k: v.to(device) for k, v in graph.items()}
        model.eval()
        torch.cuda.empty_cache()
        output = model(**graph)
        node_rep = output['last_hidden_state']
        embeddings[indices[indices != -1]] = node_rep[indices != -1].clone().detach().cpu()
        del graph, output, node_rep
    return embeddings.numpy()


def authors_tokenizer_class():
    """SToFMTranscriptomeTokenizer, exec'd from preprocessing/preprocess.py (the file runs
    its example at module level, so only its imports and the class are executed)."""
    tree = ast.parse((STOFM_CODE_PATH / 'preprocessing' / 'preprocess.py').read_text())
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef))]
    namespace = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'preprocess.py', 'exec'), namespace)
    return namespace['SToFMTranscriptomeTokenizer']


def section_data(dataset, which='smallest', max_cells=None):
    adata = load_benchmark_dataset(dataset)
    counts = adata.obs[SECTION_KEY].value_counts()
    section = counts.idxmin() if which == 'smallest' else which
    adata = adata[(adata.obs[SECTION_KEY] == section).values].copy()
    if max_cells is not None:
        adata = adata[:max_cells].copy()
    print(f'{dataset} {section}: {adata.n_obs} cells, {adata.n_vars} genes', flush=True)
    return adata


def check_demo(model, cell_encoder, device, work):
    ds = load_from_disk(str(DEMO / 'hf.dataset'))
    ours = encode_cells(cell_encoder, [np.array(c) for c in ds['input_ids']], device)
    ref = np.load(DEMO / 'ce_emb.npy')
    describe('(a) demo cell encoder vs ce_emb.npy', ours, ref)
    report('(a) demo: cell encoder on hf.dataset vs the authors\' ce_emb.npy',
           float(np.abs(ours - ref).max()), tol=1e-4)

    shipped = np.load(DEMO / 'stofm_emb.npy')
    seed_everything(1)
    exact = authors_loop(model, DEMO / 'data.h5ad', DEMO / 'ce_emb.npy', device, exact=True)
    c = describe('(a) demo: authors\' loop, exact kNN vs stofm_emb.npy', exact, shipped)
    report('(a) demo: authors\' loop (exact kNN) reproduces stofm_emb.npy (1 - mean cosine)',
           float(1 - c.mean()), tol=1e-3)
    seed_everything(1)
    approx = authors_loop(model, DEMO / 'data.h5ad', DEMO / 'ce_emb.npy', device, exact=False)
    describe('(a) demo: authors\' loop, approximate kNN (CPU fallback) vs stofm_emb.npy',
             approx, shipped)

    # (c) on the demo: embed_section on the same inputs
    xy = np.asarray(ad.read_h5ad(DEMO / 'data.h5ad').obsm['spatial'], dtype=np.float64)
    seed_everything(1)
    ours, *_ = embed_section(model, np.load(DEMO / 'ce_emb.npy'), xy, work / 'demo', device)
    report('(c) demo: embed_section vs the authors\' loop', float(np.abs(ours - exact).max()))


def check_tokens():
    tokenizer = authors_tokenizer_class()({}, nproc=1)
    for dataset, which, max_cells in [('xhs1000-39b_1p', 'smallest', None),
                                      ('chl59-8b_1p', 'smallest', 20000),
                                      ('vhb3003-3b_1p', 'smallest', 5000)]:
        adata = section_data(dataset, which, max_cells)
        ids, loc, tokens, norm, _ = gene_tokens(dataset, adata.var_names.astype(str))
        ours, n_detected = tokenize(adata.X, loc, tokens, norm)
        # the authors' input: var['ensembl_id'] as run_stofm maps it, n_counts over all genes
        a = ad.AnnData(X=sp.csr_matrix(adata.X), var=pd.DataFrame({'ensembl_id': ids},
                                                                  index=adata.var_names))
        # preprocess.py assigns the np.matrix of X.sum(axis=1); same values, flattened
        a.obs['n_counts'] = np.asarray(a.X.sum(axis=1)).ravel()
        a.obs['filter_pass'] = True
        cells, metadata = tokenizer.tokenize_anndata(a)
        ref = tokenizer.create_dataset(cells, metadata)['input_ids']
        same = len(ref) == len(ours) and all(list(r) == o.tolist() for r, o in zip(ref, ours))
        print(f'{"PASS" if same else "FAIL"}  (b) {dataset}: tokens identical to the authors\' '
              f'tokenizer ({len(ours)} cells, median {np.median(n_detected):.0f} genes, '
              f'{(n_detected >= 2048).sum()} truncated)', flush=True)
        if not same:
            failures.append(f'(b) {dataset}')


def check_section(model, cell_encoder, device, work):
    adata = section_data('xhs1000-39b_1p', '1000_batch19')
    _, loc, tokens, norm, _ = gene_tokens('xhs1000-39b_1p', adata.var_names.astype(str))
    seed_everything(1)
    cells, _ = tokenize(adata.X, loc, tokens, norm)
    cell_emb = encode_cells(cell_encoder, cells, device)
    xy = np.asarray(adata.obsm['spatial'], dtype=np.float64)

    seed_everything(1)
    ours, _, _, diag = embed_section(model, cell_emb, xy, work / 'section_b4', device, 4)
    print(f'INFO  section: {diag}', flush=True)
    folder = work / 'section_authors'
    folder.mkdir()
    ad.AnnData(obs=pd.DataFrame(index=adata.obs_names), obsm={'spatial': xy}).write_h5ad(
        folder / 'data.h5ad')
    np.save(folder / 'ce_emb.npy', cell_emb)
    seed_everything(1)
    ref = authors_loop(model, folder / 'data.h5ad', folder / 'ce_emb.npy', device, exact=True)
    report('(c) section: embed_section vs the authors\' loop', float(np.abs(ours - ref).max()))
    seed_everything(1)
    approx = authors_loop(model, folder / 'data.h5ad', folder / 'ce_emb.npy', device,
                          exact=False)
    describe('(c) section: approximate kNN (CPU fallback) vs exact', approx, ref)
    seed_everything(1)
    b1, *_ = embed_section(model, cell_emb, xy, work / 'section_b1', device, 1)
    report('(c) section: SE(2) batch size 1 vs 4', float(np.abs(b1 - ours).max()))

    # (e) obs order: the same cells in a random order and sorted by x. Cells in grid
    # squares that split_graph cuts in obs order are reported apart from the others,
    # which only see the change through the virtual cells.
    square = grid_squares(normalized_coordinates(xy))
    in_split = np.bincount(square)[square] // SPLIT_NUM >= 2
    print(f'INFO  (e) silver order: {diag["frac_cells_obs_order_split"]:.2f} of cells in '
          f'obs-order chunks, chunk/square 10-NN ratio median '
          f'{diag["obs_order_chunk_knn_ratio_median"]}, p90 '
          f'{diag["obs_order_chunk_knn_ratio_p90"]}', flush=True)
    for name, order in [('random order', np.random.default_rng(0).permutation(len(xy))),
                        ('sorted by x', np.lexsort((xy[:, 1], xy[:, 0])))]:
        seed_everything(1)
        e, _, _, d = embed_section(model, cell_emb[order], xy[order],
                                   work / f'order_{name.split()[0]}', device)
        back = np.empty_like(e)
        back[order] = e
        print(f'INFO  (e) {name}: chunk/square 10-NN ratio median '
              f'{d["obs_order_chunk_knn_ratio_median"]}, p90 '
              f'{d["obs_order_chunk_knn_ratio_p90"]}', flush=True)
        describe(f'(e) {name} vs silver order, cells in obs-order chunks',
                 back[in_split], ours[in_split])
        describe(f'(e) {name} vs silver order, other cells', back[~in_split], ours[~in_split])


def check_seed(model, cell_encoder, device, work):
    adata = section_data('chr78-11b_1p', 'smallest')
    _, loc, tokens, norm, _ = gene_tokens('chr78-11b_1p', adata.var_names.astype(str))
    cells, _ = tokenize(adata.X, loc, tokens, norm)
    cell_emb = encode_cells(cell_encoder, cells, device)
    xy = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    out = {}
    for seed in (1, 2):
        seed_everything(seed)
        out[seed], *_ = embed_section(model, cell_emb, xy, work / f'seed{seed}', device)
    describe('(d) seed 1 vs 2', out[2], out[1])


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    check_weights()
    print(f'device {device}; SToFM code {code_revision()}', flush=True)
    model, cell_encoder, _ = load_models(device)
    cache = Path(CACHE_FOLDER_PATH) / 'stofm'
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as tmp:
        work = Path(tmp)
        check_demo(model, cell_encoder, device, work)
        check_tokens()
        check_section(model, cell_encoder, device, work)
        check_seed(model, cell_encoder, device, work)
    code_revision()   # the checkout is still clean
    if failures:
        sys.exit(f'{len(failures)} check(s) failed: {failures}')
    print('All SToFM checks passed.')
