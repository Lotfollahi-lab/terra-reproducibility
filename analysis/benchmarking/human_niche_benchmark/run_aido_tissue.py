"""AIDO.Tissue zero-shot cell embeddings for the human niche benchmark.

AIDO.Tissue (Gong et al., bioRxiv 2025, 10.1101/2025.07.04.663102; renamed GB.Tissue
by GenBio AI in July 2026) is a transformer over the expressed genes of a cell and its
k nearest cells. The code is genbio-ai/ModelGenerator
(modelgenerator.backbones.gb_tissue_60m), the weights genbio-ai/GB.Tissue-60M (12
layers, width 512). This script runs the authors' embedding recipe
(experiments/GB.Tissue/emb.xenium.yaml and the embedding section of its README) with
the pretrained weights and no fine-tuning:
- input as built by CellWithNeighborDataModule (SpatialDataGenerator): the raw counts
  of the cell and of its 10 nearest cells in (x, y) (cKDTree, nearest first),
  each aligned to the model's 19,264 genes, concatenated (11 x 19,264);
- the authors' backbone forward, unchanged (GenBioCellSpatialFoundation.forward):
  per cell log1p(counts / total x 1e4) and 2 depth tokens; only expressed genes become
  tokens; 2D rotary ids (gene index, rank of the cell); output = the tokens of the
  centre cell (its expressed genes and its 2 depth tokens);
- batch size 1 and bf16 autocast (the config's batch_size 1 and precision bf16);
- the cell embedding is the maximum over the centre cell's tokens (512-d), the
  pooling of the README.

Batch size 1 is part of the recipe, not a speed setting: the backbone normalizes by
the sum over the whole batch (``X.sum()``) and its flash-attention call attends to
padding tokens, so in larger batches a cell's embedding would depend on its
batch-mates.

The authors' gather_data is swapped, inside embed only, for a copy that caches its
constant rank-offset tensor (cached_gather_data; identical outputs, ~5x faster).

The authors' DataModule and Lightning writer are replaced by a loop that builds the
same input rows (byte-identical; checks/test_aido_tissue.py) and pools on the GPU,
because:
- SpatialDataGenerator builds one kNN over every row of its file, so neighbours would
  cross sections. Here the kNN is per section;
- the writer keeps every prediction in memory and writes one file per cell
  (1.5M files for Visium HD), merged in glob order.

Genes: matched to the model's gene list by symbol, as the tissue README and
align_adata do. A panel gene whose symbol is not in the list falls back to its silver
Ensembl ID, mapped to the list with the authors' bionty map (build_map of
modelgenerator.cell.utils, used by their GB.Cell data modules; computed once by
envs/aido_tissue.sh), if no other panel gene has that list gene. This recovers the
old symbols of the Visium HD panel (H3F3A -> H3-3A, MARCH7 -> MARCHF7, ...). Genes the
model does not have (immunoglobulin and TCR constant genes, clustered
protocadherins, ...) are dropped and do not count towards the total, as in the
authors' alignment.

Guards (each stops the run): the code checkout, the installed modelgenerator and the
weights are the pinned ones; every encoder weight equals the checkpoint (no newly
initialized weights); attention and hidden dropout are 0 (the flash-attention call
applies its dropout even in eval mode); the GPU supports flash-attention 2; every
cell's output has exactly its expressed genes + 2 tokens.

The result h5ad keeps, for interpretation (not scored):
- obsm['X_aido_tissue_cell']: the same model and pooling on the cell alone (no
  neighbours), i.e. without spatial context.

Writes the embedding only; job_wrapper.sh then clusters and scores it with
score.py. Run in the aido_tissue environment (envs/aido_tissue.sh) on a GPU:
    SCRIPT=run_aido_tissue.py PYTHON_ENV=PYTHON_AIDO_TISSUE DATASET=xhs49-1b_1p \\
        bsub -G s10396 -q tiger22-inference -n 8 -M 64GB \\
        -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]" \\
        -gpu "num=1:mode=exclusive_process" -W 12:00 < run_method.lsf
(README: settings per dataset.)
"""
import argparse
import hashlib
import importlib.metadata
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')   # compute nodes have no internet
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # deterministic cuBLAS

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

# Local copies (envs/aido_tissue.sh): the authors' code (GitHub), checkpoint (Hugging
# Face) and the gene list's Ensembl map.
MODEL_ROOT = Path(os.environ.get('AIDO_TISSUE_MODEL_ROOT',
                                 '/lustre/scratch126/cellgen/lotfollahi/jr43/models'))
CODE_PATH = MODEL_ROOT / 'ModelGenerator_code'
MODEL_PATH = MODEL_ROOT / 'GB.Tissue-60M'
GENE_MAP_PATH = MODEL_ROOT / 'GB.Tissue_gene_map' / 'gene_list_ensembl.tsv'
CODE_REVISION = 'bda6f2da297ba886e3662b49eea9928feacbfd98'
MODEL_REVISION = 'e414516407edb487a0df07ad33ab031ade3f95b6'
GENE_MAP_SHA256 = '1080180e6149251ddc3bcad6101ca077646a592fc1484daaf96e8fa49f029371'   # envs/aido_tissue.sh

import modelgenerator  # noqa: E402
import modelgenerator.backbones.backbones as mgen_backbones  # noqa: E402
from modelgenerator.backbones import gb_tissue_60m  # noqa: E402
from modelgenerator.data import gather_data as authors_gather_data  # noqa: E402
from modelgenerator.data.data import next_16x  # noqa: E402

sys.path.append(str(Path(__file__).parent))
from bench_utils import (SECTION_KEY, Timer, ensembl_ids,  # noqa: E402
                         frac_duplicate_rows, git_commit, hf_snapshot_revision,
                         load_benchmark_dataset, save_embedding)

METHOD = 'aido_tissue'
EMB_KEY = 'X_aido_tissue'
CELL_EMB_KEY = 'X_aido_tissue_cell'
SEED = 42                 # emb.xenium.yaml seed_everything (no random step is used)
NEIGHBOR_NUM = 10         # emb.xenium.yaml neighbor_num (every GB.Tissue config uses 10)
N_DEPTH_TOKENS = 2        # appended to every cell by the backbone
NUM_WORKERS = 4           # emb.xenium.yaml num_workers
# Files of the installed modelgenerator that must equal the pinned checkout.
CODE_FILES = ['backbones/backbones.py', 'backbones/base.py', 'data/data.py',
              'data/base.py', 'tasks/tasks.py', 'callbacks.py',
              'huggingface_models/cellspatialfoundation/modeling_cellspatialfoundation.py',
              'huggingface_models/cellspatialfoundation/configuration_cellspatialfoundation.py',
              'huggingface_models/cellspatialfoundation/positional_embeddings.py',
              'cell/gene_lists/OS_scRNA_gene_index.19264.tsv', 'cell/utils.py']
PACKAGES = ['torch', 'flash_attn', 'transformers', 'numpy', 'scipy', 'anndata', 'scanpy',
            'pandas', 'lightning', 'peft', 'bionty', 'modelgenerator']


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def code_revision(code_path=CODE_PATH):
    """The ModelGenerator checkout must be the pinned commit without local changes, and
    the installed modelgenerator must have its files."""
    commit = git_commit(code_path)
    if commit.endswith('+dirty') or not CODE_REVISION.startswith(commit):
        raise ValueError(f'ModelGenerator at {code_path} is {commit}, expected a clean '
                         f'checkout of {CODE_REVISION} (envs/aido_tissue.sh).')
    installed = Path(modelgenerator.__file__).parent
    for name in CODE_FILES:
        if (installed / name).read_bytes() != (code_path / 'modelgenerator' / name).read_bytes():
            raise ValueError(f'Installed {installed / name} differs from the checkout '
                             f'{code_path} (envs/aido_tissue.sh).')
    return CODE_REVISION


def snapshot_revision(model_path=MODEL_PATH):
    """Commit and sha256 of pytorch_model.bin, from huggingface_hub's download metadata."""
    return hf_snapshot_revision(model_path, 'pytorch_model.bin', MODEL_REVISION,
                                'envs/aido_tissue.sh')


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def check_gpu(device):
    if device != 'cuda':
        raise RuntimeError('AIDO.Tissue needs a GPU: its attention is flash-attention only.')
    if torch.cuda.get_device_capability() < (8, 0):
        raise RuntimeError(f'{torch.cuda.get_device_name()} does not support '
                           'flash-attention 2 (compute capability < 8.0).')


def check_weights(backbone, model_path):
    """Every encoder tensor must equal the checkpoint's bert.* tensor (from_pretrained
    initializes missing weights at random without stopping)."""
    ckpt = torch.load(Path(model_path) / 'pytorch_model.bin', map_location='cpu',
                      weights_only=True)
    bert = {k[len('bert.'):]: v for k, v in ckpt.items() if k.startswith('bert.')}
    state = backbone.encoder.state_dict()
    missing = sorted(set(state) - set(bert))
    unused = sorted(set(bert) - set(state))
    if missing or unused:
        raise ValueError(f'Encoder and checkpoint differ: missing {missing[:5]}, '
                         f'unused {unused[:5]}.')
    differ = [k for k in state if not torch.equal(state[k].cpu(), bert[k].to(state[k].dtype))]
    if differ:
        raise ValueError(f'Encoder weights differ from the checkpoint: {differ[:5]}.')
    return dict(n_encoder_tensors=len(state),
                checkpoint_tensors_not_in_encoder=sorted(k for k in ckpt if not k.startswith('bert.')))


def load_backbone(device, backbone_class=gb_tissue_60m, model_path=MODEL_PATH):
    """The authors' backbone (as the Embed task builds it), loaded from the local
    snapshot, with the guards of the module doc."""
    backbone = backbone_class(None, None)
    backbone.model_path = str(model_path)
    backbone.setup()
    config = backbone.encoder.config
    if config.attention_probs_dropout_prob != 0 or config.hidden_dropout_prob != 0:
        raise ValueError('Dropout > 0: the flash-attention call would apply it at inference.')
    if backbone.rope2d_use_xy:
        raise ValueError('Expected the default rotary ids (gene index, cell rank).')
    loading = check_weights(backbone, model_path)
    return backbone.to(device).eval(), config, loading


def gene_list():
    """The model's genes, in input order (the installed copy of the pinned list)."""
    path = Path(modelgenerator.__file__).parent / 'cell' / 'gene_lists' / 'OS_scRNA_gene_index.19264.tsv'
    table = pd.read_csv(path, sep='\t')
    if list(table['index']) != list(range(len(table))) or not table['gene_name'].is_unique:
        raise ValueError(f'{path}: unexpected gene list.')
    return table['gene_name'].astype(str).to_numpy()


def gene_map(genes, path=GENE_MAP_PATH):
    """Ensembl ID of each gene of the list ('' if bionty has none), from the authors'
    build_map (envs/aido_tissue.sh)."""
    if GENE_MAP_SHA256 is not None and file_sha256(path) != GENE_MAP_SHA256:
        raise ValueError(f'{path} has sha256 {file_sha256(path)}, expected {GENE_MAP_SHA256}.')
    table = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    if list(table['gene_name']) != list(genes):
        raise ValueError(f'{path} does not follow the model\'s gene list.')
    return table['ensembl_id'].to_numpy()


def match_genes(panel, panel_ensembl, genes, genes_ensembl):
    """Position of each panel gene in the model's gene list (-1: not used).

    Exact symbol first (align_adata). A panel gene whose symbol is not in the list
    falls back to its Ensembl ID, if exactly one list gene has that ID and no other
    panel gene takes it."""
    position = {g: i for i, g in enumerate(genes)}
    cols = np.array([position.get(g, -1) for g in panel])
    taken = set(cols[cols >= 0])
    by_id = {}
    for i, e in enumerate(genes_ensembl):
        if e:
            by_id.setdefault(e, []).append(i)
    candidates = {}
    for j in np.where(cols < 0)[0]:
        targets = by_id.get(panel_ensembl[j], [])
        if len(targets) == 1 and targets[0] not in taken:
            candidates.setdefault(targets[0], []).append(j)
    by_ensembl, conflicts = {}, {}
    for target, js in candidates.items():
        if len(js) == 1:
            cols[js[0]] = target
            by_ensembl[str(panel[js[0]])] = str(genes[target])
        else:
            conflicts[str(genes[target])] = [str(panel[j]) for j in js]
    if len(set(cols[cols >= 0])) != (cols >= 0).sum():
        raise RuntimeError('Two panel genes on one model gene.')
    return cols, by_ensembl, conflicts


def aligned_counts(X, cols, n_genes):
    """Raw counts in the model's gene order (CSR, float32); unmatched genes dropped,
    model genes without a panel gene are 0."""
    X = sp.csr_matrix(X)
    keep = np.where(cols >= 0)[0]
    X = X[:, keep].tocoo()
    out = sp.csr_matrix((X.data.astype(np.float32), (X.row, cols[keep][X.col])),
                        shape=(X.shape[0], n_genes), dtype=np.float32)
    out.eliminate_zeros()   # stored entries = expressed genes (= the cell's tokens)
    out.sort_indices()
    if (out.data < 0).any():
        raise ValueError('Negative counts.')
    return out


def nearest_cells(xy, k=NEIGHBOR_NUM):
    """SpatialDataGenerator._precompute_neighbors on one section: the k + 1 nearest
    cells (the first is taken to be the cell itself and skipped by the authors)."""
    if len(xy) < k + 1:
        raise ValueError(f'Section with {len(xy)} cells: fewer than {k + 1}.')
    _, indices = cKDTree(xy).query(xy, k=k + 1)
    return indices


def neighbour_diagnostics(xy, indices, k=NEIGHBOR_NUM):
    """Ties in the neighbour lists (diagnostics only; a separate query, so the lists
    used above are not touched)."""
    dist, _ = cKDTree(xy).query(xy, k=k + 2)
    n = len(xy)
    return dict(
        median_knn_distance=float(np.median(dist[:, k])),
        # the k-th and (k+1)-th nearest are equally far: the neighbour set is a tie-break
        frac_neighbour_set_tied=float((dist[:, k] == dist[:, k + 1]).mean()),
        # two neighbours equally far: their order (rotary cell index) is a tie-break
        frac_neighbour_order_tied=float((np.diff(dist[:, 1:k + 1], axis=1) == 0).any(axis=1).mean()),
        # the authors skip the first entry as the cell itself
        n_self_not_first=int((indices[:, 0] != np.arange(n)).sum()))


class NeighbourRows(Dataset):
    """SpatialDataGenerator.__getitem__'s 'sequences': the counts of the cell, then of its
    neighbours nearest first, concatenated (float32)."""

    def __init__(self, X, rows):
        self.X, self.rows = X, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[self.rows[i]].toarray().reshape(-1))


_RANK_OFFSETS = {}


def cached_gather_data(data, labels, pad_token_id):
    """modelgenerator.data.gather_data with its constant rank-offset tensor cached.

    The authors' version rebuilds that tensor from a Python list of one entry per
    input position (211,926 for 11 cells) on every call, three times per forward
    pass, which costs ~70 ms per cell on the CPU. The tensor (values, int64 dtype,
    device) and every operation are the same, so the outputs are identical; checks (a)
    and (b) compare against the authors' unpatched pipeline."""
    value_nums = labels.sum(1)
    max_num = next_16x(max(value_nums))

    fake_data = torch.full((data.shape[0], max_num), pad_token_id, device=data.device)
    data = torch.hstack([data, fake_data])

    fake_label = torch.full((labels.shape[0], max_num), 1, device=labels.device)
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = torch.tensor(-float('Inf'), device=labels.device)

    key = (labels.shape[1], str(labels.device))
    if key not in _RANK_OFFSETS:
        _RANK_OFFSETS[key] = torch.tensor(
            [(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
    labels += _RANK_OFFSETS[key]

    labels = torch.hstack([labels, fake_label])
    fake_label_gene_value, fake_label_gene_idx = labels.topk(max_num)

    new_data = torch.gather(data, 1, fake_label_gene_idx)
    padding_labels = fake_label_gene_value == 1

    return new_data, padding_labels


class fast_gather_data:
    """Use cached_gather_data in the backbone inside this block only (the authors'
    pipeline in checks/test_aido_tissue.py runs unpatched)."""

    def __enter__(self):
        if mgen_backbones.gather_data is not authors_gather_data:
            raise RuntimeError('modelgenerator gather_data is already replaced.')
        mgen_backbones.gather_data = cached_gather_data

    def __exit__(self, *exc):
        mgen_backbones.gather_data = authors_gather_data


@torch.inference_mode()
def embed(backbone, X, rows, device, num_workers=NUM_WORKERS, log_every=50000):
    """Max over each centre cell's output tokens, one cell per forward pass.

    rows: (n, c) row indices into X, centre cell first (c = 1: the cell alone).
    Returns the embedding and each cell's number of output tokens; checks that it is
    the cell's expressed genes + the 2 depth tokens."""
    n = len(rows)
    expected = np.diff(X.indptr)[rows[:, 0]] + N_DEPTH_TOKENS
    out = torch.empty((n, backbone.encoder.config.hidden_size), dtype=torch.float32,
                      device=device)
    n_tokens = np.empty(n, dtype=np.int64)
    loader = DataLoader(NeighbourRows(X, rows), batch_size=1, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    start = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16), fast_gather_data():
        for i, sequence in enumerate(loader):
            result = backbone(input_ids=sequence.to(device, non_blocking=True))
            hidden = result.last_hidden_state[0]
            if rows.shape[1] == 1:
                # single-cell path: all tokens are returned, padding last
                hidden = hidden[:expected[i]]
            n_tokens[i] = hidden.shape[0]
            out[i] = hidden.max(dim=0).values.float()
            if log_every and (i + 1) % log_every == 0:
                rate = (time.perf_counter() - start) / (i + 1)
                print(f'  {i + 1}/{n} cells, {rate * 1e3:.1f} ms/cell', flush=True)
    if (n_tokens != expected).any():
        raise RuntimeError(f'{(n_tokens != expected).sum()} cells with an unexpected number '
                           'of output tokens.')
    return out.cpu().numpy(), n_tokens


def sequence_lengths(X, rows):
    """Encoder input length per cell: expressed genes + 2 per cell, padded to 16."""
    per_cell = np.diff(X.indptr)[rows] + N_DEPTH_TOKENS
    return (np.ceil(per_cell.sum(axis=1) / 16) * 16).astype(np.int64)


def main(args):
    adata = load_benchmark_dataset(args.dataset, subsample=args.subsample)
    dataset = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    adata.uns['benchmark']['dataset_name'] = dataset
    print(adata)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    check_gpu(device)
    code_commit = code_revision()
    commit, weights_sha256 = snapshot_revision()
    seed_everything(SEED)

    with Timer() as t_load:
        backbone, config, loading = load_backbone(device)
    genes = gene_list()
    if len(genes) != config.max_position_embeddings:
        raise ValueError('Gene list and model input width differ.')
    panel = adata.var_names.astype(str).to_numpy()
    cols, by_ensembl, conflicts = match_genes(panel, ensembl_ids(args.dataset, panel),
                                              genes, gene_map(genes))
    X = aligned_counts(adata.X, cols, len(genes))
    panel_counts = np.asarray(adata.X.sum(axis=0)).ravel()
    unmatched = panel[cols < 0]
    frac_counts_unmatched = float(panel_counts[cols < 0].sum() / panel_counts.sum())
    print(f'{(cols >= 0).sum()} of {len(panel)} panel genes matched ({len(by_ensembl)} by '
          f'Ensembl ID); {len(unmatched)} unmatched with {frac_counts_unmatched:.2%} of the '
          f'counts: {unmatched[:20].tolist()}', flush=True)

    xy_all = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    emb = np.full((adata.n_obs, config.hidden_size), np.nan, dtype=np.float32)
    cell_emb = np.full_like(emb, np.nan)
    seq_len = np.zeros(adata.n_obs, dtype=np.int64)
    sections, seconds = [], t_load.seconds
    for section in adata.obs[SECTION_KEY].cat.categories:
        rows = np.where((adata.obs[SECTION_KEY] == section).values)[0]
        if not len(rows):
            continue
        Xs = X[rows]
        neighbours = nearest_cells(xy_all[rows])
        diag = neighbour_diagnostics(xy_all[rows], neighbours)
        with Timer() as t_section:
            emb[rows], _ = embed(backbone, Xs, neighbours, device, args.num_workers)
        with Timer() as t_cell:
            cell_emb[rows], _ = embed(backbone, Xs, np.arange(len(rows))[:, None], device,
                                      args.num_workers)
        seconds += t_section.seconds + t_cell.seconds
        seq_len[rows] = sequence_lengths(Xs, neighbours)
        sections.append(dict(
            section=section, n_cells=len(rows), **diag,
            n_zero_count_cells=int((np.diff(Xs.indptr) == 0).sum()),
            median_tokens=float(np.median(seq_len[rows])), max_tokens=int(seq_len[rows].max()),
            seconds=t_section.seconds, cell_only_seconds=t_cell.seconds))
        print(f'{section}: {len(rows)} cells, median {np.median(seq_len[rows]):.0f} tokens '
              f'(max {seq_len[rows].max()}), 10-NN distance {diag["median_knn_distance"]:.1f} um, '
              f'{t_section.seconds:.0f} s + {t_cell.seconds:.0f} s cell only', flush=True)
    adata.obsm[EMB_KEY] = emb
    adata.obsm[CELL_EMB_KEY] = cell_emb

    if not (np.isfinite(emb).all() and np.isfinite(cell_emb).all()):
        raise ValueError('Missing or non-finite AIDO.Tissue embeddings.')
    n_expressed = np.diff(X.indptr)
    save_embedding(adata, METHOD, EMB_KEY, dict(
        embedding_seconds=seconds,
        model=dict(repo='genbio-ai/GB.Tissue-60M (formerly AIDO.Tissue-60M)', revision=commit,
                   weights_sha256=weights_sha256, path=str(MODEL_PATH),
                   n_parameters=sum(p.numel() for p in backbone.parameters()),
                   config=config.to_dict(), **loading),
        code=dict(repo='genbio-ai/ModelGenerator', commit=code_commit, path=str(CODE_PATH),
                  installed=str(Path(modelgenerator.__file__).parent)),
        gene_map=dict(path=str(GENE_MAP_PATH), sha256=file_sha256(GENE_MAP_PATH)),
        n_genes_panel=len(panel), n_genes_matched=int((cols >= 0).sum()),
        genes_matched_by_ensembl=by_ensembl, gene_fallback_conflicts=conflicts,
        unmatched_genes=unmatched.tolist(), frac_counts_unmatched=frac_counts_unmatched,
        n_zero_count_cells=int((n_expressed == 0).sum()),
        median_expressed_genes_per_cell=float(np.median(n_expressed)),
        median_tokens=float(np.median(seq_len)), max_tokens=int(seq_len.max()),
        ms_per_cell=1e3 * sum(s['seconds'] for s in sections) / adata.n_obs,
        frac_duplicate_embeddings=frac_duplicate_rows(emb),
        frac_duplicate_cell_embeddings=frac_duplicate_rows(cell_emb),
        cell_embedding_key=CELL_EMB_KEY,
        device=torch.cuda.get_device_name(),
        peak_gpu_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
        package_versions={p: importlib.metadata.version(p) for p in PACKAGES},
        params=dict(seed=SEED, neighbor_num=NEIGHBOR_NUM, batch_size=1,
                    precision='bf16 autocast (Lightning bf16-mixed)',
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    input='raw counts of the cell and its 10 nearest cells (cKDTree on '
                          'x, y, per section), aligned to the 19,264 genes',
                    normalization='in the backbone: per cell log1p(counts / total x 1e4) '
                                  'over the matched genes, + 2 depth tokens',
                    gene_matching='exact symbol; else silver Ensembl ID via the authors\' '
                                  'bionty map (build_map)',
                    embedding='max over the centre cell\'s output tokens (expressed genes + '
                              '2 depth tokens)',
                    spatial_context='10 nearest cells per section; no coordinates in the model',
                    integration='none'),
        sections=sections))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--subsample', type=int, default=None,
                        help='Random subset of cells (smoke tests only).')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS,
                        help='DataLoader workers building the input rows (speed only).')
    main(parser.parse_args())
