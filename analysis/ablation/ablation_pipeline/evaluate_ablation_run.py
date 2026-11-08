"""
This script evaluates one trained ablation run with the niche identification protocol of the mouse ablations
(`analysis/mouse_ablation/stjepa_v2_ablation.ipynb`, stjepa-reproducibility): neighborhood embeddings from the target
encoder (`embed_dataset`), a kNN graph (k = 15), Leiden clustering on the fixed resolution grid 0.6-1.4 (step 0.1;
identical for all variants, i.e. chosen without labels) and NMI and ARI (and AMI, chance-adjusted) against the niche
labels of each label key. If the cells come from two or more sections, it also computes the batch-integration metrics
of the mouse notebook.

Models with special tokens (the batch metatoken) are evaluated twice: with the metatoken removed at inference, as in
the mouse evaluation (`drop`, the default outputs), and with the metatoken kept as in training (`keep`,
`keep_special_tokens=True` of TERRA's `embed_dataset`; outputs prefixed `<dataset_name>_keep_metatoken_`). Both modes get all
metrics, including the batch-integration metrics.

Labels and sections: the obs of the label files (`--label_file_paths`; the job array passes LABEL_FILE_PATHS, by
default the silver files) are joined to the tokenized cells by `cell_id` (an error is raised if an id occurs twice or a
tokenized cell has no row). Each label file is one section; a cell's section is the file its row came from. A label key
may be missing from some files (e.g. a section-specific annotation): its metrics are computed over the cells where it
is set and not its unassigned label (`--unassigned_label`: one value for all keys, or `<key>=<value>` pairs separated
by ';', one per key). All embedded cells are clustered, including unlabelled ones; unlike the mouse
evaluation, which removed unlabelled cells before embedding, they stay in the kNN graph (as in training), so cluster
counts and metric levels are comparable between variants, not with the mouse runs.

Batch integration between the sections (two or more sections with labelled cells; cells 21 and 42 of the mouse
notebook), over the labelled cells (a label for at least one key), the cells the mouse notebook embedded and scored; their
embeddings are those of the all-cell pass used for the clustering:
- iLISI: `scib_metrics.ilisi_knn` on the 90-nearest-neighbor graph of `scib_metrics.nearest_neighbors.pynndescent`
  (random_state 0), i.e. the scaled median of the per-cell LISI (higher is better; 1: fully mixed).
- ASW: silhouette score of the sections (euclidean) on 10,000 random cells (random_state 0); 0 is best (mixed),
  1 means separated sections.
- MMD: the notebook's `mmd_comparable`: biased MMD^2 between the first two sections (in the order of the label files)
  with an RBF kernel on standardized embeddings, its width the median pairwise distance of 1,000 random cells, on
  2,000 random cells per section (lower is better). The cells are drawn from np.random.default_rng(0) instead of the
  unseeded global state of the notebook, and the distances are computed in row chunks (same values, bounded memory).
These three are label-free: they reward mixing even where the sections differ in region composition. Also written:
- iLISI_max: the asymptotic upper bound of iLISI for the section sizes of the scored cells, (1 / sum p_s^2 - 1) /
  (n_sections - 1) (0.8 at 2:1), which randomly mixed cells do not reach (the per-cell LISI uses about 30 effective
  neighbors; random 2:1 mixing gives about 0.75); iLISI_random: the mean iLISI of the same 90-NN graph with the
  sections permuted over the cells (ILISI_RANDOM_PERMUTATIONS permutations, seeds 0, 1, ...; one permutation varies by
  about 1-2%), i.e. of random mixing for these section sizes; iLISI_relative = iLISI /
  iLISI_random (about 1: as mixed as random; comparable between section ratios).
- ASW_label_<key>, for each label key that labels cells of two or more sections: the section silhouette within each
  label (labels with at least MIN_CELLS_PER_LABEL_SECTION cells in two or more sections; at most
  MAX_CELLS_PER_LABEL random cells each), mean |silhouette| over the cells of a label, averaged over the labels (0:
  mixed within regions; region-conditioned, so it does not reward mixing across regions). The mouse keys each label one
  section, so the mouse runs have none.
- `<prefix>neighborhood_emb_cross_section_neighbors.csv`: per label key, label and section, the mean fraction of a
  cell's neighbors in the k = 15 graph of the clustering that come from other sections, next to the fraction expected
  under random mixing (other sections' share of all embedded cells): label-aware for section-specific keys, it shows
  whether mixing is confined to some regions or uniform (mixing without biology).
scib_metrics (with pynndescent and jax) is imported only if there are two or more sections; jax runs on the CPU.

The embedded dataset follows from the run's tokenized training dataset (data.tokenized_data_folder_path of
model_config.yaml) and `--eval_gene_set` (EVAL_GENE_SET of the campaign's dataset.env, passed by the job arrays and
list_pending_runs.py; `get_eval_dataset_path`), not from the run config, so that the run ids do not depend on it: for a
dataset with one section (OVCA, all_genes), the run's own tokenized training dataset; with two or more sections (mouse,
shared_genes), the shared-gene re-tokenization of the same cells with the run's tokenization (filter_shared_genes.py,
tokenize_dataset.py --gene_set shared_genes), as the published mouse evaluation embedded shared-gene data, so that the
integration metrics do not measure the gene panels. It is tokenized by the pipeline
with the training tokenizer settings, not by `harmonize_tokenize_embed_pipeline` (which needs ~0.5 TB of memory for
the dense-path OVCA normalization arms without the ranked-list cut, and broke ties at the top-64 cut
non-deterministically); the run aborts if its provenance (ablation_provenance.json of tokenize_dataset.py) records
another TERRA code version, tokenization, token dictionary or factor file than the training dataset's, or if a cell id
occurs twice in it. The run aborts if the checkpoint
does not load (`load_checkpoint` otherwise silently returns a randomly initialized model), and, before any metric, if
an embedding is not finite (with the counts per mode, section and labelled/unlabelled). `NUMBA_CPU_NAME=generic`
(unless set) compiles the numba kNN code for the same generic CPU on every node, as in the mouse notebooks, so that the
graph does not depend on the CPU. The embeddings themselves come from the GPU of the job, in bf16 autocast as in
training (`meta.use_bfloat16`), so they can differ slightly between GPU models; `--fp32_embeddings` embeds in fp32
instead (a copy of the model folder with `use_bfloat16: false`). evaluation_complete records the GPU and the precision.

Outputs in `--output_folder_path` (mouse layout `artifacts/ablation/<variant>/<run>/`), per metatoken mode with the
prefix `<dataset_name>_` (drop) or `<dataset_name>_keep_metatoken_` (keep):
- `<prefix>neighborhood_emb_niche_identification_metrics_res_sweep.csv`: one row per resolution and label key
  (`region_key`), the columns of the mouse runs (resolution, n_clusters (over all embedded cells), region_key, NMI,
  ARI), then AMI, n_labelled_cells (the cells scored for the key), n_label_categories (their labels) and
  n_clusters_labelled (the clusters among them), so that differences in granularity can be read off
- `<prefix>neighborhood_emb_batch_integration_metrics.csv` (two or more sections; columns `metric,score` with the rows
  iLISI, ASW and MMD, as in the mouse notebook, then iLISI_max, iLISI_random, iLISI_relative and ASW_label_<key>)
- `<prefix>neighborhood_emb_cross_section_neighbors.csv` (two or more sections; see above)
- `<prefix>neighborhood_emb_avg_cosine_similarity.csv`: collapse statistics (`embedding_statistics`): the average
  pairwise cosine similarity of the embeddings (as in the mouse runs) and of the mean-centered embeddings, kept for
  continuity, although neither detects collapse (centered it is about -1/(n-1) by construction, raw about 1 minus the
  between-cell variance fraction, 0.99 for healthy and collapsed embeddings alike); the spectrum of the centered
  embeddings (participation ratio, PCs for 90% of the variance, the top PC's share, RankMe) and its scale (between-cell
  variance fraction, per-dimension standard deviation), and the collapse flags with their thresholds
- `<prefix>neighborhood_emb_clusters.csv` (section, labels and Leiden cluster per resolution for every cell)
- `<prefix>neighborhood_emb.npy` (float32 embeddings, rows in the order of the clusters CSV)
- `evaluation_complete`, written last, once all modes are done (and removed before an evaluation starts): `<key>:
  <value>` lines with what produced the metrics. Compared with the current inputs (MARKER_INPUT_KEYS): the modes, label
  keys, label files and unassigned label of each key, the sha1 of the evaluation code (this script and the part of
  ablation_utils.py it uses, `evaluation_code_sha1s`), the TERRA code version, the checkpoint (epoch and md5), the
  evaluation data (path, i.e. the training dataset and --eval_gene_set, and md5 of its provenance file; a changed
  EVAL_GENE_SET thus evaluates the trained run again) and the embedding precision; recorded only: the unassigned
  label as given, the modes with batch-integration metrics, the number of evaluated cells, the GPU, the torch version
  and the time.
All modes are evaluated again unless evaluation_complete matches the current inputs and every output file is complete,
or always with `--overwrite`. `--check_only` evaluates nothing: exit code 0 if the evaluation is up to date,
NEEDS_EVALUATION_EXIT_CODE (3) with the reasons if not (also without evaluation_complete or a packaged checkpoint), any
other code on errors.

Usage:
```bash
python evaluate_ablation_run.py --model_folder_path [RUN_FOLDER] --output_folder_path [EVAL_FOLDER] \
    --dataset_name [DATASET_NAME] --label_file_paths [LABEL_FILE ...] --label_keys [KEY ...] \
    --unassigned_label [UNASSIGNED_LABEL] --eval_gene_set [EVAL_GENE_SET] [--fp32_embeddings] \
    [--overwrite | --check_only]
```
"""

import os

# Before scanpy / numba / jax are imported: numba compiles for this CPU name instead of the node's own CPU, and jax
# (iLISI) runs on the CPU, so that it neither depends on the node nor preallocates GPU memory next to torch
os.environ.setdefault("NUMBA_CPU_NAME", "generic")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import sklearn.metrics as skm  # noqa: E402
import yaml  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ablation_utils import (EVAL_GENE_SETS, PROVENANCE_FILE_NAME, SECTION_KEY, file_md5,  # noqa: E402
                            format_unassigned_labels, get_code_version, get_eval_dataset_path, get_section_names,
                            parse_unassigned_labels)

# Same resolution grid as the mouse sweep (base 1.0, 4 steps of 0.1 below and above)
RESOLUTIONS = [round(step * 0.1, 1) for step in range(6, 15)]
# Rows of every batch-integration metrics file (ASW_label_<key> only for keys that label two or more sections)
INTEGRATION_METRICS = ["iLISI", "ASW", "MMD", "iLISI_max", "iLISI_random", "iLISI_relative"]
# Elements of the temporary difference array of the chunked pairwise distances (float32: 256 MB)
DISTANCE_CHUNK_ELEMENTS = 2 ** 26
# Label-conditioned section ASW: labels with at least this many cells in two or more sections, at most this many
# random cells per label
MIN_CELLS_PER_LABEL_SECTION = 10
MAX_CELLS_PER_LABEL = 2000
# Permutations of the sections for the iLISI of random mixing (iLISI_random)
ILISI_RANDOM_PERMUTATIONS = 5
# Collapse flags of the embeddings of a mode (embedding_statistics). Complete collapse: the between-cell variance is
# below this fraction of the mean squared norm (i.e. a raw average cosine similarity above about 0.999; healthy
# embeddings with the raw 0.99 of the mouse runs have about 0.01, cells collapsed onto one point up to 1% noise 1e-4).
# Dimensional collapse: the top PC holds more than this share of the centered variance, i.e. the cells vary along one
# direction only (rank 1 up to 1% noise: about 0.99). Not the participation ratio, which one dominant axis (e.g. two
# sections or niches far apart) pushes below 2 although the remainder is full rank (two niches at +-6 along one axis
# with 0.1 isotropic noise in 768 dimensions: participation ratio 1.5, top PC 0.82). Smaller drops (e.g. a
# participation ratio below 25% of the base run) are for the comparison of runs (compare_ablations.py).
COLLAPSE_MAX_VARIANCE_FRACTION = 1e-3
COLLAPSE_MAX_TOP_PC_FRACTION = 0.95
# evaluation_complete lines compared with the current inputs; the other lines are recorded only
MARKER_INPUT_KEYS = ["modes",
                     "label_keys",
                     "label_file_paths",
                     "unassigned_labels",
                     "eval_code_sha1",
                     "terra_code_version",
                     "checkpoint_epoch",
                     "checkpoint_md5",
                     "eval_data_path",
                     "eval_data_provenance_md5",
                     "embedding_precision"]
NEEDS_EVALUATION_EXIT_CODE = 3


def load_labels(label_file_paths: list, label_keys: list, unassigned_labels: dict) -> pd.DataFrame:
    """
    Labels and sections of all cells of the label files, indexed by `cell_id`: one column per label key, NaN where the
    key is missing (from the file), not set or its unassigned label (`unassigned_labels`), and the section (the label
    file) of each cell.
    """
    if SECTION_KEY in label_keys:
        raise ValueError(f"'{SECTION_KEY}' holds the section of each cell and cannot be a label key.")
    section_names = get_section_names(label_file_paths)
    frames = []
    for file_path, section_name in zip(label_file_paths, section_names):
        obs = sc.read_h5ad(file_path, backed="r").obs
        if "cell_id" not in obs.columns:
            raise KeyError(f"{file_path} has no obs['cell_id'] (the tokenizer cell ids).")
        frame = pd.DataFrame({"cell_id": obs["cell_id"].astype(str).to_numpy(), SECTION_KEY: section_name})
        for key in label_keys:
            if key in obs.columns:
                values = obs[key]
                frame[key] = np.where(values.isna().to_numpy(), None, values.astype(str).to_numpy())
        frames.append(frame)
        print(f"{file_path}: {len(frame)} cells (section {section_name}), label keys "
              f"{[key for key in label_keys if key in obs.columns]}.")
    labels = pd.concat(frames, ignore_index=True)
    absent_keys = [key for key in label_keys if key not in labels.columns]
    if absent_keys:
        raise KeyError(f"Label keys {absent_keys} are in none of the label files.")
    duplicated = labels["cell_id"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(f"{labels.loc[duplicated, 'cell_id'].nunique()} cell ids occur more than once in the label "
                         f"files, e.g. {list(labels.loc[duplicated, 'cell_id'].unique()[:3])}.")
    labels = labels.set_index("cell_id")
    sections = pd.Categorical(labels[SECTION_KEY], categories=section_names)
    labels = labels[label_keys].astype(object)
    for key in label_keys:
        labels[key] = labels[key].where(labels[key].notna() & (labels[key] != unassigned_labels[key]), np.nan)
    labels[SECTION_KEY] = sections
    return labels


def read_marker(marker_file_path: Path) -> dict:
    """Values recorded in `evaluation_complete` (empty if there is none)."""
    if not marker_file_path.exists():
        return {}
    return dict(line.split(": ", 1) for line in marker_file_path.read_text().splitlines() if ": " in line)


def integration_complete(integration_file_path: Path) -> bool:
    """Whether the batch-integration metrics file exists with all metrics."""
    try:
        return set(INTEGRATION_METRICS) <= set(pd.read_csv(integration_file_path)["metric"])
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
        return False


def average_cosine_similarity(X: np.ndarray) -> float:
    """Average cosine similarity over all pairs of distinct rows (rows of norm 0 have no direction and are skipped)."""
    norms = np.linalg.norm(X, axis=1)
    X = X[norms > 0] / norms[norms > 0, None]
    n = X.shape[0]
    if n < 2:
        return np.nan
    S = X.sum(axis=0)
    return float((S @ S - n) / (n * (n - 1)))


def embedding_statistics(embeddings: np.ndarray) -> dict:
    """
    Collapse statistics of the embeddings (one row per cell): the average cosine similarities of the mouse runs, the
    shape of the spectrum of the centered embeddings (eigenvalues of their covariance; participation ratio
    (sum l)^2 / sum l^2, PCs for 90% of the variance, the top PC's share, RankMe (exp of the entropy of the normalized
    singular values), which a small isotropic noise floor inflates) and their scale (between-cell variance fraction
    tr(Cov) / mean ||x||^2, per-dimension standard deviation). A spectrum metric alone misses cells collapsed onto one
    point up to noise, a scale metric alone misses dimensional collapse; the flags use one of each (without any
    between-cell variance, the top PC's share is 1).
    """
    X = embeddings.astype(np.float64)
    X_centered = X - X.mean(axis=0)
    eigenvalues = np.clip(np.linalg.eigvalsh(X_centered.T @ X_centered / len(X))[::-1], 0, None)
    total_variance = eigenvalues.sum()
    if total_variance > 0:
        shares = eigenvalues / total_variance
        singular_value_shares = np.sqrt(eigenvalues) / np.sqrt(eigenvalues).sum()
        singular_value_shares = singular_value_shares[singular_value_shares > 0]
        participation_ratio = float(total_variance ** 2 / (eigenvalues ** 2).sum())
        n_pcs_90 = int(min(np.searchsorted(np.cumsum(shares), 0.9) + 1, len(shares)))
        top_pc_share = float(shares[0])
        rankme = float(np.exp(-(singular_value_shares * np.log(singular_value_shares)).sum()))
    else:
        participation_ratio, n_pcs_90, top_pc_share, rankme = 0.0, 0, 1.0, 0.0
    mean_squared_norm = float((X ** 2).sum(axis=1).mean())
    variance_fraction = float(total_variance / mean_squared_norm) if mean_squared_norm > 0 else 0.0
    dim_std = X_centered.std(axis=0)
    return {"n_cells": len(X),
            "n_dims": X.shape[1],
            "avg_cosine_similarity": average_cosine_similarity(X),
            "avg_cosine_similarity_centered": average_cosine_similarity(X_centered),
            "between_cell_variance_fraction": variance_fraction,
            "participation_ratio": participation_ratio,
            "n_pcs_90_variance": n_pcs_90,
            "top_pc_variance_fraction": top_pc_share,
            "rankme": rankme,
            "mean_dim_std": float(dim_std.mean()),
            "min_dim_std": float(dim_std.min()),
            "complete_collapse": variance_fraction < COLLAPSE_MAX_VARIANCE_FRACTION,
            "dimensional_collapse": bool(top_pc_share > COLLAPSE_MAX_TOP_PC_FRACTION),
            "collapse_max_variance_fraction": COLLAPSE_MAX_VARIANCE_FRACTION,
            "collapse_max_top_pc_variance_fraction": COLLAPSE_MAX_TOP_PC_FRACTION}


def check_finite(embeddings: dict, obs: pd.DataFrame, label_keys: list):
    """
    Raise before any metric if an embedding is not finite (NaN rows would otherwise pass silently into the kNN graph
    and Leiden, or make the metrics fail after both embedding passes), with the counts per mode, section and
    labelled (a label for at least one key) / unlabelled cell.
    """
    labelled = np.where(obs[label_keys].notna().any(axis=1).to_numpy(), "labelled", "unlabelled")
    problems = []
    for mode, mode_embeddings in embeddings.items():
        if len(mode_embeddings) != len(obs):
            raise ValueError(f"'{mode}': {len(mode_embeddings)} embeddings for {len(obs)} cells.")
        non_finite = ~np.isfinite(mode_embeddings).all(axis=1)
        if non_finite.any():
            counts = (pd.DataFrame({SECTION_KEY: obs[SECTION_KEY].to_numpy()[non_finite],
                                    "cells": labelled[non_finite]})
                      .groupby([SECTION_KEY, "cells"], observed=True).size())
            per_section = ", ".join(f"{section} {cells}: {count}" for (section, cells), count in counts.items())
            problems.append(f"'{mode}': {int(non_finite.sum())} of {len(obs)} cells ({per_section})")
    if problems:
        raise FloatingPointError(f"Non-finite embeddings, no metric computed: {'; '.join(problems)}.")


def ilisi_max(batch: np.ndarray) -> float:
    """
    Asymptotic upper bound of iLISI (scaled as scib_metrics.ilisi_knn) for these batch sizes: (1 / sum p^2 - 1) /
    (n_batches - 1), e.g. 0.8 for two batches at 2:1 (random mixing stays below it with ilisi_knn's neighborhoods).
    """
    shares = pd.Series(batch).value_counts(normalize=True).to_numpy()
    return float((1 / (shares ** 2).sum() - 1) / (len(shares) - 1))


def label_conditioned_asw(X: np.ndarray, batch: np.ndarray, labels: pd.Series, seed: int = 0) -> tuple[float, int]:
    """
    Section silhouette within each label that has at least MIN_CELLS_PER_LABEL_SECTION cells in two or more sections
    (at most MAX_CELLS_PER_LABEL random cells per label): mean |silhouette| over the label's cells, averaged over the
    labels (0: the sections are mixed within each region); NaN and 0 labels if no label qualifies.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for label in sorted(labels.dropna().unique()):
        idx = np.flatnonzero((labels == label).to_numpy())
        section_counts = pd.Series(batch[idx]).value_counts()
        sections = section_counts.index[section_counts >= MIN_CELLS_PER_LABEL_SECTION]
        idx = idx[np.isin(batch[idx], sections)]
        if len(sections) < 2:
            continue
        if len(idx) > MAX_CELLS_PER_LABEL:
            idx = np.sort(rng.choice(idx, MAX_CELLS_PER_LABEL, replace=False))
        if len(np.unique(batch[idx])) < 2:
            continue
        scores.append(np.abs(skm.silhouette_samples(X[idx], batch[idx], metric="euclidean")).mean())
    return (float(np.mean(scores)) if scores else np.nan), len(scores)


def cross_section_neighbors(adata: sc.AnnData, neighbors_key: str, label_keys: list) -> pd.DataFrame:
    """
    Per label key, label and section: the mean fraction of the cells' neighbors (kNN graph of `neighbors_key`, all
    embedded cells) from other sections, and the fraction expected under random mixing (the other sections' share of
    all embedded cells).
    """
    distances = adata.obsp[adata.uns[neighbors_key]["distances_key"]].tocsr()
    section = adata.obs[SECTION_KEY].to_numpy()
    rows = np.repeat(np.arange(adata.n_obs), np.diff(distances.indptr))
    other = pd.Series(section[distances.indices] != section[rows]).groupby(rows).mean()
    other_fraction = other.reindex(np.arange(adata.n_obs)).to_numpy()
    section_share = pd.Series(section).value_counts(normalize=True)
    records = []
    for key in label_keys:
        frame = pd.DataFrame({"label": adata.obs[key].to_numpy(), "section": section, "fraction": other_fraction})
        frame = frame[frame["label"].notna()]
        for (label, section_name), group in frame.groupby(["label", "section"], observed=True):
            records.append({"region_key": key,
                            "label": label,
                            "section": section_name,
                            "n_cells": len(group),
                            "cross_section_neighbor_fraction": float(group["fraction"].mean()),
                            "expected_fraction": float(1 - section_share[section_name])})
    return pd.DataFrame(records)


def evaluation_code_sha1s() -> dict:
    """
    sha1 of the code that produces the metrics, recorded in evaluation_complete: this script, and of ablation_utils.py
    only what it uses (the source of the functions imported from it and of the ablation_utils functions they reference,
    the values of the constants), so that edits to the tokenization, setup or launch helpers there do not make every
    evaluation stale. package_run.py is not part of it: the checkpoint md5 covers what it links.
    """
    import inspect

    import ablation_utils

    module_values = vars(ablation_utils)
    pending = [name for name, value in globals().items() if module_values.get(name) is value]
    used = {}
    while pending:
        name = pending.pop()
        value = module_values[name]
        if name in used or name.startswith("__") or inspect.ismodule(value) or inspect.isclass(value):
            continue
        if not inspect.isfunction(value):
            used[name] = repr(value)
        elif value.__module__ == ablation_utils.__name__:
            used[name] = inspect.getsource(value)
            codes = [value.__code__]
            while codes:
                code = codes.pop()
                pending += [global_name for global_name in code.co_names if global_name in module_values]
                codes += [constant for constant in code.co_consts if inspect.iscode(constant)]
    used_source = "".join(f"{name}\n{used[name]}\n" for name in sorted(used))
    return {"evaluate_ablation_run.py": hashlib.sha1(Path(__file__).resolve().read_bytes()).hexdigest(),
            "ablation_utils.py:used": hashlib.sha1(used_source.encode()).hexdigest()}


def get_device_info() -> dict:
    """GPU of the embeddings (cuda:0, as embed_dataset) and torch version, recorded in evaluation_complete."""
    import torch

    return {"gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else
                    "none (CPU: no CUDA autocast, fp32)"),
            "torch_version": f"{torch.__version__} (CUDA {torch.version.cuda})"}


def get_marker_inputs(args, unassigned_labels: dict, model_config: dict) -> dict:
    """
    The inputs of the evaluation as recorded in evaluation_complete (MARKER_INPUT_KEYS): the checkpoint is the one the
    packaged run folder links (model_checkpoint.pt), its epoch read without loading the weights.
    """
    import torch

    code_sha1s = evaluation_code_sha1s()
    checkpoint_file_path = (Path(args.model_folder_path) / "model_checkpoint.pt").resolve()
    checkpoint = torch.load(checkpoint_file_path, map_location="cpu", mmap=True, weights_only=False)
    checkpoint_epoch = checkpoint["epoch"]
    del checkpoint
    eval_dataset_path = get_eval_dataset_path(model_config["data"]["tokenized_data_folder_path"],
                                              model_config["data"]["dataset_name"], args.eval_gene_set)
    provenance_file_path = Path(eval_dataset_path) / PROVENANCE_FILE_NAME
    return {"modes": " ".join(get_modes(args, model_config)),
            "label_keys": " ".join(args.label_keys),
            "label_file_paths": " ".join(args.label_file_paths),
            "unassigned_labels": format_unassigned_labels(unassigned_labels),
            "eval_code_sha1": hashlib.sha1(" ".join(code_sha1s.values()).encode()).hexdigest(),
            "eval_code_files": " ".join(f"{file_name}={sha1}" for file_name, sha1 in code_sha1s.items()),
            "terra_code_version": get_code_version(),
            "checkpoint": str(checkpoint_file_path),
            "checkpoint_epoch": str(checkpoint_epoch),
            "checkpoint_md5": file_md5(checkpoint_file_path),
            "eval_data_path": str(eval_dataset_path),
            "eval_data_provenance_md5": (file_md5(provenance_file_path) if provenance_file_path.exists() else
                                         "none"),
            "embedding_precision": ("fp32" if args.fp32_embeddings or not model_config["meta"]["use_bfloat16"] else
                                    "bf16")}


def get_modes(args, model_config: dict) -> list:
    """Metatoken modes of the run (keeping the metatoken differs from dropping it only with special tokens)."""
    return [mode for mode in args.metatoken_modes if mode == "drop" or model_config["meta"]["special_tokens"]]


def get_file_prefix(args, mode: str) -> str:
    return (f"{args.dataset_name}_{args.emb_key}" if mode == "drop" else
            f"{args.dataset_name}_keep_metatoken_{args.emb_key}")


def evaluation_differences(args, output_folder_path: Path, expected: dict) -> list:
    """
    Why the run must be evaluated (again): evaluation_complete is missing or records other inputs (MARKER_INPUT_KEYS),
    or an output of a mode it records is missing or incomplete (the niche metrics, collapse statistics, clusters and
    embeddings, and with batch-integration metrics their file and the cross-section neighbors). Empty if the
    evaluation is up to date.
    """
    recorded = read_marker(output_folder_path / "evaluation_complete")
    if not recorded:
        return [f"no {output_folder_path / 'evaluation_complete'}"]
    reasons = []
    for key in MARKER_INPUT_KEYS:
        if recorded.get(key) != expected[key]:
            detail = ""
            if key == "eval_code_sha1":
                recorded_sha1s = dict(item.split("=", 1) for item in recorded.get("eval_code_files", "").split())
                changed_files = [file_name for file_name, sha1 in evaluation_code_sha1s().items()
                                 if recorded_sha1s.get(file_name) != sha1]
                detail = f" (changed: {changed_files})"
            reasons.append(f"{key}: recorded {recorded.get(key, '<missing>')!r}, now {expected[key]!r}{detail}")
    integration_modes = recorded.get("batch_integration_metrics", "").split()
    for mode in expected["modes"].split():
        file_prefix = get_file_prefix(args, mode)
        metrics_file_path = output_folder_path / f"{file_prefix}_niche_identification_metrics_res_sweep.csv"
        try:
            missing_keys = sorted(set(args.label_keys) - set(pd.read_csv(metrics_file_path)["region_key"]))
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
            missing_keys = args.label_keys
        if missing_keys:
            reasons.append(f"{metrics_file_path.name} is missing or lacks label keys {missing_keys}")
        integration_file_path = output_folder_path / f"{file_prefix}_batch_integration_metrics.csv"
        if mode in integration_modes and not integration_complete(integration_file_path):
            reasons.append(f"{integration_file_path.name} is missing or incomplete")
        file_names = [f"{file_prefix}_avg_cosine_similarity.csv", f"{file_prefix}_clusters.csv", f"{file_prefix}.npy"]
        if mode in integration_modes:
            file_names.append(f"{file_prefix}_cross_section_neighbors.csv")
        reasons += [f"{file_name} is missing" for file_name in file_names
                    if not (output_folder_path / file_name).exists()]
    return reasons


def pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Euclidean distances as computed in the mouse notebook, np.linalg.norm(A[:, None] - B[None, :], axis=2), in row
    chunks: every distance is computed by the same operations (identical values), but the temporary difference array
    has at most DISTANCE_CHUNK_ELEMENTS elements instead of len(A) x len(B) x dims (24 GB for 2,000 x 2,000 x 768).
    """
    chunk_size = max(1, DISTANCE_CHUNK_ELEMENTS // max(1, B.size))
    return np.concatenate([np.linalg.norm(A[start:start + chunk_size, None] - B[None, :], axis=2)
                           for start in range(0, len(A), chunk_size)])


def mmd_comparable(adata: sc.AnnData, emb_key: str, batch_key: str, n_sub: int = 2000, n_sigma: int = 1000,
                   seed: int = 0) -> float:
    """
    `mmd_comparable` of the mouse notebook (stjepa_v2_ablation.ipynb, cell 21): biased MMD^2 with an RBF kernel between
    the first two batches (categories of `batch_key` in order, here the label files) on embeddings standardized to zero
    mean and unit variance; the kernel width is the median pairwise distance of n_sigma random cells, and n_sub random
    cells per batch are compared. Unlike the notebook, the random cells are drawn from np.random.default_rng(seed)
    instead of the unseeded global state, and the distances are computed in row chunks (identical values).
    """
    rng = np.random.default_rng(seed)
    # Normalize to zero mean, unit variance
    X = StandardScaler().fit_transform(adata.obsm[emb_key])

    # Compute sigma on normalized data
    idx = rng.choice(len(X), min(n_sigma, len(X)), replace=False)
    sub = X[idx]
    dists = pairwise_distances(sub, sub)
    sigma = np.median(dists[dists > 0])

    # Split by batch: the first two batches with cells
    batch = adata.obs[batch_key]
    batches = [value for value in batch.cat.categories if (batch == value).any()]
    if len(batches) > 2:
        print(f"MMD between the first two of {len(batches)} batches: {batches[:2]}.")
    b0, b1 = X[(batch == batches[0]).to_numpy()], X[(batch == batches[1]).to_numpy()]

    idx0 = rng.choice(len(b0), min(n_sub, len(b0)), replace=False)
    idx1 = rng.choice(len(b1), min(n_sub, len(b1)), replace=False)

    def rbf(A, B):
        d = pairwise_distances(A, B)
        return np.exp(-d**2 / (2 * sigma**2))

    b0s, b1s = b0[idx0], b1[idx1]
    return rbf(b0s, b0s).mean() + rbf(b1s, b1s).mean() - 2 * rbf(b0s, b1s).mean()


def batch_integration_metrics(adata: sc.AnnData, emb_key: str, batch_key: str, label_keys: list) -> pd.DataFrame:
    """
    iLISI, ASW and MMD between the batches of `batch_key` (mouse notebook, cell 42), with fixed seeds, then iLISI_max,
    iLISI_random, iLISI_relative and the label-conditioned ASW of each label key that labels cells of two or more
    batches.
    """
    from scib_metrics import ilisi_knn
    from scib_metrics.nearest_neighbors import pynndescent

    X = adata.obsm[emb_key]
    batch = adata.obs[batch_key].to_numpy()
    results = []

    # iLISI
    knn = pynndescent(X, n_neighbors=90, random_state=0)
    ilisi = float(np.mean(ilisi_knn(knn, batch)))
    print(f"iLISI: {ilisi:.4f}")
    results.append({"metric": "iLISI", "score": ilisi})

    # ASW
    asw = float(skm.silhouette_score(X, batch, metric="euclidean", sample_size=10000, random_state=0))
    print(f"ASW batch: {asw:.4f}")
    results.append({"metric": "ASW", "score": asw})

    # MMD
    mmd_score = float(mmd_comparable(adata, emb_key, batch_key))
    print(f"MMD: {mmd_score:.4f}")
    results.append({"metric": "MMD", "score": mmd_score})

    # The iLISI bound for these section sizes, the iLISI of random mixing on the same graph (sections permuted), and
    # the label-conditioned ASW (mixing within regions)
    ilisi_ceiling = ilisi_max(batch)
    ilisi_random = float(np.mean([np.mean(ilisi_knn(knn, np.random.default_rng(seed).permutation(batch)))
                                  for seed in range(ILISI_RANDOM_PERMUTATIONS)]))
    print(f"iLISI_max: {ilisi_ceiling:.4f}; iLISI_random: {ilisi_random:.4f}; iLISI_relative: "
          f"{ilisi / ilisi_random:.4f}")
    results.append({"metric": "iLISI_max", "score": ilisi_ceiling})
    results.append({"metric": "iLISI_random", "score": ilisi_random})
    results.append({"metric": "iLISI_relative", "score": ilisi / ilisi_random})
    for key in label_keys:
        if adata.obs.loc[adata.obs[key].notna(), batch_key].nunique() < 2:
            print(f"{key} labels cells of one section only: no label-conditioned ASW.")
            continue
        asw_label, n_labels = label_conditioned_asw(X, batch, adata.obs[key])
        print(f"ASW_label_{key}: {asw_label:.4f} over {n_labels} labels in two or more sections")
        results.append({"metric": f"ASW_label_{key}", "score": asw_label})
    return pd.DataFrame(results)


def check_eval_dataset(model_config: dict, eval_dataset_path: str):
    """
    The evaluation dataset must be the run's own tokenization of its gene set: its provenance (tokenize_dataset.py)
    records the training dataset's TERRA code version, tokenization settings, token dictionary and factor file.
    """
    train_dataset_path = model_config["data"]["tokenized_data_folder_path"]
    if Path(eval_dataset_path).resolve() == Path(train_dataset_path).resolve():
        return
    provenances = []
    for dataset_path in [train_dataset_path, eval_dataset_path]:
        provenance_file_path = Path(dataset_path) / PROVENANCE_FILE_NAME
        if not provenance_file_path.exists():
            print(f"WARNING: {dataset_path} has no {PROVENANCE_FILE_NAME}; cannot check that the evaluation data has "
                  f"the training tokenization.")
            return
        with open(provenance_file_path) as file:
            provenances.append(json.load(file))
    keys = ["code_version", "tokenization", "token_dictionary_md5", "norm_factor_md5"]
    differences = [key for key in keys if provenances[0].get(key) != provenances[1].get(key)]
    if differences:
        raise ValueError(f"The evaluation data {eval_dataset_path} was not tokenized like the training data "
                         f"{train_dataset_path} (differs in: {', '.join(differences)}); regenerate the configs or "
                         f"retokenize.")


def main(args):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    unassigned_labels = parse_unassigned_labels(args.unassigned_label, args.label_keys)
    output_folder_path = Path(args.output_folder_path)
    marker_file_path = output_folder_path / "evaluation_complete"

    # Up to date if evaluation_complete records the current inputs and every metric file is complete (--check_only:
    # only report it); a run not packaged yet (before training) needs its evaluation
    model_config_file_path = Path(args.model_folder_path) / "model_config.yaml"
    checkpoint_file_path = Path(args.model_folder_path) / "model_checkpoint.pt"
    if args.check_only and not marker_file_path.exists():
        print(f"Evaluation needed: no {marker_file_path}.")
        sys.exit(NEEDS_EVALUATION_EXIT_CODE)
    if not model_config_file_path.exists() or not checkpoint_file_path.exists():
        if args.check_only:
            print(f"Evaluation needed: {args.model_folder_path} is not packaged (no model_config.yaml or "
                  f"model_checkpoint.pt).")
            sys.exit(NEEDS_EVALUATION_EXIT_CODE)
        raise FileNotFoundError(f"{args.model_folder_path} is not packaged (package_run.py).")
    with open(model_config_file_path) as file:
        model_config = yaml.safe_load(file)
    marker = get_marker_inputs(args, unassigned_labels, model_config)
    reasons = evaluation_differences(args, output_folder_path, marker)
    if args.check_only:
        if reasons:
            print(f"Evaluation needed ({marker_file_path}):\n  " + "\n  ".join(reasons))
            sys.exit(NEEDS_EVALUATION_EXIT_CODE)
        print(f"Evaluation up to date ({marker_file_path}).")
        return
    if not reasons and not args.overwrite:
        print(f"Evaluation up to date ({marker_file_path}); --overwrite evaluates again.")
        return
    print("Evaluating all modes " + ("(--overwrite)." if not reasons else "because:\n  " + "\n  ".join(reasons)))
    # Removed first, so that a failed evaluation leaves no marker next to new metric files
    output_folder_path.mkdir(parents=True, exist_ok=True)
    marker_file_path.unlink(missing_ok=True)

    from datasets import load_from_disk

    # Evaluation dataset of this run (the training data or its shared-gene re-tokenization), and the labels and
    # sections of its cells (checked before embedding)
    eval_dataset_path = marker["eval_data_path"]
    print(f"Evaluation data: {eval_dataset_path}")
    check_eval_dataset(model_config, eval_dataset_path)
    dataset = load_from_disk(eval_dataset_path)
    cell_ids = [str(cell_id) for cell_id in dataset["cell_id"]]
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError(f"{len(cell_ids) - len(set(cell_ids))} cell ids occur more than once in {eval_dataset_path}.")
    labels = load_labels(args.label_file_paths, args.label_keys, unassigned_labels)
    missing_cell_ids = pd.Index(cell_ids).difference(labels.index)
    if len(missing_cell_ids) > 0:
        raise ValueError(f"{len(missing_cell_ids)} tokenized cells are not in the label files, e.g. "
                         f"{list(missing_cell_ids[:3])}.")
    obs = labels.loc[cell_ids]
    print(f"{len(cell_ids)} tokenized cells of {len(labels)} cells in the label files; labelled cells per key: "
          f"{obs[args.label_keys].notna().sum().to_dict()} (unassigned labels {unassigned_labels}); cells per "
          f"section: {obs[SECTION_KEY].value_counts(sort=False).to_dict()}.")
    for key in args.label_keys:
        if obs[key].notna().sum() == 0:
            raise ValueError(f"No tokenized cell has a label for {key}.")
    # Batch integration between the sections, over the labelled cells (the cells of the mouse notebook); fails here,
    # before embedding, without scib_metrics
    labelled = obs[args.label_keys].notna().any(axis=1)
    batch_integration = obs.loc[labelled, SECTION_KEY].nunique() >= 2
    if batch_integration:
        import scib_metrics  # noqa: F401
        print(f"Batch integration over the {int(labelled.sum())} labelled cells; per section: "
              f"{obs.loc[labelled, SECTION_KEY].value_counts(sort=False).to_dict()}.")
    elif obs[SECTION_KEY].nunique() >= 2:
        print("Only one section has labelled cells: no batch-integration metrics.")
    columns = [column for column in dataset.features if column != "cell_id"]
    dataset.set_format(type="torch", columns=columns, output_all_columns=True)

    # All embeddings first: embed_dataset forks dataloader workers, which must happen before the metrics start threads
    # (jax for iLISI, numba for the kNN graphs); a fork of a multithreaded process can deadlock
    modes = marker["modes"].split()
    embeddings = {}
    for mode in modes:
        print(f"Embedding with the metatoken {'removed' if mode == 'drop' else 'kept'} at inference "
              f"({marker['embedding_precision']})...")
        embeddings[mode] = embed(args, dataset, model_config, keep_special_tokens=(mode == "keep"))
    check_finite(embeddings, obs, args.label_keys)
    for mode in modes:
        file_prefix = get_file_prefix(args, mode)
        print(f"Evaluating the embeddings with the metatoken {'removed' if mode == 'drop' else 'kept'} at "
              f"inference...")
        evaluate(args, embeddings.pop(mode), obs, output_folder_path, file_prefix, batch_integration)
    # What produced the metrics: run_ablation_array.lsf skips a run only if they match the current inputs
    # (--check_only)
    marker.update({"unassigned_label": args.unassigned_label,
                   "batch_integration_metrics": " ".join(modes) if batch_integration else "none (one section)",
                   "eval_data_cells": str(len(cell_ids)),
                   **get_device_info(),
                   "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds")})
    marker_file_path.write_text("".join(f"{key}: {value}\n" for key, value in marker.items()))
    print(f"Wrote {marker_file_path}.")


def embed(args, dataset, model_config: dict, keep_special_tokens: bool) -> np.ndarray:
    """
    Neighborhood embeddings with the target encoder of the final checkpoint (TERRA raises if it does not load), in
    bf16 autocast on the GPU as in training (`meta.use_bfloat16`), or with --fp32_embeddings in fp32: embed_dataset
    reads the precision from the model folder's config, so it then embeds from a temporary copy of the folder (links
    to the checkpoint, token dictionary and factors) whose config has `use_bfloat16: false`.
    """
    from terra.inference.embed import embed_dataset

    model_folder_path = Path(args.model_folder_path)
    temporary_folder_path = None
    if args.fp32_embeddings and model_config["meta"]["use_bfloat16"]:
        temporary_folder_path = Path(tempfile.mkdtemp(prefix=".fp32_model_", dir=args.output_folder_path))
        for file_name in ["model_checkpoint.pt", "token_dictionary.pkl", "norm_factors.csv"]:
            (temporary_folder_path / file_name).symlink_to((model_folder_path / file_name).resolve())
        fp32_model_config = {**model_config, "meta": {**model_config["meta"], "use_bfloat16": False}}
        with open(temporary_folder_path / "model_config.yaml", "w") as file:
            yaml.safe_dump(fp32_model_config, file, sort_keys=False)
        model_folder_path = temporary_folder_path
    try:
        output_embed = embed_dataset(dataset=dataset,
                                     model_folder_path=str(model_folder_path),
                                     emb_layer=None,
                                     agg_excluded_genes=None,
                                     top_k=None,
                                     batch_size=128,
                                     pin_memory=False,
                                     num_workers=args.num_workers,
                                     include_spatial_cell_emb=False,
                                     keep_special_tokens=keep_special_tokens)
    finally:
        if temporary_folder_path is not None:
            shutil.rmtree(temporary_folder_path)
    return np.asarray(output_embed[args.emb_key], dtype=np.float32)


def evaluate(args, embeddings, obs, output_folder_path, file_prefix, batch_integration):
    """
    Compute the collapse statistics and the batch-integration metrics of the embeddings between the sections (two or
    more), cluster them on the resolution grid and score the clusters against the niche labels of each label key.
    """
    metrics_file_path = output_folder_path / f"{file_prefix}_niche_identification_metrics_res_sweep.csv"
    integration_file_path = output_folder_path / f"{file_prefix}_batch_integration_metrics.csv"
    cross_section_file_path = output_folder_path / f"{file_prefix}_cross_section_neighbors.csv"
    adata = sc.AnnData(obs=obs.copy(), obsm={args.emb_key: embeddings})

    # Collapse statistics (with the average cosine similarities of the mouse runs, which do not detect collapse)
    statistics = embedding_statistics(embeddings)
    print(f"Embedding statistics: {statistics}")
    if statistics["complete_collapse"] or statistics["dimensional_collapse"]:
        print(f"WARNING: the embeddings look collapsed (between-cell variance fraction "
              f"{statistics['between_cell_variance_fraction']:.2e}, threshold {COLLAPSE_MAX_VARIANCE_FRACTION}; "
              f"top PC share {statistics['top_pc_variance_fraction']:.3f}, threshold {COLLAPSE_MAX_TOP_PC_FRACTION}; "
              f"participation ratio {statistics['participation_ratio']:.2f}).")
    pd.DataFrame([{"emb_key": args.emb_key, **statistics}]).to_csv(
        output_folder_path / f"{file_prefix}_avg_cosine_similarity.csv", index=False)

    # Batch integration between the sections over the labelled cells (the mouse notebook removed unlabelled cells before
    # embedding)
    if batch_integration:
        labelled = adata.obs[args.label_keys].notna().any(axis=1).to_numpy()
        integration_df = batch_integration_metrics(adata[labelled].copy(), args.emb_key, SECTION_KEY,
                                                   args.label_keys)
        integration_df.to_csv(integration_file_path, index=False)
        print(f"Saved batch integration metrics to {integration_file_path}.")
    else:
        integration_file_path.unlink(missing_ok=True)
        cross_section_file_path.unlink(missing_ok=True)

    # Leiden resolution sweep on all embedded cells (scanpy igraph flavor, as in the mouse sweep) and NMI / ARI / AMI
    # against the labels of each key, over the cells it labels, with the granularity of the partition
    sc.pp.neighbors(adata, n_neighbors=15, use_rep=args.emb_key, key_added=args.emb_key)
    if batch_integration:
        cross_section_neighbors(adata, args.emb_key, args.label_keys).to_csv(cross_section_file_path, index=False)
    results = []
    for resolution in RESOLUTIONS:
        cluster_key = f"{args.emb_key}_clusters_res_{resolution}"
        sc.tl.leiden(adata,
                     neighbors_key=args.emb_key,
                     key_added=cluster_key,
                     resolution=resolution,
                     flavor="igraph",
                     n_iterations=2)
        for label_key in args.label_keys:
            mask = adata.obs[label_key].notna().to_numpy()
            labels_true = adata.obs.loc[mask, label_key].astype(str)
            labels_pred = adata.obs.loc[mask, cluster_key]
            results.append({"resolution": resolution,
                            "n_clusters": adata.obs[cluster_key].nunique(),
                            "region_key": label_key,
                            "NMI": skm.normalized_mutual_info_score(labels_true, labels_pred),
                            "ARI": skm.adjusted_rand_score(labels_true, labels_pred),
                            "AMI": skm.adjusted_mutual_info_score(labels_true, labels_pred),
                            "n_labelled_cells": int(mask.sum()),
                            "n_label_categories": labels_true.nunique(),
                            "n_clusters_labelled": labels_pred.nunique()})
            print(results[-1])

    adata.obs.to_csv(output_folder_path / f"{file_prefix}_clusters.csv")
    np.save(output_folder_path / f"{file_prefix}.npy", embeddings)
    pd.DataFrame(results).to_csv(metrics_file_path, index=False)
    print(f"Saved metrics to {metrics_file_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--model_folder_path',
                        type=str,
                        required=True,
                        help='Packaged run folder (model_config.yaml, model_checkpoint.pt, token_dictionary.pkl).')
    parser.add_argument('--output_folder_path',
                        type=str,
                        required=True,
                        help='Folder to store the evaluation outputs of this run.')
    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        help='Dataset name used in output file names.')
    parser.add_argument('--label_file_paths', '--silver_file_paths',
                        dest='label_file_paths',
                        type=str,
                        nargs='+',
                        required=True,
                        help='AnnData files with the tokenizer cell ids (obs["cell_id"]) and the niche labels, one per '
                             'section (LABEL_FILE_PATHS, by default the silver files; --silver_file_paths is the '
                             'former name).')
    parser.add_argument('--label_keys',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Keys of the niche labels in obs; metrics are computed for each.')
    parser.add_argument('--unassigned_label',
                        type=str,
                        required=True,
                        help='Label of cells outside annotated regions, excluded from the metrics: one value for all '
                             'label keys, or <key>=<value> pairs separated by ";", one per key.')
    parser.add_argument('--eval_gene_set',
                        type=str,
                        choices=EVAL_GENE_SETS,
                        required=True,
                        help='Gene set of the evaluation data (EVAL_GENE_SET of dataset.env): the run\'s tokenized '
                             'training dataset (all_genes) or its shared-gene re-tokenization (shared_genes).')
    parser.add_argument('--emb_key',
                        type=str,
                        default="neighborhood_emb",
                        help='Embedding to evaluate.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help='Number of dataloader workers used for embedding.')
    parser.add_argument('--metatoken_modes',
                        nargs='+',
                        choices=['drop', 'keep'],
                        default=['drop', 'keep'],
                        help='Remove (drop, as in the mouse evaluation) and/or keep the metatoken at inference.')
    parser.add_argument('--fp32_embeddings',
                        action='store_true',
                        help='Embed in fp32 instead of bf16 autocast (meta.use_bfloat16, as in training).')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--overwrite',
                      action='store_true',
                      help='Evaluate all modes again even if evaluation_complete matches the current inputs.')
    mode.add_argument('--check_only',
                      action='store_true',
                      help=f'Evaluate nothing; exit code 0 if evaluation_complete matches the current inputs and the '
                           f'metric files are complete, {NEEDS_EVALUATION_EXIT_CODE} (with the reasons) if the run '
                           f'needs its evaluation, any other code on errors.')
    args = parser.parse_args()

    main(args)
