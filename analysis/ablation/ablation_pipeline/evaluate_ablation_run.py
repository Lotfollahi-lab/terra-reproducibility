"""
This script evaluates one trained ablation run with the niche identification protocol of the mouse ablations
(`analysis/mouse_ablation/stjepa_v2_ablation.ipynb`, stjepa-reproducibility): neighborhood embeddings from the target
encoder (`embed_dataset`), a kNN graph (k = 15), Leiden clustering on the fixed resolution grid 0.6-1.4 (step 0.1;
identical for all variants, i.e. chosen without labels) and NMI and ARI against the niche labels of each label key.
If the cells come from two or more sections, it also computes the batch-integration metrics of the mouse notebook.

Models with special tokens (the batch metatoken) are evaluated twice: with the metatoken removed at inference, as in
the mouse evaluation (`drop`, the default outputs), and with the metatoken kept as in training (`keep`,
`keep_special_tokens=True` of TERRA's `embed_dataset`; outputs prefixed `<dataset_name>_keep_metatoken_`). Both modes get all
metrics, including the batch-integration metrics.

Labels and sections: the obs of the label files (`--label_file_paths`; the job array passes LABEL_FILE_PATHS, by
default the silver files) are joined to the tokenized cells by `cell_id` (an error is raised if an id occurs twice or a
tokenized cell has no row). Each label file is one section; a cell's section is the file its row came from. A label key
may be missing from some files (e.g. a section-specific annotation): its metrics are computed over the cells where it
is set and not `--unassigned_label`. All embedded cells are clustered, including unlabelled ones; unlike the mouse
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
scib_metrics (with pynndescent and jax) is imported only if there are two or more sections; jax runs on the CPU.

The embedded dataset is `eval_tokenized_data_folder_path` of the run config (generate_ablation_runs.py; configs
without it: the training dataset): for a dataset with one section (OVCA), the run's own tokenized training dataset;
with two or more sections (mouse), the shared-gene re-tokenization of the same cells with the run's tokenization
(filter_shared_genes.py, tokenize_dataset.py --gene_set shared_genes), as the published mouse evaluation embedded
shared-gene data, so that the integration metrics do not measure the gene panels. It is tokenized by the pipeline
with the training tokenizer settings, not by `harmonize_tokenize_embed_pipeline` (which needs ~0.5 TB of memory for
the dense-path OVCA normalization arms without the ranked-list cut, and broke ties at the top-64 cut
non-deterministically). The run aborts if the checkpoint
does not load (`load_checkpoint` otherwise silently returns a randomly initialized model). `NUMBA_CPU_NAME=generic`
(unless set) compiles the numba kNN code for the same generic CPU on every node, as in the mouse notebooks, so that the
graph does not depend on the node.

Outputs in `--output_folder_path` (mouse layout `artifacts/ablation/<variant>/<run>/`), per metatoken mode with the
prefix `<dataset_name>_` (drop) or `<dataset_name>_keep_metatoken_` (keep):
- `<prefix>neighborhood_emb_niche_identification_metrics_res_sweep.csv` (columns as in the mouse runs; one row per
  resolution and label key, `region_key`)
- `<prefix>neighborhood_emb_batch_integration_metrics.csv` (two or more sections; columns `metric,score` with the rows
  iLISI, ASW and MMD, as in the mouse notebook)
- `<prefix>neighborhood_emb_avg_cosine_similarity.csv`: average pairwise cosine similarity of the embeddings (collapse
  check, as in the mouse runs; saturates near 0.99 for healthy embeddings) and of the mean-centered embeddings
- `<prefix>neighborhood_emb_clusters.csv` (section, labels and Leiden cluster per resolution for every cell)
- `<prefix>neighborhood_emb.npy` (float32 embeddings, rows in the order of the clusters CSV)
- `evaluation_complete` once all modes are done, with the modes, label keys, label files, unassigned label and the
  modes with batch-integration metrics.
A mode is evaluated again if its niche metrics lack a label key or, with two or more sections, its batch-integration
metrics are missing; all modes are, if `evaluation_complete` records other label files or another unassigned label.

Usage:
```bash
python evaluate_ablation_run.py --model_folder_path [RUN_FOLDER] --output_folder_path [EVAL_FOLDER] \
    --dataset_name [DATASET_NAME] --label_file_paths [LABEL_FILE ...] --label_keys [KEY ...] \
    --unassigned_label [UNASSIGNED_LABEL]
```
"""

import os

# Before scanpy / numba / jax are imported: numba compiles for this CPU name instead of the node's own CPU, and jax
# (iLISI) runs on the CPU, so that it neither depends on the node nor preallocates GPU memory next to torch
os.environ.setdefault("NUMBA_CPU_NAME", "generic")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import sklearn.metrics as skm  # noqa: E402
import yaml  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ablation_utils import SECTION_KEY, get_section_names  # noqa: E402

# Same resolution grid as the mouse sweep (base 1.0, 4 steps of 0.1 below and above)
RESOLUTIONS = [round(step * 0.1, 1) for step in range(6, 15)]
INTEGRATION_METRICS = ["iLISI", "ASW", "MMD"]
# Elements of the temporary difference array of the chunked pairwise distances (float32: 256 MB)
DISTANCE_CHUNK_ELEMENTS = 2 ** 26


def load_labels(label_file_paths: list, label_keys: list, unassigned_label: str) -> pd.DataFrame:
    """
    Labels and sections of all cells of the label files, indexed by `cell_id`: one column per label key, NaN where the
    key is missing (from the file), not set or the unassigned label, and the section (the label file) of each cell.
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
    labels = labels.where(labels.notna() & (labels != unassigned_label), np.nan)
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


def batch_integration_metrics(adata: sc.AnnData, emb_key: str, batch_key: str) -> pd.DataFrame:
    """iLISI, ASW and MMD between the batches of `batch_key` (mouse notebook, cell 42), with fixed seeds."""
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
    return pd.DataFrame(results)


def main(args):
    from datasets import load_from_disk

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    output_folder_path = Path(args.output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    # Evaluation dataset of this run (the training data or its shared-gene re-tokenization), and the labels and
    # sections of its cells (checked before embedding)
    with open(Path(args.model_folder_path) / "model_config.yaml") as file:
        model_config = yaml.safe_load(file)
    eval_dataset_path = (model_config.get("eval_tokenized_data_folder_path")
                         or model_config["data"]["tokenized_data_folder_path"])
    print(f"Evaluation data: {eval_dataset_path}")
    dataset = load_from_disk(eval_dataset_path)
    cell_ids = [str(cell_id) for cell_id in dataset["cell_id"]]
    labels = load_labels(args.label_file_paths, args.label_keys, args.unassigned_label)
    missing_cell_ids = pd.Index(cell_ids).difference(labels.index)
    if len(missing_cell_ids) > 0:
        raise ValueError(f"{len(missing_cell_ids)} tokenized cells are not in the label files, e.g. "
                         f"{list(missing_cell_ids[:3])}.")
    obs = labels.loc[cell_ids]
    print(f"{len(cell_ids)} tokenized cells of {len(labels)} cells in the label files; labelled cells per key: "
          f"{obs[args.label_keys].notna().sum().to_dict()}; cells per section: "
          f"{obs[SECTION_KEY].value_counts(sort=False).to_dict()}.")
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

    # Keeping the metatoken only differs from dropping it if the model has special tokens
    modes = [mode for mode in args.metatoken_modes if mode == "drop" or model_config["meta"]["special_tokens"]]
    # The metric files do not record the labels they were computed with: if evaluation_complete records other label
    # files or another unassigned label, all modes are evaluated again
    marker_file_path = output_folder_path / "evaluation_complete"
    marker = {"label_file_paths": " ".join(args.label_file_paths), "unassigned_label": args.unassigned_label}
    previous_marker = read_marker(marker_file_path)
    changed = [key for key, value in marker.items() if key in previous_marker and previous_marker[key] != value]
    overwrite = args.overwrite
    if changed and not overwrite:
        print(f"{marker_file_path} records other {' and '.join(changed)}; evaluating all modes again...")
        overwrite = True
    pending_modes = {}
    for mode in modes:
        file_prefix = (f"{args.dataset_name}_{args.emb_key}" if mode == "drop" else
                       f"{args.dataset_name}_keep_metatoken_{args.emb_key}")
        metrics_file_path = output_folder_path / f"{file_prefix}_niche_identification_metrics_res_sweep.csv"
        integration_file_path = output_folder_path / f"{file_prefix}_batch_integration_metrics.csv"
        if metrics_file_path.exists() and not overwrite:
            missing_keys = sorted(set(args.label_keys) - set(pd.read_csv(metrics_file_path)["region_key"]))
            if missing_keys:
                print(f"{metrics_file_path} lacks label keys {missing_keys}; recomputing...")
            elif batch_integration and not integration_complete(integration_file_path):
                print(f"{integration_file_path} is missing or incomplete; recomputing the '{mode}' evaluation...")
            else:
                print(f"{metrics_file_path} already exists. Skipping the '{mode}' evaluation...")
                continue
        pending_modes[mode] = (file_prefix, metrics_file_path, integration_file_path)
    # All embeddings first: embed_dataset forks dataloader workers, which must happen before the metrics start threads
    # (jax for iLISI, numba for the kNN graphs); a fork of a multithreaded process can deadlock
    embeddings = {}
    for mode in pending_modes:
        print(f"Embedding with the metatoken {'removed' if mode == 'drop' else 'kept'} at inference...")
        embeddings[mode] = embed(args, dataset, keep_special_tokens=(mode == "keep"))
    for mode, (file_prefix, metrics_file_path, integration_file_path) in pending_modes.items():
        print(f"Evaluating the embeddings with the metatoken {'removed' if mode == 'drop' else 'kept'} at "
              f"inference...")
        evaluate(args, embeddings.pop(mode), obs, output_folder_path, file_prefix, metrics_file_path,
                 integration_file_path, batch_integration)
    # Records the labels, so that a campaign with other labels evaluates finished runs again (run_ablation_array.lsf
    # skips a run only if they match)
    marker_file_path.write_text(
        f"modes: {' '.join(modes)}\n"
        f"label_keys: {' '.join(args.label_keys)}\n"
        f"label_file_paths: {marker['label_file_paths']}\n"
        f"unassigned_label: {marker['unassigned_label']}\n"
        f"batch_integration_metrics: {' '.join(modes) if batch_integration else 'none (one section)'}\n")


def embed(args, dataset, keep_special_tokens: bool) -> np.ndarray:
    """Neighborhood embeddings with the target encoder of the final checkpoint (TERRA raises if it does not load)."""
    from terra.inference.embed import embed_dataset

    output_embed = embed_dataset(dataset=dataset,
                                 model_folder_path=args.model_folder_path,
                                 emb_layer=None,
                                 agg_excluded_genes=None,
                                 top_k=None,
                                 batch_size=128,
                                 pin_memory=False,
                                 num_workers=args.num_workers,
                                 include_spatial_cell_emb=False,
                                 keep_special_tokens=keep_special_tokens)
    return np.asarray(output_embed[args.emb_key], dtype=np.float32)


def evaluate(args, embeddings, obs, output_folder_path, file_prefix, metrics_file_path, integration_file_path,
             batch_integration):
    """
    Compute the batch-integration metrics of the embeddings between the sections (two or more), cluster them on the
    resolution grid and score the clusters against the niche labels of each label key.
    """
    if len(embeddings) != len(obs):
        raise ValueError(f"{len(embeddings)} embeddings for {len(obs)} cells.")
    adata = sc.AnnData(obs=obs.copy(), obsm={args.emb_key: embeddings})

    # Average cosine similarity of the embeddings (collapse check, as in the mouse runs) and of the mean-centered
    # embeddings: the shared mean direction makes the first saturate near 0.99 even for informative embeddings
    X = embeddings.astype(np.float64)
    pd.DataFrame([{"emb_key": args.emb_key,
                   "avg_cosine_similarity": average_cosine_similarity(X),
                   "avg_cosine_similarity_centered": average_cosine_similarity(X - X.mean(axis=0))}]).to_csv(
        output_folder_path / f"{file_prefix}_avg_cosine_similarity.csv", index=False)
    del X

    # Batch integration between the sections over the labelled cells (the mouse notebook removed unlabelled cells before
    # embedding), written before the niche metrics (which mark the mode as complete)
    if batch_integration:
        labelled = adata.obs[args.label_keys].notna().any(axis=1).to_numpy()
        integration_df = batch_integration_metrics(adata[labelled].copy(), args.emb_key, SECTION_KEY)
        integration_df.to_csv(integration_file_path, index=False)
        print(f"Saved batch integration metrics to {integration_file_path}.")
    else:
        integration_file_path.unlink(missing_ok=True)

    # Leiden resolution sweep on all embedded cells (scanpy igraph flavor, as in the mouse sweep) and NMI / ARI
    # against the labels of each key, over the cells it labels
    sc.pp.neighbors(adata, n_neighbors=15, use_rep=args.emb_key, key_added=args.emb_key)
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
                            "ARI": skm.adjusted_rand_score(labels_true, labels_pred)})
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
                        help='Label of cells outside annotated regions; excluded from the metrics.')
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
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Recompute the metrics even if they exist.')
    args = parser.parse_args()

    main(args)
