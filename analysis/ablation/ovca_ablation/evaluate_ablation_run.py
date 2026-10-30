"""
This script evaluates one trained OVCA ablation run with the niche identification protocol of the mouse ablations
(`analysis/mouse_ablation/stjepa_v2_ablation.ipynb`, stjepa-reproducibility): neighborhood embeddings from the target
encoder (`embed_dataset`, metatoken padded as in the mouse evaluation), a kNN graph (k = 15), Leiden clustering on the
fixed resolution grid 0.6-1.4 (step 0.1; identical for all variants, i.e. chosen without labels) and NMI and ARI
against the pathologist niche labels. Cells labelled 'Unassigned' are embedded and clustered but excluded from the
metrics. Unlike the mouse evaluation, which removed unlabelled cells before embedding, they stay in the kNN graph
(as in training), so cluster counts and metric levels are comparable between OVCA variants, not with the mouse runs.

Unlike the mouse evaluation, which re-tokenized separate evaluation data with `harmonize_tokenize_embed_pipeline`,
the run's own tokenized training dataset is embedded: training and evaluation data are the same OVCA section, so this
gives identical tokens without re-tokenizing (which needs ~0.5 TB of memory for the dense-path normalization arms and
breaks ties at the top-64 cut non-deterministically). Labels are joined through the dataset's `cell_id` column, and the
run aborts if the checkpoint does not load (`load_checkpoint` otherwise silently returns a randomly initialized model).

Outputs in `--output_folder_path` (mouse layout `artifacts/ablation/<variant>/<run>/`):
- `<dataset_name>_neighborhood_emb_niche_identification_metrics_res_sweep.csv` (columns as in the mouse runs)
- `<dataset_name>_neighborhood_emb_avg_cosine_similarity.csv`
- `<dataset_name>_neighborhood_emb_clusters.csv` (label and Leiden cluster per resolution for every cell)
- `<dataset_name>_neighborhood_emb.npy` (float16 embeddings, rows in the order of the clusters CSV)

Usage:
```bash
python evaluate_ablation_run.py --model_folder_path [RUN_FOLDER] --output_folder_path [EVAL_FOLDER]
```
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.metrics as skm
import yaml

from ablation_utils import DATASET_NAME, SILVER_FILE_PATH

# Same resolution grid as the mouse sweep (base 1.0, 4 steps of 0.1 below and above)
RESOLUTIONS = [round(step * 0.1, 1) for step in range(6, 15)]


def require_loaded_checkpoint():
    """Make `embed_dataset` fail if its checkpoint does not load (epoch 0 is only returned on load errors)."""
    import app.inference.embed as embed_module

    load_checkpoint = embed_module.load_checkpoint

    def checked_load_checkpoint(*args, **kwargs):
        outputs = load_checkpoint(*args, **kwargs)
        if outputs[-2] == 0:
            raise RuntimeError("Checkpoint could not be loaded (see 'Encountered exception' above).")
        return outputs

    embed_module.load_checkpoint = checked_load_checkpoint


def main(args):
    from datasets import load_from_disk
    from app.inference import embed_dataset

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    output_folder_path = Path(args.output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    file_prefix = f"{args.dataset_name}_{args.emb_key}"
    metrics_file_path = output_folder_path / f"{file_prefix}_niche_identification_metrics_res_sweep.csv"
    if metrics_file_path.exists() and not args.overwrite:
        print(f"{metrics_file_path} already exists. Skipping evaluation...")
        return

    # Tokenized training dataset of this run
    with open(Path(args.model_folder_path) / "model_config.yaml") as file:
        model_config = yaml.safe_load(file)
    dataset = load_from_disk(model_config["data"]["tokenized_data_folder_path"])
    cell_ids = dataset["cell_id"]
    columns = [column for column in dataset.features if column != "cell_id"]
    dataset.set_format(type="torch", columns=columns, output_all_columns=True)

    # Neighborhood embeddings with the target encoder of the final checkpoint
    require_loaded_checkpoint()
    output_embed = embed_dataset(dataset=dataset,
                                 model_folder_path=args.model_folder_path,
                                 emb_layer=None,
                                 agg_excluded_genes=None,
                                 top_k=None,
                                 batch_size=128,
                                 pin_memory=False,
                                 num_workers=args.num_workers)
    embeddings = output_embed[args.emb_key]
    if len(embeddings) != len(cell_ids):
        raise ValueError(f"{len(embeddings)} embeddings for {len(cell_ids)} cells.")

    # Join the pathologist labels through the tokenizer cell ids
    obs = sc.read_h5ad(args.silver_file_path, backed="r").obs
    labels = obs.set_index("cell_id")[args.label_key].astype(str).replace({args.unassigned_label: np.nan})
    missing_cell_ids = pd.Index(cell_ids).difference(labels.index)
    if len(missing_cell_ids) > 0:
        raise ValueError(f"{len(missing_cell_ids)} tokenized cells are not in the silver data, e.g. "
                         f"{list(missing_cell_ids[:3])}.")
    adata = sc.AnnData(obs=pd.DataFrame({args.label_key: labels.loc[cell_ids].values}, index=cell_ids),
                       obsm={args.emb_key: embeddings})
    print(f"Embedded {adata.n_obs} of {len(obs)} cells; {adata.obs[args.label_key].notna().sum()} have a niche label.")

    # Average cosine similarity of embeddings (collapse check, as in the mouse runs)
    X = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    S = X.sum(axis=0)
    avg_cos_sim = (S @ S - X.shape[0]) / (X.shape[0] * (X.shape[0] - 1))
    pd.DataFrame([{"emb_key": args.emb_key, "avg_cosine_similarity": avg_cos_sim}]).to_csv(
        output_folder_path / f"{file_prefix}_avg_cosine_similarity.csv", index=False)

    # Leiden resolution sweep (scanpy igraph flavor, as in the mouse sweep) and NMI / ARI against the labels
    sc.pp.neighbors(adata, n_neighbors=15, use_rep=args.emb_key, key_added=args.emb_key)
    mask = adata.obs[args.label_key].notna().values
    labels_true = adata.obs.loc[mask, args.label_key]
    results = []
    for resolution in RESOLUTIONS:
        cluster_key = f"{args.emb_key}_clusters_res_{resolution}"
        sc.tl.leiden(adata,
                     neighbors_key=args.emb_key,
                     key_added=cluster_key,
                     resolution=resolution,
                     flavor="igraph",
                     n_iterations=2)
        labels_pred = adata.obs.loc[mask, cluster_key]
        results.append({"resolution": resolution,
                        "n_clusters": adata.obs[cluster_key].nunique(),
                        "region_key": args.label_key,
                        "NMI": skm.normalized_mutual_info_score(labels_true, labels_pred),
                        "ARI": skm.adjusted_rand_score(labels_true, labels_pred)})
        print(results[-1])

    adata.obs.to_csv(output_folder_path / f"{file_prefix}_clusters.csv")
    np.save(output_folder_path / f"{file_prefix}.npy", embeddings.astype(np.float16))
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
    parser.add_argument('--silver_file_path',
                        type=str,
                        default=SILVER_FILE_PATH,
                        help='OVCA silver AnnData with the tokenizer cell ids and the pathologist labels.')
    parser.add_argument('--dataset_name',
                        type=str,
                        default=DATASET_NAME,
                        help='Dataset name used in output file names.')
    parser.add_argument('--label_key',
                        type=str,
                        default="path_region",
                        help='Key of the pathologist niche labels in `obs`.')
    parser.add_argument('--unassigned_label',
                        type=str,
                        default="Unassigned",
                        help='Label of cells outside annotated regions; excluded from the metrics.')
    parser.add_argument('--emb_key',
                        type=str,
                        default="neighborhood_emb",
                        help='Embedding to evaluate.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help='Number of dataloader workers used for embedding.')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Recompute the metrics even if they exist.')
    args = parser.parse_args()

    main(args)
