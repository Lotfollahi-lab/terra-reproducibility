"""End-to-end benchmark analysis for a trained TERRA model.

Script version of xhs1000-39b_1p_analysis.ipynb. Loads a trained TERRA model
(identified by --model_timestamp and --checkpoint), runs inference on the
benchmark test datasets, identifies niches / cell types, and writes plots and
metrics into the model's artifact folder.

Plots for each benchmark dataset are saved under a per-dataset subfolder:
    {artifacts_folder_path}/{dataset}/{model_timestamp}/figures/{dataset_name}/

Run in the terra-rapids environment (needs rapids_singlecell + a GPU):
    source /nfs/team361/sb75/.venvs/terra-rapids/bin/activate
    python xhs1000-39b_1p_analysis.py \
        --model_timestamp 24062026_143538_479 --checkpoint terra-ep49.pth.tar

Submit on the cluster via xhs1000-39b_1p_analysis.lsf.
"""
import argparse
import logging
import os
import random
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: plots are saved to disk, never shown

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model_timestamp", required=True,
                    help="Timestamp of the trained model run (folder under "
                         "artifacts/<dataset>/), e.g. 24062026_143538_479.")
parser.add_argument("--checkpoint", required=True,
                    help="Checkpoint file to load, e.g. terra-ep49.pth.tar.")
parser.add_argument("--dataset", default="xhs1000-39b_1p",
                    help="Dataset the model was trained on (selects the model "
                         "artifact folder).")
parser.add_argument("--artifacts_folder_path",
                    default="/nfs/team361/sb75/terra-reproducibility/artifacts")
parser.add_argument("--datasets_folder_path", default="/nfs/team361/sb75/DATASETS")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--rerun_inference", action="store_true",
                    help="Recompute inference even if a cached adata exists.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Libraries
# ---------------------------------------------------------------------------
sys.path.append("../../utils")

import math  # noqa: E402,F401
import pickle  # noqa: E402,F401
import re  # noqa: E402,F401
from collections import defaultdict  # noqa: E402,F401
from datetime import datetime  # noqa: E402,F401
from itertools import chain, compress  # noqa: E402,F401

import anndata as ad  # noqa: E402,F401
import celltypist  # noqa: E402,F401
import dask  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rapids_singlecell as rsc  # noqa: E402
import scanpy as sc  # noqa: E402
import scib_metrics  # noqa: E402,F401
import seaborn as sns  # noqa: E402,F401
import torch  # noqa: E402,F401

dask.config.set({"dataframe.query-planning": True})
import dask.dataframe.dask_expr  # noqa: E402,F401
import squidpy as sq  # noqa: E402,F401

from datasets import concatenate_datasets, load_from_disk  # noqa: E402,F401
from scib_metrics import ilisi_knn  # noqa: E402,F401
from scib_metrics.nearest_neighbors import pynndescent  # noqa: E402,F401
from scipy.sparse import csr_matrix  # noqa: E402,F401
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402,F401
from sklearn.metrics.cluster import (adjusted_rand_score,  # noqa: E402
                                      normalized_mutual_info_score)
from sklearn.model_selection import train_test_split  # noqa: E402,F401
from sklearn.preprocessing import StandardScaler  # noqa: E402,F401

# TERRA public API (previously imported from the now-removed `app` package).
from terra import (embed_dataset, get_average_gene_embed,  # noqa: E402,F401
                   get_gene_embed, infer)
from terra.training import train  # noqa: E402,F401
from terra.datasets import prepare_dataset  # noqa: E402
from terra.utils.config import create_params_from_YAML_wandb_config  # noqa: E402
from terra.utils.distributed import init_distributed  # noqa: E402,F401
from terra.utils.embedding import collect_adata_from_folder  # noqa: E402,F401
from terra.utils.evaluation import (classification_metrics,  # noqa: E402,F401
                                     clustering_metrics)

from analysis_utils import *  # noqa: E402,F403  (custom_palette, compute_similarity_stats, ...)

# ---------------------------------------------------------------------------
# Config / params
# ---------------------------------------------------------------------------
dataset = args.dataset
load_timestamp = args.model_timestamp
read_checkpoint = args.checkpoint
tokenizer = "cell_graph"
emb_layer = None
top_k = None
exclude_unshared_tokens = False
agg_type = "avg"
batch_size = args.batch_size

artifacts_folder_path = args.artifacts_folder_path
datasets_folder_path = args.datasets_folder_path
raw_data_folder_path = f"{datasets_folder_path}/silver/{dataset}"
model_folder_path = f"{artifacts_folder_path}/{dataset}/{load_timestamp}"

rank = 0
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)

params = create_params_from_YAML_wandb_config(f"{model_folder_path}/params.yaml", logger)
if read_checkpoint:
    params["state"]["read_checkpoint"] = read_checkpoint
    params["state"]["load_checkpoint"] = True
layer = params["meta"]["enc_depth"]
params["data"]["batch_size"] = batch_size

old_tokenized_data_folder_path = Path(params["data"]["tokenized_data_folder_path"])
filename = old_tokenized_data_folder_path.name
_, after_none = filename.split("None", 1)
tokenized_data_folder_path = (old_tokenized_data_folder_path.parent.parent
                              / "hst_corpus_110m"
                              / f"hst_corpus_110m_32_None{after_none}")

# Notebook setup
warnings.filterwarnings("ignore", category=ResourceWarning)
plt.rcParams["font.size"] = 5
plt.rcParams["text.usetex"] = False
plt.rcParams["svg.fonttype"] = "none"
os.environ["NUMBA_CPU_NAME"] = "generic"
sc.settings.set_figure_params(dpi=300, dpi_save=300, figsize=(6, 4),
                              facecolor="white", fontsize=10)

# Set per-dataset below; declared here so the functions can read it.
dataset_name = None
figures_folder_path = None
spot_sizes = None


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
def make_large_cmap(n):
    base = mpl.colormaps["turbo"].resampled(n)
    return [mcolors.to_hex(base(i)) for i in range(n)]


def _ckpt_tag():
    return read_checkpoint.split("-")[1].split(".")[0]


def configure_dataset_params(cfg):
    """Apply the per-dataset overrides to the shared `params` dict."""
    params["data"]["dataset_name"] = cfg["dataset_name"]
    base = params["data"]["raw_data_folder_path"].rsplit("/", 1)[0]
    params["data"]["raw_data_folder_path"] = base + "/" + cfg["raw_subdir"]
    params["data"]["tokenized_data_folder_path"] = (
        str(tokenized_data_folder_path).rsplit("_", 1)[0] + ".dataset")
    params["data"]["precomputed_split"] = (
        "/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer/"
        + params["data"]["tokenized_data_folder_path"].split("/")[-1].replace(
            ".dataset", f"_precomputed_split_{cfg['dataset_name']}.pkl"))
    params["data"]["precomputed_n_nonzero_tokens"] = None
    params["data"]["test_batch_ids"] = cfg["test_batch_ids"]
    params["data"]["val_batch_ids"] = None
    params["data"].pop("precomputed_epoch_n_nonzero_tokens", None)
    params["data"].pop("precomputed_epoch_splits", None)


def run_inference(dataset_name, params, read_checkpoint, obs_cols, emb_layers,
                  top_k, rerun_inference=False, write_to_disk=True):
    adata_path = f"{model_folder_path}/adata_{_ckpt_tag()}_{dataset_name}.h5ad"
    if os.path.exists(adata_path) and not rerun_inference:
        adata = sc.read_h5ad(adata_path)
    else:
        if params["data"]["precomputed_split"]:
            test_dataset, _, _ = prepare_dataset(params)
        else:
            _, _, test_dataset = prepare_dataset(params)
        adata = infer(
            params, test_dataset,
            load_folder_path=model_folder_path,
            dataset_ids=list(set(bid.split("_")[0]
                                 for bid in params["data"]["test_batch_ids"])),
            obs_cols=obs_cols, emb_layers=emb_layers, agg_type="avg",
            top_k=top_k, masked_tokens=None, agg_excluded_genes=None,
            include_spatial_cell_emb=True, cell_gene_ids=[],
            neighborhood_gene_ids=[], ignore_spc_tokens=True)

    adata.obs["batch"] = [cid.split("_")[1].replace("batch", "")
                          for cid in adata.obs.index.tolist()]
    if write_to_disk:
        adata.write(adata_path)
    return adata


def compute_umap(adata, emb_keys, emb_layer, write_to_disk=True):
    for emb_key in emb_keys:
        if f"{emb_key}_connectivities" not in adata.obsp.keys():
            rsc.pp.neighbors(adata, n_neighbors=15,
                             use_rep=f"{emb_key}_emb_layer_{emb_layer}",
                             key_added=emb_key)
        if f"{emb_key}_umap" not in adata.obsm.keys():
            rsc.tl.umap(adata, neighbors_key=emb_key, key_added=f"{emb_key}_umap")
    if write_to_disk:
        adata.write(f"{model_folder_path}/adata_{_ckpt_tag()}_{dataset_name}.h5ad")
    return adata


def compute_latent_clusters(adata, emb_keys=["cell", "neighborhood"],
                            obs_col="niche_type", n_unique_vals=None,
                            latent_leiden_resolution_cell=1.0,
                            latent_leiden_resolution_neighborhood=0.4,
                            latent_leiden_resolution_agg=0.4, write_to_disk=True):
    latent_cluster_key = None
    for emb_key in emb_keys:
        latent_leiden_resolutions = []
        if emb_key == "cell" or "ctx_1" in emb_key:
            latent_leiden_resolution = latent_leiden_resolution_cell
        elif emb_key == "neighborhood" or "ctx" in emb_key:
            latent_leiden_resolution = latent_leiden_resolution_neighborhood
        elif emb_key == "agg":
            latent_leiden_resolution = latent_leiden_resolution_agg
        adata.obsm["X_umap"] = adata.obsm[f"{emb_key}_umap"]
        if not n_unique_vals:
            n_unique_vals = len(adata.obs[obs_col].value_counts() < 100)
        print(f"Target number of unique values: {n_unique_vals}.")
        n_latent_clusters = 0
        while n_latent_clusters != n_unique_vals:
            if latent_leiden_resolution in latent_leiden_resolutions:
                latent_leiden_resolution += random.uniform(0.0001, 0.001)
            latent_leiden_resolutions.append(latent_leiden_resolution)
            print(f"Leiden resolution: {latent_leiden_resolution}.")
            latent_cluster_key = (f"{emb_key}_emb_leiden_res"
                                  + str(latent_leiden_resolution).replace(".", "_"))
            rsc.tl.leiden(adata, neighbors_key=emb_key,
                          key_added=latent_cluster_key,
                          resolution=latent_leiden_resolution)
            n_latent_clusters = len(adata.obs[latent_cluster_key].value_counts() < 100)
            print(f"Number of latent clusters {n_latent_clusters}.")
            if n_latent_clusters == n_unique_vals:
                break
            elif n_latent_clusters < (n_unique_vals - 8):
                latent_leiden_resolution += 0.1
            elif n_latent_clusters < (n_unique_vals - 5):
                latent_leiden_resolution += 0.05
            elif n_latent_clusters < (n_unique_vals - 3):
                latent_leiden_resolution += 0.02
            elif n_latent_clusters < n_unique_vals:
                latent_leiden_resolution += 0.001
            elif n_latent_clusters > (n_unique_vals + 8):
                latent_leiden_resolution -= 0.1
            elif n_latent_clusters > (n_unique_vals + 5):
                latent_leiden_resolution -= 0.05
            elif n_latent_clusters > (n_unique_vals + 3):
                latent_leiden_resolution -= 0.02
            else:
                latent_leiden_resolution -= 0.001

        labels = adata.obs[obs_col]
        labels_pred = adata.obs[latent_cluster_key]
        nmi = normalized_mutual_info_score(labels, labels_pred,
                                           average_method="arithmetic")
        ari = adjusted_rand_score(labels, labels_pred)
        print(f"NMI is {nmi}. ARI is {ari}.")
        pd.DataFrame({"metric": ["NMI", "ARI"], "value": [nmi, ari]}).to_csv(
            f"{model_folder_path}/{_ckpt_tag()}_{dataset_name}_{latent_cluster_key}_metrics.csv",
            index=False)

        # Plot UMAPs (saved into the per-dataset figures folder).
        os.makedirs(figures_folder_path, exist_ok=True)
        for label in ["batch", obs_col, latent_cluster_key]:
            if len(np.unique(adata.obs[label])) > 1:
                palette = make_large_cmap(adata.obs[label].nunique())
                sc.pl.umap(adata, neighbors_key=emb_key,
                           title=(emb_key + "_emb_" + label
                                  if emb_key not in label else label),
                           color=label,
                           palette=(custom_palette if label == latent_cluster_key
                                    else palette if label == obs_col else None),
                           show=False)
                suffix = "_" + emb_key + "_emb" if emb_key not in label else ""
                plt.savefig(
                    f"{figures_folder_path}/{_ckpt_tag()}_{dataset_name}{suffix}_{label}_umap.svg",
                    dpi=300, bbox_inches="tight")
                plt.close()

        # Plot clusters in tissue, per batch.
        batch_ids = adata.obs["batch"].unique().tolist()
        for i, batch_id in enumerate(batch_ids):
            adata_batch = adata[adata.obs["batch"] == batch_id]
            fname = f"{_ckpt_tag()}_{dataset_name}_{latent_cluster_key}_batch_{batch_id}.svg"
            sc.pl.spatial(adata_batch, color=latent_cluster_key,
                          palette=custom_palette,
                          spot_size=spot_sizes[i % len(spot_sizes)], save=fname)
            shown = f"{figures_folder_path}/show{fname}"
            if os.path.exists(shown):
                os.rename(shown, f"{figures_folder_path}/{fname}")

        if write_to_disk:
            adata.write(f"{model_folder_path}/adata_{_ckpt_tag()}_{dataset_name}.h5ad")
    return adata, latent_cluster_key


def compute_embedding_similarity(adata, emb_layer, emb_keys=["cell", "neighborhood"],
                                 label="niche_type", write_to_disk=True):
    for emb_key in emb_keys:
        similarity_stats = compute_similarity_stats(
            adata, n_samples=None, type_key=label,
            embedding_key=f"{emb_key}_emb_layer_{emb_layer}")
        avg_cos_sim = np.mean(similarity_stats["Mean Similarity"])
        print(f"Average {label} cosine similarity is {avg_cos_sim}.")
        pd.DataFrame({"metric": [f"avg_{label}_cos_sim"],
                      "value": [avg_cos_sim]}).to_csv(
            f"{model_folder_path}/{_ckpt_tag()}_{dataset_name}_avg_{label}_cos_sim.csv",
            index=False)
        plot_similarity_heatmaps(
            similarity_stats,
            save_file_path=f"{figures_folder_path}/{_ckpt_tag()}_{dataset_name}_{emb_key}_emb_{label}_cos_sim.svg")


def report_embedding_pca(adata, emb_layer):
    """Print PCA / effective-rank diagnostics for the cell and neighborhood embeddings."""
    for emb_key in ["cell", "neighborhood"]:
        Z = adata.obsm[f"{emb_key}_emb_layer_{emb_layer}"]
        pca = PCA(n_components=min(Z.shape[0], Z.shape[1]))
        pca.fit(Z)
        print(f"[{emb_key}] cumulative variance first 10 dims:",
              pca.explained_variance_ratio_[:10].sum())
        std_per_dim = np.std(Z, axis=0)
        print(f"[{emb_key}] std min/max/mean:",
              std_per_dim.min(), std_per_dim.max(), std_per_dim.mean())
        eigvals = pca.explained_variance_
        eigvals = eigvals[eigvals > 0]
        p = eigvals / eigvals.sum()
        print(f"[{emb_key}] effective rank:", np.exp(-(p * np.log(p)).sum()))


# ---------------------------------------------------------------------------
# Per-dataset benchmark configuration
# ---------------------------------------------------------------------------
# analysis: "niche" -> cluster the neighborhood embedding against the niche
# label; "cell" -> cluster the cell embedding against the cell-type label.
DATASETS = [
    dict(dataset_name="xhs49-1b_1p", raw_subdir="xhs49-1b_1p",
         label_key="niche_name", emb_keys=["neighborhood"], n_unique_vals=8,
         res_cell=1.0, res_neighborhood=0.4, spot_size=20,
         test_batch_ids=["49_batch0"]),
    dict(dataset_name="xhs1000-39b_1p", raw_subdir="xhs1000-39b_1p",
         label_key="niche_type", emb_keys=["neighborhood"], n_unique_vals=12,
         res_cell=1.0, res_neighborhood=0.4, spot_size=20,
         test_batch_ids=["1000_batch11", "1000_batch19", "1000_batch32"]),
    dict(dataset_name="xhk1007-41b_1p", raw_subdir="xhk1007-CV1-41b_1p",
         label_key="level_1_cell_type", emb_keys=["cell"], n_unique_vals=5,
         res_cell=0.75, res_neighborhood=0.2, spot_size=30,
         test_batch_ids=["1007_batch1", "1007_batch14"]),
    dict(dataset_name="xhb1002-75b_3p", raw_subdir="xhb1002-AT9-6b_1p",
         label_key="leiden_log1pnorm_res-1", emb_keys=["cell"], n_unique_vals=12,
         res_cell=1.0, res_neighborhood=0.4, spot_size=30,
         test_batch_ids=["1002_batch3", "1002_batch15", "1002_batch27"]),
    dict(dataset_name="chl59-8b_1p", raw_subdir="chl59-8b_1p",
         label_key="niche", emb_keys=["neighborhood"], n_unique_vals=12,
         res_cell=1.0, res_neighborhood=0.6, spot_size=80,
         test_batch_ids=["59_batch0", "59_batch6", "59_batch7",
                         "59_batch4", "59_batch1", "59_batch5"]),
    dict(dataset_name="chr78-11b_1p", raw_subdir="chr78-11b_1p",
         label_key="broad_anatomy", emb_keys=["neighborhood"], n_unique_vals=21,
         res_cell=1.0, res_neighborhood=0.4, spot_size=1920 * 0.093,
         test_batch_ids=["78_batch3", "78_batch10", "78_batch1", "78_batch0",
                         "78_batch2", "78_batch6", "78_batch7", "78_batch9",
                         "78_batch8", "78_batch4", "78_batch5"]),
]


# ---------------------------------------------------------------------------
# Run the benchmark analysis for every dataset
# ---------------------------------------------------------------------------
emb_layer = params["meta"]["enc_depth"]

for cfg in DATASETS:
    dataset_name = cfg["dataset_name"]
    label_key = cfg["label_key"]
    emb_keys = cfg["emb_keys"]
    spot_sizes = [cfg["spot_size"]] * len(cfg["test_batch_ids"])
    figures_folder_path = f"{model_folder_path}/figures/{dataset_name}"
    os.makedirs(figures_folder_path, exist_ok=True)
    sc.settings.figdir = figures_folder_path

    logger.info(f"===== Analyzing {dataset_name} =====")
    try:
        configure_dataset_params(cfg)

        adata = run_inference(dataset_name, params, read_checkpoint,
                              obs_cols=[label_key], emb_layers=[emb_layer],
                              top_k=top_k, rerun_inference=args.rerun_inference,
                              write_to_disk=True)
        adata = adata[~adata.obs[label_key].isna()]

        adata = compute_umap(adata, emb_keys=emb_keys, emb_layer=emb_layer,
                             write_to_disk=True)
        adata, latent_cluster_key = compute_latent_clusters(
            adata, emb_keys=emb_keys, obs_col=label_key,
            n_unique_vals=cfg["n_unique_vals"],
            latent_leiden_resolution_cell=cfg["res_cell"],
            latent_leiden_resolution_neighborhood=cfg["res_neighborhood"],
            write_to_disk=True)
        compute_embedding_similarity(adata, emb_layer=emb_layer,
                                     emb_keys=emb_keys, label=latent_cluster_key,
                                     write_to_disk=True)
        report_embedding_pca(adata, emb_layer)
        logger.info(f"Finished {dataset_name}. Plots in {figures_folder_path}")
    except Exception:
        logger.error(f"Analysis FAILED for {dataset_name}:\n{traceback.format_exc()}")

logger.info("Benchmark analysis complete.")
