"""
This script collects the niche identification metrics of all evaluated OVCA ablation runs and plots one panel per
ablation category in the style of the mouse comparison notebooks (`analysis/mouse_ablation/*_comparison.ipynb`):
bars show the mean over all Leiden resolutions (0.6-1.4) and replicates, error bars the standard deviation, and dots
the individual values colored by replicate. The base run is shown as the reference bar of each category, labelled
with its setting in that category (the mouse architecture panel had no base bar). Runs without results are skipped.

The category panels use the mouse evaluation (metatoken removed at inference). An extra panel,
`metatoken_inference`, compares removing and keeping the metatoken at inference for the base and metatoken dropout
runs, next to the run without metatoken.

Usage:
```bash
python compare_ablations.py
```
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ablation_utils import ARTIFACT_FOLDER_PATH, CATEGORY_ORDER, DATASET_NAME

BASE_LABELS = {"tokenizer": "Cell Graph (Combined)",
               "sequence_length": "64",
               "expression_encoding": "MLP",
               "spatial_encoding": "Rank",
               "masking_ratio": "0.6",
               "metatoken": "Metatoken",
               "normalization_unnormalized_ranks": "No Norm",
               "normalization_full": "No Norm",
               "normalization_unnormalized_counts": "No Norm"}
DOT_COLORS = ["#F0E442", "#E69F00", "#D55E00"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def collect_metrics(manifest_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    records = []
    for _, run in manifest_df.iterrows():
        for mode, prefix in [("drop", f"{dataset_name}_"), ("keep", f"{dataset_name}_keep_metatoken_")]:
            metrics_file_path = (Path(run["eval_folder_path"]) /
                                 f"{prefix}neighborhood_emb_niche_identification_metrics_res_sweep.csv")
            if not metrics_file_path.exists():
                continue
            metrics_df = pd.read_csv(metrics_file_path)
            for key in ["run_name", "variant", "category", "label", "replicate"]:
                metrics_df[key] = run[key]
            metrics_df["metatoken_mode"] = mode
            records.append(metrics_df)
    if not records:
        raise FileNotFoundError("No evaluated runs found.")
    return pd.concat(records, ignore_index=True)


def plot_category(metrics_df: pd.DataFrame, metric: str, title: str):
    """Horizontal bars (mean over resolutions and replicates) with SD and per-run dots, sorted by mean."""
    summary_df = (metrics_df.groupby("label")[metric].agg(["mean", "std"]).fillna(0)
                  .sort_values("mean", ascending=False))
    fig, ax = plt.subplots(figsize=(2.6, 0.24 * len(summary_df) + 0.7), constrained_layout=True)
    y = np.arange(len(summary_df))
    ax.barh(y, summary_df["mean"], height=0.55, color="#56B4E9", edgecolor="white", alpha=0.8, zorder=2)
    ax.errorbar(summary_df["mean"], y, xerr=summary_df["std"], fmt="none", ecolor="0.25", elinewidth=0.8,
                capsize=2.0, zorder=3)
    rng = np.random.default_rng(42)
    for idx, label in enumerate(summary_df.index):
        label_df = metrics_df[metrics_df["label"] == label]
        ax.scatter(label_df[metric], idx + rng.uniform(-0.15, 0.15, len(label_df)), s=8,
                   color=[DOT_COLORS[(replicate - 1) % len(DOT_COLORS)] for replicate in label_df["replicate"]],
                   edgecolor="0.25", linewidth=0.4, zorder=4)
        ax.text(max(label_df[metric].max(), summary_df.loc[label, "mean"]) * 1.02, idx,
                f"{summary_df.loc[label, 'mean']:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(summary_df.index)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric} (↑)")
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    return fig


def main(args):
    artifact_folder_path = Path(args.artifact_folder_path)
    results_folder_path = artifact_folder_path / "results"
    figures_folder_path = artifact_folder_path / "figures"
    results_folder_path.mkdir(parents=True, exist_ok=True)
    figures_folder_path.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(artifact_folder_path / "manifest.tsv", sep="\t")
    all_metrics_df = collect_metrics(manifest_df, args.dataset_name)
    all_metrics_df.to_csv(results_folder_path / "niche_identification_ovca_ablations.csv", index=False)
    summary_df = (all_metrics_df.groupby(["category", "variant", "label", "metatoken_mode"])[["NMI", "ARI", "n_clusters"]]
                  .agg(["mean", "std"]).join(all_metrics_df.groupby(["variant", "metatoken_mode"])["replicate"]
                                             .nunique().rename(("n_replicates", ""))))
    summary_df.to_csv(results_folder_path / "niche_identification_ovca_ablations_summary.csv")
    print(summary_df.round(3).to_string())

    # Metatoken at inference: removed (as in the mouse evaluation) vs kept, for runs trained with it
    metatoken_df = all_metrics_df[all_metrics_df["category"].isin(["base", "metatoken"])].copy()
    if not metatoken_df.empty:
        metatoken_df["label"] = metatoken_df.apply(
            lambda run: (run["label"] if run["variant"] == "no_metatk" else
                         f"{'Metatoken' if run['category'] == 'base' else run['label']} "
                         f"({'removed' if run['metatoken_mode'] == 'drop' else 'kept'})"), axis=1)
        for metric in ["NMI", "ARI"]:
            fig = plot_category(metatoken_df, metric, "Metatoken at inference (OVCA)")
            for extension in ["svg", "pdf", "png"]:
                fig.savefig(figures_folder_path / f"ovca_ablation_metatoken_inference_niche_{metric}.{extension}")
            plt.close(fig)

    # Category panels with the mouse evaluation (metatoken removed at inference)
    metrics_df = all_metrics_df[all_metrics_df["metatoken_mode"] == "drop"]
    base_df = metrics_df[metrics_df["category"] == "base"]
    for category in CATEGORY_ORDER[1:]:
        category_df = metrics_df[metrics_df["category"] == category]
        if category_df.empty:
            continue
        if category in BASE_LABELS and not base_df.empty:
            category_df = pd.concat([base_df.assign(label=BASE_LABELS[category]), category_df])
        for metric in ["NMI", "ARI"]:
            fig = plot_category(category_df, metric, f"{category.replace('_', ' ').capitalize()} (OVCA)")
            for extension in ["svg", "pdf", "png"]:
                fig.savefig(figures_folder_path / f"ovca_ablation_{category}_niche_{metric}.{extension}")
            plt.close(fig)
    print(f"Saved results to {results_folder_path} and figures to {figures_folder_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--artifact_folder_path',
                        type=str,
                        default=ARTIFACT_FOLDER_PATH,
                        help='Folder with manifest.tsv and the evaluation outputs.')
    parser.add_argument('--dataset_name',
                        type=str,
                        default=DATASET_NAME,
                        help='Dataset name in the metric file names.')
    args = parser.parse_args()

    main(args)
