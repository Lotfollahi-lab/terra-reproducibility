"""
This script collects the niche identification metrics of all evaluated runs of an ablation campaign and plots one
panel per ablation category, metric and label key in the style of the mouse comparison notebooks
(`analysis/mouse_ablation/*_comparison.ipynb`): bars show the mean over all Leiden resolutions (0.6-1.4) and
replicates, error bars the standard deviation (ddof = 0, as `np.std` in the notebooks), and dots the individual values
colored by replicate. The base run is shown as the reference bar of each category, labelled with its setting in that
category (the mouse architecture panel had no base bar). Runs without results are skipped and listed.

The category panels use the mouse evaluation (metatoken removed at inference). An extra panel,
`metatoken_inference`, compares removing and keeping the metatoken at inference for the base and metatoken dropout
runs, next to the run without metatoken.

Batch-integration metrics (datasets with two or more sections; `evaluate_ablation_run.py`) get the same panels, one
per category and metric plus `metatoken_inference`, if any run has them: iLISI (higher is better), ASW (the silhouette
of the sections; 0 is best, the value is plotted) and MMD (lower is better). Bars show the mean over replicates; the
bars are sorted best first.

Outputs under the artifact folder:
- `results/niche_identification_ablations.csv` (all metric rows), `results/niche_identification_ablations_summary.csv`
  (mean and SD per variant, metatoken mode and label key) and `figures/ablation_<category>_<label key>_<metric>.{svg,pdf,png}`
- with batch-integration metrics: `results/batch_integration_ablations.csv` (one row per run, metatoken mode and
  metric), `results/batch_integration_ablations_summary.csv` (mean and SD per variant and metatoken mode) and
  `figures/ablation_<category>_batch_integration_<metric>.{svg,pdf,png}`

Usage:
```bash
python compare_ablations.py --artifact_folder_path [ARTIFACT_FOLDER] --dataset_name [DATASET_NAME] --label_keys [KEY ...]
```
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ablation_utils import CATEGORY_ORDER

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
METRICS = ["NMI", "ARI"]
# Batch-integration metrics and which values are best: higher, lower or closest to zero (axis label suffix)
INTEGRATION_METRICS = {"iLISI": "higher", "ASW": "zero", "MMD": "lower"}
BEST_SYMBOLS = {"higher": "↑", "lower": "↓", "zero": "best: 0"}
EMB_KEY = "neighborhood_emb"
# Metatoken modes and the prefixes of their metric files (evaluate_ablation_run.py)
METATOKEN_MODES = {"drop": "{dataset_name}_", "keep": "{dataset_name}_keep_metatoken_"}
RUN_KEYS = ["run_id", "run_name", "variant", "category", "label", "replicate"]
# Panel layout (inches): the axes keep this width, the figure grows with the longest tick label
AXES_WIDTH = 1.9
VALUE_LABEL_WIDTH = 0.45
MAX_X_TICKS = 4

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


def sd(values: pd.Series) -> float:
    """Population standard deviation (ddof = 0), as np.std in the mouse comparison notebooks."""
    return float(np.std(values.to_numpy(dtype=float)))


def expected_modes(config_file_path: str) -> list:
    """Metatoken modes evaluated for a run: 'keep' only differs from 'drop' for models with special tokens."""
    with open(config_file_path) as file:
        special_tokens = yaml.safe_load(file)["meta"]["special_tokens"]
    return ["drop", "keep"] if special_tokens else ["drop"]


def collect_metrics(manifest_df: pd.DataFrame, dataset_name: str, label_keys: list) -> pd.DataFrame:
    """Metric rows of all evaluated runs; prints the (run, mode) pairs without results and missing label keys."""
    records, missing_pairs, missing_keys = [], [], []
    for _, run in manifest_df.iterrows():
        for mode in expected_modes(run["config_file_path"]):
            prefix = METATOKEN_MODES[mode].format(dataset_name=dataset_name)
            metrics_file_path = (Path(run["eval_folder_path"]) /
                                 f"{prefix}{EMB_KEY}_niche_identification_metrics_res_sweep.csv")
            if not metrics_file_path.exists():
                missing_pairs.append((run["run_id"], mode))
                continue
            metrics_df = pd.read_csv(metrics_file_path)
            absent_keys = [key for key in label_keys if key not in set(metrics_df["region_key"])]
            if absent_keys:
                missing_keys.append((run["run_id"], mode, absent_keys))
            metrics_df = metrics_df[metrics_df["region_key"].isin(label_keys)].copy()
            for key in RUN_KEYS:
                metrics_df[key] = run[key]
            metrics_df["metatoken_mode"] = mode
            records.append(metrics_df)
    n_expected = sum(len(expected_modes(file_path)) for file_path in manifest_df["config_file_path"])
    print(f"Results for {n_expected - len(missing_pairs)} of {n_expected} (run, metatoken mode) pairs.")
    if missing_pairs:
        print(f"Missing (run, metatoken mode) pairs ({len(missing_pairs)}):")
        for run_id, mode in missing_pairs:
            print(f"  {run_id}\t{mode}")
    for run_id, mode, absent_keys in missing_keys:
        print(f"  {run_id}\t{mode}: no metrics for label keys {absent_keys}")
    if not records:
        raise FileNotFoundError("No evaluated runs found.")
    return pd.concat(records, ignore_index=True)


def collect_integration_metrics(manifest_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Batch-integration metrics of all evaluated runs, one row per run and metatoken mode with a column per metric
    (empty if no run has them, e.g. for a single section); prints the (run, mode) pairs without them.
    """
    records, missing_pairs = [], []
    for _, run in manifest_df.iterrows():
        for mode in expected_modes(run["config_file_path"]):
            prefix = METATOKEN_MODES[mode].format(dataset_name=dataset_name)
            integration_file_path = Path(run["eval_folder_path"]) / f"{prefix}{EMB_KEY}_batch_integration_metrics.csv"
            if not integration_file_path.exists():
                missing_pairs.append((run["run_id"], mode))
                continue
            scores = pd.read_csv(integration_file_path).set_index("metric")["score"]
            records.append({**{key: run[key] for key in RUN_KEYS}, "metatoken_mode": mode,
                            **{metric: scores.get(metric, np.nan) for metric in INTEGRATION_METRICS}})
    if not records:
        print("No batch-integration metrics (single section, or no run evaluated yet).")
        return pd.DataFrame()
    print(f"Batch-integration metrics for {len(records)} of {len(records) + len(missing_pairs)} (run, metatoken mode) "
          f"pairs.")
    if missing_pairs:
        print(f"Missing (run, metatoken mode) pairs ({len(missing_pairs)}):")
        for run_id, mode in missing_pairs:
            print(f"  {run_id}\t{mode}")
    return pd.DataFrame(records)


def metatoken_inference_runs(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Runs of the metatoken_inference panel: base and metatoken dropout with the metatoken removed or kept at
    inference, and the run without metatoken."""
    metatoken_df = metrics_df[metrics_df["category"].isin(["base", "metatoken"])].copy()
    if not metatoken_df.empty:
        metatoken_df["label"] = metatoken_df.apply(
            lambda run: (run["label"] if run["variant"] == "no_metatk" else
                         f"{'Metatoken' if run['category'] == 'base' else run['label']} "
                         f"({'removed' if run['metatoken_mode'] == 'drop' else 'kept'})"), axis=1)
    return metatoken_df


def category_runs(metrics_df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Runs of a category panel with the mouse evaluation (metatoken removed), with the base run as reference bar."""
    metrics_df = metrics_df[metrics_df["metatoken_mode"] == "drop"]
    category_df = metrics_df[metrics_df["category"] == category]
    base_df = metrics_df[metrics_df["category"] == "base"]
    if category in BASE_LABELS and not category_df.empty and not base_df.empty:
        category_df = pd.concat([base_df.assign(label=BASE_LABELS[category]), category_df])
    return category_df


def tick_label_width(labels: list, fontsize: float) -> float:
    """Width in inches of the longest tick label, as rendered."""
    fig = plt.figure()
    renderer = fig.canvas.get_renderer()
    width = max(fig.text(0, 0, label, fontsize=fontsize).get_window_extent(renderer).width for label in labels)
    plt.close(fig)
    return width / fig.dpi


def plot_category(metrics_df: pd.DataFrame, metric: str, title: str, best: str = "higher"):
    """
    Horizontal bars (mean over resolutions and replicates) with SD and per-run dots, sorted best first (`best`: higher,
    lower or zero, i.e. closest to zero).
    """
    summary_df = metrics_df.groupby("label")[metric].agg(mean="mean", sd=sd)
    order = {"higher": summary_df["mean"].sort_values(ascending=False).index,
             "lower": summary_df["mean"].sort_values().index,
             "zero": summary_df["mean"].abs().sort_values().index}[best]
    summary_df = summary_df.loc[order]
    labels = list(summary_df.index)
    # The axes keep their width; the figure is as wide as the longest label needs, so labels never squeeze the axes
    fontsize = plt.rcParams["font.size"]
    width = tick_label_width(labels, fontsize) + AXES_WIDTH + VALUE_LABEL_WIDTH + 0.3
    fig, ax = plt.subplots(figsize=(width, 0.24 * len(summary_df) + 0.7), constrained_layout=True)
    y = np.arange(len(summary_df))
    ax.barh(y, summary_df["mean"], height=0.55, color="#56B4E9", edgecolor="white", alpha=0.8, zorder=2)
    ax.errorbar(summary_df["mean"], y, xerr=summary_df["sd"], fmt="none", ecolor="0.25", elinewidth=0.8,
                capsize=2.0, zorder=3)
    if best == "zero" or metrics_df[metric].min() < 0:
        ax.axvline(0, color="0.25", linewidth=0.6, zorder=1)
    # Value labels right of the bars and dots (and of zero, for negative bars)
    x_min, x_max = min(metrics_df[metric].min(), 0), max(metrics_df[metric].max(), 0)
    offset = 0.02 * (x_max - x_min) if x_max > x_min else 0.02
    rng = np.random.default_rng(42)
    for idx, label in enumerate(labels):
        label_df = metrics_df[metrics_df["label"] == label]
        ax.scatter(label_df[metric], idx + rng.uniform(-0.15, 0.15, len(label_df)), s=8,
                   color=[DOT_COLORS[(replicate - 1) % len(DOT_COLORS)] for replicate in label_df["replicate"]],
                   edgecolor="0.25", linewidth=0.4, zorder=4)
        mean = summary_df.loc[label, "mean"]
        ax.text(max(label_df[metric].max(), mean, 0) + offset, idx,
                f"{mean:.3g}" if 0 < abs(mean) < 0.01 else f"{mean:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric} ({BEST_SYMBOLS[best]})")
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=MAX_X_TICKS - 1))
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    return fig


def save_figure(fig, figures_folder_path: Path, name: str):
    for extension in ["svg", "pdf", "png"]:
        fig.savefig(figures_folder_path / f"{name}.{extension}")
    plt.close(fig)


def summarize(metrics_df: pd.DataFrame, group_keys: list, value_keys: list) -> pd.DataFrame:
    """Mean and SD (ddof = 0) of the values per group, with the number of replicates."""
    return (metrics_df.groupby(group_keys)[value_keys].agg(["mean", sd])
            .join(metrics_df.groupby(group_keys)["replicate"].nunique().rename(("n_replicates", ""))))


def main(args):
    artifact_folder_path = Path(args.artifact_folder_path)
    results_folder_path = artifact_folder_path / "results"
    figures_folder_path = artifact_folder_path / "figures"
    results_folder_path.mkdir(parents=True, exist_ok=True)
    figures_folder_path.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(artifact_folder_path / "manifest.tsv", sep="\t")
    all_metrics_df = collect_metrics(manifest_df, args.dataset_name, args.label_keys)
    all_metrics_df.to_csv(results_folder_path / "niche_identification_ablations.csv", index=False)
    summary_df = summarize(all_metrics_df, ["region_key", "category", "variant", "label", "metatoken_mode"],
                           ["NMI", "ARI", "n_clusters"])
    summary_df.to_csv(results_folder_path / "niche_identification_ablations_summary.csv")
    print(summary_df.round(3).to_string())

    for label_key in args.label_keys:
        key_df = all_metrics_df[all_metrics_df["region_key"] == label_key]
        if key_df.empty:
            continue

        # Metatoken at inference: removed (as in the mouse evaluation) vs kept, for runs trained with it
        metatoken_df = metatoken_inference_runs(key_df)
        if not metatoken_df.empty:
            for metric in METRICS:
                fig = plot_category(metatoken_df, metric, f"Metatoken at inference ({label_key})")
                save_figure(fig, figures_folder_path, f"ablation_metatoken_inference_{label_key}_{metric}")

        # Category panels with the mouse evaluation (metatoken removed at inference)
        for category in CATEGORY_ORDER[1:]:
            category_df = category_runs(key_df, category)
            if category_df.empty:
                continue
            for metric in METRICS:
                fig = plot_category(category_df, metric, f"{category.replace('_', ' ').capitalize()} ({label_key})")
                save_figure(fig, figures_folder_path, f"ablation_{category}_{label_key}_{metric}")

    # Batch integration between the sections, if evaluated: the same panels
    integration_df = collect_integration_metrics(manifest_df, args.dataset_name)
    if not integration_df.empty:
        integration_df.melt(id_vars=RUN_KEYS + ["metatoken_mode"], value_vars=list(INTEGRATION_METRICS),
                            var_name="metric", value_name="score").to_csv(
            results_folder_path / "batch_integration_ablations.csv", index=False)
        integration_summary_df = summarize(integration_df, ["category", "variant", "label", "metatoken_mode"],
                                           list(INTEGRATION_METRICS))
        integration_summary_df.to_csv(results_folder_path / "batch_integration_ablations_summary.csv")
        print(integration_summary_df.round(3).to_string())
        metatoken_df = metatoken_inference_runs(integration_df)
        for metric, best in INTEGRATION_METRICS.items():
            if not metatoken_df.empty:
                fig = plot_category(metatoken_df, metric, "Metatoken at inference (batch integration)", best)
                save_figure(fig, figures_folder_path, f"ablation_metatoken_inference_batch_integration_{metric}")
            for category in CATEGORY_ORDER[1:]:
                category_df = category_runs(integration_df, category)
                if category_df.empty:
                    continue
                fig = plot_category(category_df, metric,
                                    f"{category.replace('_', ' ').capitalize()} (batch integration)", best)
                save_figure(fig, figures_folder_path, f"ablation_{category}_batch_integration_{metric}")
    print(f"Saved results to {results_folder_path} and figures to {figures_folder_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--artifact_folder_path',
                        type=str,
                        required=True,
                        help='Folder with manifest.tsv, the run configs and the evaluation outputs.')
    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        help='Dataset name in the metric file names.')
    parser.add_argument('--label_keys',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Label keys (region_key of the metric files) to summarize and plot.')
    args = parser.parse_args()

    main(args)
