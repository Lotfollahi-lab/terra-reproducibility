"""
This script collects the evaluation outputs of all evaluated runs of an ablation campaign (`evaluate_ablation_run.py`)
and plots one panel per ablation category, metric and label key in the style of the mouse comparison notebooks
(`analysis/mouse_ablation/*_comparison.ipynb`). Runs count as evaluated if they have `evaluation_complete` (written last
by the evaluation); the others are listed.

Statistics: every run is first averaged over the Leiden resolutions (0.6-1.4), so that one run is one observation.
Bars show the mean over the runs (replicates) of an arm, error bars the standard deviation over them (ddof = 1;
`--error_bars ci95`: the half-width of the 95% t confidence interval), dots the runs colored by replicate, and every
tick label the number of runs n. Arms with one run get no error bar; panels with such an arm show a band around the
base mean: plus/minus the pooled seed SD (the SD within arms, pooled over all arms with two or more runs; with ci95,
the 95% interval of a single run around the base mean). Replicate r is trained with seed 1000 r in every arm, so the
tables also give the differences to the base run of the same seed (`_diff_to_base`; base in the same metatoken mode,
or for the run trained without the metatoken, base in the mode of the category panels, the comparison they show).
`--error_bars published` reproduces the published panels: mean and SD (ddof = 0, as np.std in the notebooks) over all
resolution x replicate values, so the SD mixes the resolution sweep with the seed spread and contains no seed spread
for arms with one run; it implies `--category_mode drop` (the published mode) unless that is given. Unlike the
published figure, the architecture panel has a base bar. Every figure states what its bars, error bars and band show.

Metatoken modes: the category panels show every run with its training input, the metatoken kept at inference for the
runs trained with it and removed for the run without (`--category_mode drop`, the default with `--error_bars
published`: removed for all, as in the mouse evaluation, which is out of distribution for runs trained with the
metatoken). The base run is the reference bar of
each category, labelled with its setting in that category. The metatoken contrasts have their own panels:
`metatoken_inference` (base and metatoken dropout with the metatoken removed and kept, next to the run without
metatoken) and `metatoken_contrasts` (seed-paired differences, METATOKEN_CONTRASTS: kept minus removed for metatoken
dropout, a clean inference-time test as both inputs occur in training, and for base, where removing it is also out of
distribution; no metatoken and metatoken dropout minus base, all with their training input, the effect of training
with the token). The per-arm differences between the modes are in the `_metatoken_mode_differences` tables.

Metrics:
- Niche identification: NMI, ARI and AMI (chance-adjusted), the value labels with the mean number of Leiden clusters
  (k, all embedded cells), as the cluster number at a resolution differs between arms. The tables add the scores at
  the resolution whose number of clusters among the labelled cells is closest to the number of label categories
  (`_matched`, a granularity-matched robustness check).
- Batch integration (two or more sections): iLISI relative to the iLISI of random mixing on the same graph
  (iLISI_relative, about 1: as mixed as random, comparable between section ratios; the panels show iLISI only if no
  run has it), ASW, MMD and ASW_label_<key> (the section silhouette within labels, for
  keys that label two or more sections; none for the mouse keys). These are label-free (except ASW_label) and reward
  mixing without shared biology, so they are to be read with the niche scores: `ablation_summary_<label key>` plots NMI
  against iLISI_relative per arm. Per label key, `cross_section_mixing` is the label-aware readout: the mean over the
  labels with at least MIN_CELLS_PER_LABEL cells (weighted by cells) of the fraction of a cell's kNN neighbors from
  other sections, relative to random mixing (1: as mixed as random, 0: no neighbors from other sections; the tables
  give the minimum and maximum over the labels, which show whether mixing is confined to some regions). For keys that
  label one section each (mouse), regions differ between the sections, so more mixing is not better: these panels are
  unsorted and have no direction.
- Collapse (`_avg_cosine_similarity.csv`): a run is flagged if the evaluation flagged complete or dimensional collapse,
  or if its participation ratio or between-cell variance fraction is below `--collapse_fraction` of the base mean
  (partial collapse, e.g. onto a few directions), base in the same metatoken mode or, for the run trained without the
  metatoken, in the mode of the category panels. Flagged runs are listed, marked in the tables
  and drawn as red crosses.

Evaluation data: the figures and tables state the gene set of the evaluation (`EVAL_GENE_SET` of `dataset.env`; mouse:
the shared-gene re-tokenization of the training cells, as published). Runs whose evaluation_complete records other
evaluation data than their config are excluded and listed. Runs evaluated by an earlier version (without AMI and the
granularity columns, the integration readouts, the collapse statistics or the evaluation data in evaluation_complete)
are included with a warning, their missing values empty; re-evaluate them (FORCE_EVAL=1).

Outputs under the artifact folder, `results/`:
- `niche_identification_ablations.csv` (all metric rows), `niche_identification_ablations_runs.csv` (per run,
  metatoken mode and label key: resolution means, matched scores, cross-section mixing, differences to base, collapse
  flag), `niche_identification_ablations_summary.csv` (per arm, metatoken mode and label key: n_runs and per value the
  mean, SD, 95% CI and number of runs with the value (`_n`, fewer than n_runs if a run lacks it), the published SD of
  NMI, ARI and AMI, the number of flagged runs and whether the row is the mode of the category panels) and `niche_identification_metatoken_mode_differences.csv`
- with batch-integration metrics: `batch_integration_ablations.csv` (one row per run, metatoken mode and metric),
  `batch_integration_ablations_summary.csv` and `batch_integration_metatoken_mode_differences.csv`
- `collapse_diagnostics.csv` (per run and metatoken mode: the statistics, their ratio to base and the flags)
- `ablation_summary.csv` (per arm and label key, in the mode of the category panels: niche scores, cluster numbers,
  integration readouts and collapse flags)
and `figures/`: `ablation_<category>_<label key>_<metric>` (NMI, ARI, AMI, cross_section_mixing),
`ablation_<category>_batch_integration_<metric>`, `ablation_metatoken_{inference,contrasts}_*` and
`ablation_summary_<label key>`, each as svg, pdf and png.

Usage:
```bash
python compare_ablations.py --artifact_folder_path [ARTIFACT_FOLDER] --dataset_name [DATASET_NAME] --label_keys [KEY ...]
```
"""

import argparse
import shlex
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from ablation_utils import CATEGORY_ORDER

BASE_LABELS = {"tokenizer": "Cell Graph (Combined)",
               "sequence_length": "64",
               "expression_encoding": "MLP",
               "spatial_encoding": "Rank",
               "masking_ratio": "0.6",
               "metatoken": "Metatoken",
               "normalization_unnormalized_ranks": "No Norm",
               "normalization_full": "No Norm",
               "normalization_unnormalized_counts": "No Norm",
               "architecture": "768 Dim, 3 Enc., 1 Pred. Layer"}
DOT_COLORS = ["#F0E442", "#E69F00", "#D55E00"]
FLAG_COLOR = "#CC0000"
CATEGORY_COLORS = plt.get_cmap("tab20").colors
NICHE_METRICS = ["NMI", "ARI", "AMI"]
# Columns of the niche metrics files added in Nov 2026 (empty for runs evaluated before)
GRANULARITY_COLUMNS = ["AMI", "n_labelled_cells", "n_label_categories", "n_clusters_labelled"]
CLUSTER_COLUMNS = ["n_clusters", "n_clusters_labelled", "n_label_categories", "n_labelled_cells"]
# Batch-integration metrics and which values are best: higher, lower or closest to zero (axis label suffix);
# ASW_label_<key> rows are best at zero as well
INTEGRATION_METRICS = {"iLISI": "higher", "iLISI_relative": "higher", "ASW": "zero", "MMD": "lower"}
ASW_LABEL_PREFIX = "ASW_label_"
BEST_SYMBOLS = {"higher": "↑", "lower": "↓", "zero": "best: 0"}
# Label-aware mixing per label key (_cross_section_neighbors.csv), over labels with at least this many cells
CROSS_SECTION_METRIC = "cross_section_mixing"
MIN_CELLS_PER_LABEL = 10
# What the readouts other than the niche scores mean (figure captions)
LABEL_FREE_TEXT = "Label-free: rewards mixing of the sections even without shared biology; read with the niche scores."
METRIC_TEXTS = {"iLISI": LABEL_FREE_TEXT,
                "iLISI_relative": "iLISI divided by the iLISI of random mixing (1). " + LABEL_FREE_TEXT,
                "ASW": "Silhouette of the sections. " + LABEL_FREE_TEXT,
                "MMD": "RBF MMD between the sections on standardized embeddings. " + LABEL_FREE_TEXT,
                ASW_LABEL_PREFIX: "Section silhouette within labels shared by the sections (region-conditioned).",
                CROSS_SECTION_METRIC: "Cross-section mixing: kNN neighbors from other sections relative to random "
                                      "mixing (dashed, 1), mean over the labels of the key; per region, not a "
                                      "quality score on its own (unsorted)."}
# Collapse statistics of _avg_cosine_similarity.csv; the relative ones are compared with the base mean
COLLAPSE_COLUMNS = ["participation_ratio", "between_cell_variance_fraction", "n_pcs_90_variance",
                    "top_pc_variance_fraction", "rankme", "mean_dim_std", "complete_collapse", "dimensional_collapse"]
RELATIVE_COLLAPSE_COLUMNS = ["participation_ratio", "between_cell_variance_fraction"]
EMB_KEY = "neighborhood_emb"
# Metatoken modes and the prefixes of their metric files (evaluate_ablation_run.py)
METATOKEN_MODES = {"drop": "{dataset_name}_", "keep": "{dataset_name}_keep_metatoken_"}
MODE_NAMES = {"drop": "removed", "keep": "kept"}
# Seed-paired contrasts of the metatoken panel: (label, (variant, mode), (variant, mode)), first minus second
METATOKEN_CONTRASTS = [
    ("Metatoken Dropout: kept - removed", ("metatoken_dropout", "keep"), ("metatoken_dropout", "drop")),
    ("Base: kept - removed", ("base", "keep"), ("base", "drop")),
    ("No Metatoken - Base (kept)", ("no_metatk", "drop"), ("base", "keep")),
    ("Metatoken Dropout - Base (both kept)", ("metatoken_dropout", "keep"), ("base", "keep"))]
# Columns of ablation_summary.csv printed at the end
PRINTED_SUMMARY_COLUMNS = ["region_key", "category", "label", "n_runs", "n_collapse_flagged", "NMI_mean", "NMI_sd",
                           "AMI_mean", "n_clusters_mean", "iLISI_relative_mean", "cross_section_mixing_mean",
                           "participation_ratio_mean"]
RUN_KEYS = ["run_id", "run_name", "variant", "category", "label", "replicate", "training_mode"]
EVAL_DATA_TEXTS = {"shared_genes": "shared-gene re-tokenization of the training cells (as published)",
                   "all_genes": "the run's tokenized training data (all genes)"}
# Panel layout (inches): the axes keep this width, the figure grows with the longest tick label
AXES_WIDTH = 1.9
VALUE_LABEL_WIDTH = 0.45
NOTE_LABEL_WIDTH = 0.35
CAPTION_CHARS_PER_INCH = 19
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


def published_sd(values: pd.Series) -> float:
    """Population standard deviation (ddof = 0), as np.std in the mouse comparison notebooks."""
    return float(np.std(values.to_numpy(dtype=float)))


def ci95(sd: pd.Series, n: pd.Series) -> pd.Series:
    """Half-width of the 95% t confidence interval of a mean (empty for one run)."""
    n = n.astype(float).where(n > 1)
    return stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)


def read_dataset_env(dataset_env_file_path: Path) -> dict:
    """Values of dataset.env (generate_ablation_runs.py; empty if there is none)."""
    if not dataset_env_file_path.exists():
        return {}
    values = {}
    for line in dataset_env_file_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = " ".join(shlex.split(value))
    return values


def read_marker(marker_file_path: Path) -> dict:
    """Values recorded in `evaluation_complete` (empty if there is none), as in evaluate_ablation_run.py."""
    if not marker_file_path.exists():
        return {}
    return dict(line.split(": ", 1) for line in marker_file_path.read_text().splitlines() if ": " in line)


def run_settings(config_file_path: str) -> dict:
    """
    Metatoken modes evaluated for a run ('keep' only differs from 'drop' for models with special tokens), the mode of
    its training input and the dataset its evaluation embeds (evaluate_ablation_run.py).
    """
    with open(config_file_path) as file:
        config = yaml.safe_load(file)
    modes = ["drop", "keep"] if config["meta"]["special_tokens"] else ["drop"]
    return {"modes": modes,
            "training_mode": modes[-1],
            "eval_data_path": (config.get("eval_tokenized_data_folder_path") or
                               config["data"]["tokenized_data_folder_path"])}


def print_pairs(message: str, pairs: list):
    if pairs:
        print(f"{message} ({len(pairs)}):")
        for pair in pairs:
            print("  " + "\t".join(str(value) for value in pair))


def evaluated_runs(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """
    The runs with evaluation_complete, without those whose evaluation_complete records other evaluation data than
    their config; prints the runs without it, the excluded runs and the runs evaluated by an earlier version that did
    not record the evaluation data (included).
    """
    keep, not_evaluated, other_data, unrecorded = [], [], [], []
    for index, run in manifest_df.iterrows():
        marker = read_marker(Path(run["eval_folder_path"]) / "evaluation_complete")
        if not marker:
            not_evaluated.append((run["run_id"],))
            continue
        if "eval_data_path" not in marker:
            unrecorded.append((run["run_id"],))
        elif marker["eval_data_path"] != run["eval_data_path"]:
            other_data.append((run["run_id"], marker["eval_data_path"]))
            continue
        keep.append(index)
    print(f"Evaluated runs (evaluation_complete): {len(keep) + len(other_data)} of {len(manifest_df)}.")
    print_pairs("Not evaluated (no evaluation_complete: not run yet, running or failed)", not_evaluated)
    print_pairs("WARNING: excluded, evaluation_complete records other evaluation data than the run config "
                "(re-evaluate)", other_data)
    print_pairs("WARNING: evaluation_complete does not record the evaluation data (evaluated by an earlier version; "
                "included, re-evaluate with FORCE_EVAL=1)", unrecorded)
    return manifest_df.loc[keep]


def collect_metrics(manifest_df: pd.DataFrame, dataset_name: str, label_keys: list) -> pd.DataFrame:
    """
    Metric rows of all evaluated runs; prints the (run, mode) pairs without results, missing label keys and the pairs
    evaluated before AMI and the granularity columns (GRANULARITY_COLUMNS; empty).
    """
    records, missing_pairs, missing_keys, old_pairs = [], [], [], []
    for _, run in manifest_df.iterrows():
        for mode in run["modes"]:
            prefix = METATOKEN_MODES[mode].format(dataset_name=dataset_name)
            metrics_file_path = (Path(run["eval_folder_path"]) /
                                 f"{prefix}{EMB_KEY}_niche_identification_metrics_res_sweep.csv")
            if not metrics_file_path.exists():
                missing_pairs.append((run["run_id"], mode))
                continue
            metrics_df = pd.read_csv(metrics_file_path)
            absent_keys = [key for key in label_keys if key not in set(metrics_df["region_key"])]
            if absent_keys:
                missing_keys.append((run["run_id"], mode, f"no metrics for label keys {absent_keys}"))
            absent_columns = [column for column in GRANULARITY_COLUMNS if column not in metrics_df]
            if absent_columns:
                old_pairs.append((run["run_id"], mode))
                metrics_df[absent_columns] = np.nan
            metrics_df = metrics_df[metrics_df["region_key"].isin(label_keys)].copy()
            for key in RUN_KEYS:
                metrics_df[key] = run[key]
            metrics_df["metatoken_mode"] = mode
            records.append(metrics_df)
    n_expected = sum(len(modes) for modes in manifest_df["modes"])
    print(f"Niche metrics for {n_expected - len(missing_pairs)} of {n_expected} (run, metatoken mode) pairs.")
    print_pairs("Missing (run, metatoken mode) pairs", missing_pairs)
    print_pairs("Missing label keys", missing_keys)
    print_pairs(f"WARNING: evaluated before {', '.join(GRANULARITY_COLUMNS)} were added (empty; re-evaluate with "
                f"FORCE_EVAL=1)", old_pairs)
    if not records:
        raise FileNotFoundError("No evaluated runs found.")
    return pd.concat(records, ignore_index=True)


def collect_integration_metrics(manifest_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Batch-integration metrics of all evaluated runs, one row per run and metatoken mode with a column per metric
    (INTEGRATION_METRICS and any ASW_label_<key>; empty if no run has them, e.g. for a single section); prints the
    (run, mode) pairs without them and those evaluated before iLISI_max and iLISI_relative were added.
    """
    records, missing_pairs, old_pairs = [], [], []
    for _, run in manifest_df.iterrows():
        for mode in run["modes"]:
            prefix = METATOKEN_MODES[mode].format(dataset_name=dataset_name)
            integration_file_path = Path(run["eval_folder_path"]) / f"{prefix}{EMB_KEY}_batch_integration_metrics.csv"
            if not integration_file_path.exists():
                missing_pairs.append((run["run_id"], mode))
                continue
            scores = pd.read_csv(integration_file_path).set_index("metric")["score"]
            if "iLISI_relative" not in scores:
                old_pairs.append((run["run_id"], mode))
            asw_label_scores = scores[scores.index.str.startswith(ASW_LABEL_PREFIX)].to_dict()
            records.append({**{key: run[key] for key in RUN_KEYS}, "metatoken_mode": mode,
                            **{metric: scores.get(metric, np.nan) for metric in INTEGRATION_METRICS},
                            **asw_label_scores})
    if not records:
        print("No batch-integration metrics (single section, or no run evaluated yet).")
        return pd.DataFrame()
    print(f"Batch-integration metrics for {len(records)} of {len(records) + len(missing_pairs)} (run, metatoken mode) "
          f"pairs.")
    print_pairs("Missing (run, metatoken mode) pairs", missing_pairs)
    print_pairs("WARNING: evaluated before iLISI_max and iLISI_relative were added (empty; re-evaluate with "
                "FORCE_EVAL=1)", old_pairs)
    return pd.DataFrame(records)


def collect_cross_section_mixing(manifest_df: pd.DataFrame, dataset_name: str, label_keys: list) -> pd.DataFrame:
    """
    Cross-section mixing per run, metatoken mode and label key from `_cross_section_neighbors.csv`: the mean over the
    labels with at least MIN_CELLS_PER_LABEL cells, weighted by cells, of the fraction of a cell's neighbors from other
    sections divided by the fraction expected under random mixing, with its minimum and maximum over the labels. Pairs
    with batch-integration metrics but without the file (earlier version) are printed.
    """
    records, old_pairs = [], []
    for _, run in manifest_df.iterrows():
        for mode in run["modes"]:
            prefix = METATOKEN_MODES[mode].format(dataset_name=dataset_name)
            neighbors_file_path = Path(run["eval_folder_path"]) / f"{prefix}{EMB_KEY}_cross_section_neighbors.csv"
            if not neighbors_file_path.exists():
                if (Path(run["eval_folder_path"]) / f"{prefix}{EMB_KEY}_batch_integration_metrics.csv").exists():
                    old_pairs.append((run["run_id"], mode))
                continue
            neighbors_df = pd.read_csv(neighbors_file_path)
            neighbors_df = neighbors_df[neighbors_df["n_cells"] >= MIN_CELLS_PER_LABEL]
            for key in label_keys:
                key_df = neighbors_df[neighbors_df["region_key"] == key]
                if key_df.empty:
                    continue
                ratio = key_df["cross_section_neighbor_fraction"] / key_df["expected_fraction"]
                records.append({**{run_key: run[run_key] for run_key in RUN_KEYS}, "metatoken_mode": mode,
                                "region_key": key,
                                CROSS_SECTION_METRIC: float(np.average(ratio, weights=key_df["n_cells"])),
                                f"{CROSS_SECTION_METRIC}_min": float(ratio.min()),
                                f"{CROSS_SECTION_METRIC}_max": float(ratio.max()),
                                f"{CROSS_SECTION_METRIC}_n_labels": len(key_df)})
    print_pairs("WARNING: batch-integration metrics without _cross_section_neighbors.csv (earlier version; "
                "re-evaluate with FORCE_EVAL=1)", old_pairs)
    return pd.DataFrame(records)


def collect_collapse_statistics(manifest_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Collapse statistics (COLLAPSE_COLUMNS) per run and metatoken mode; prints the pairs evaluated before they were
    added (only the cosine similarities; empty).
    """
    records, old_pairs = [], []
    for _, run in manifest_df.iterrows():
        for mode in run["modes"]:
            prefix = METATOKEN_MODES[mode].format(dataset_name=dataset_name)
            statistics_file_path = Path(run["eval_folder_path"]) / f"{prefix}{EMB_KEY}_avg_cosine_similarity.csv"
            if not statistics_file_path.exists():
                continue
            statistics = pd.read_csv(statistics_file_path).iloc[0]
            if "participation_ratio" not in statistics:
                old_pairs.append((run["run_id"], mode))
            records.append({**{key: run[key] for key in RUN_KEYS}, "metatoken_mode": mode,
                            **{column: statistics.get(column, np.nan) for column in COLLAPSE_COLUMNS}})
    print_pairs("WARNING: evaluated before the collapse statistics were added (not checked for collapse; re-evaluate "
                "with FORCE_EVAL=1)", old_pairs)
    return pd.DataFrame(records, columns=RUN_KEYS + ["metatoken_mode"] + COLLAPSE_COLUMNS)


def base_reference_modes(df: pd.DataFrame, category_mode: str) -> pd.Series:
    """
    Metatoken mode of the base runs each row is compared with: the row's own mode, or for runs trained with another
    input than base (No Metatoken), base in the mode of the category panels, the comparison those panels show.
    """
    base_training_modes = df.loc[df["category"] == "base", "training_mode"]
    if base_training_modes.empty:
        return df["metatoken_mode"]
    base_panel_mode = base_training_modes.iloc[0] if category_mode == "training" else "drop"
    return df["metatoken_mode"].where(df["training_mode"] == base_training_modes.iloc[0], base_panel_mode)


def collapse_flags(collapse_df: pd.DataFrame, collapse_fraction: float, category_mode: str) -> pd.DataFrame:
    """
    Adds the ratio of the relative collapse statistics (RELATIVE_COLLAPSE_COLUMNS) to the base mean in the reference
    mode (base_reference_modes) and the flag with its reasons: complete or dimensional collapse as flagged by the
    evaluation, or a ratio below `collapse_fraction` (partial collapse). Runs without the statistics are not flagged.
    """
    base_df = collapse_df[collapse_df["category"] == "base"].groupby("metatoken_mode")[RELATIVE_COLLAPSE_COLUMNS].mean()
    reference_modes = base_reference_modes(collapse_df, category_mode)
    for mode in set(reference_modes) - set(base_df.dropna().index):
        print(f"WARNING: no base run with collapse statistics in metatoken mode {mode}; runs compared with it are "
              f"flagged only by the evaluation's absolute thresholds.")
    for column in RELATIVE_COLLAPSE_COLUMNS:
        collapse_df[f"{column}_vs_base"] = (collapse_df[column].astype(float) /
                                            reference_modes.map(base_df[column]).astype(float))

    def reasons(run) -> str:
        found = [name.replace("_", " ") for name in ["complete_collapse", "dimensional_collapse"]
                 if str(run[name]) == "True"]
        found += [f"{column.replace('_', ' ')} {run[f'{column}_vs_base']:.0%} of base"
                  for column in RELATIVE_COLLAPSE_COLUMNS if run[f"{column}_vs_base"] < collapse_fraction]
        return "; ".join(found)

    collapse_df["collapse_reason"] = collapse_df.apply(reasons, axis=1) if len(collapse_df) else []
    collapse_df["collapse_flag"] = collapse_df["collapse_reason"] != ""
    return collapse_df


def in_category_panel(df: pd.DataFrame, category_mode: str) -> pd.Series:
    """Whether a row is in the metatoken mode of the category panels (the training input, or drop)."""
    return df["metatoken_mode"] == (df["training_mode"] if category_mode == "training" else "drop")


def add_run_flags(df: pd.DataFrame, collapse_df: pd.DataFrame, category_mode: str) -> pd.DataFrame:
    """Adds whether a row is in the mode of the category panels and the collapse flag of its run and mode."""
    df["category_panel"] = in_category_panel(df, category_mode)
    df = df.merge(collapse_df[["run_id", "metatoken_mode", "collapse_flag", "collapse_reason"]],
                  on=["run_id", "metatoken_mode"], how="left")
    df["collapse_flag"] = df["collapse_flag"].fillna(False).astype(bool)
    df["collapse_reason"] = df["collapse_reason"].fillna("")
    return df


def run_means(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per run, metatoken mode and label key: the scores and cluster numbers averaged over the Leiden resolutions,
    and the scores at the resolution whose number of clusters among the labelled cells is closest to the number of
    label categories (`_matched`, the lowest such resolution; empty without the granularity columns).
    """
    group_keys = RUN_KEYS + ["metatoken_mode", "region_key"]
    means_df = metrics_df.groupby(group_keys)[NICHE_METRICS + CLUSTER_COLUMNS].mean()
    means_df["n_resolutions"] = metrics_df.groupby(group_keys)["resolution"].nunique()
    distance = (metrics_df["n_clusters_labelled"] - metrics_df["n_label_categories"]).abs()
    matched_df = (metrics_df.assign(distance=distance).sort_values(group_keys + ["distance", "resolution"])
                  .groupby(group_keys).head(1).set_index(group_keys))
    for column in NICHE_METRICS + ["resolution"]:
        means_df[f"{column}_matched"] = matched_df[column].where(matched_df["distance"].notna())
    return means_df.reset_index()


def add_differences_to_base(df: pd.DataFrame, value_keys: list, by: list, category_mode: str) -> pd.DataFrame:
    """
    Adds `<value>_diff_to_base`: the difference to the base run of the same replicate (seed; and the same `by` values),
    with the base in the reference mode (base_reference_modes: the same metatoken mode, or for No Metatoken the mode of
    the category panels).
    """
    keys = by + ["replicate", "metatoken_mode"]
    base_df = df.loc[df["category"] == "base", keys + value_keys]
    reference_df = (df[by + ["replicate"]].assign(metatoken_mode=base_reference_modes(df, category_mode))
                    .merge(base_df, on=keys, how="left"))
    for key in value_keys:
        df[f"{key}_diff_to_base"] = df[key].to_numpy(dtype=float) - reference_df[key].to_numpy(dtype=float)
    return df


def summarize(df: pd.DataFrame, group_keys: list, value_keys: list) -> pd.DataFrame:
    """
    Per group (arm): the number of runs and per value the mean, SD (ddof = 1; empty for one run), half-width of the
    95% t confidence interval and number of runs with the value (`_n`; fewer than n_runs if a run lacks it, e.g. AMI
    of a run evaluated before it was added).
    """
    grouped = df.groupby(group_keys)
    summary_df = grouped["run_id"].nunique().rename("n_runs").to_frame()
    for key in value_keys:
        statistics_df = grouped[key].agg(["mean", "std", "count"])
        summary_df[f"{key}_mean"] = statistics_df["mean"]
        summary_df[f"{key}_sd"] = statistics_df["std"]
        summary_df[f"{key}_ci95"] = ci95(statistics_df["std"], statistics_df["count"])
        summary_df[f"{key}_n"] = statistics_df["count"]
    return summary_df


def mode_differences(df: pd.DataFrame, group_keys: list, value_keys: list) -> pd.DataFrame:
    """Per arm (runs trained with a metatoken): the summary over the runs of metatoken removed minus kept."""
    index_keys = ["run_id"] + group_keys
    keep_df = df[df["metatoken_mode"] == "keep"].set_index(index_keys)[value_keys]
    drop_df = df[df["metatoken_mode"] == "drop"].set_index(index_keys)[value_keys]
    difference_df = (drop_df - keep_df).dropna(how="all").reset_index()
    if difference_df.empty:
        return pd.DataFrame()
    return summarize(difference_df, group_keys, value_keys)


def pooled_seed_sd(df: pd.DataFrame, value_key: str) -> tuple[float, int]:
    """The SD of the runs within an arm pooled over the arms with two or more runs, and its degrees of freedom."""
    statistics_df = df.groupby("variant")[value_key].agg(["var", "count"])
    statistics_df = statistics_df[statistics_df["count"] >= 2]
    dof = int((statistics_df["count"] - 1).sum())
    if dof == 0:
        return np.nan, 0
    return float(np.sqrt(((statistics_df["count"] - 1) * statistics_df["var"]).sum() / dof)), dof


def metatoken_inference_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Runs of the metatoken_inference panel: base and metatoken dropout with the metatoken removed or kept at
    inference, and the run without metatoken."""
    metatoken_df = df[df["category"].isin(["base", "metatoken"])].copy()
    if not metatoken_df.empty:
        metatoken_df["label"] = metatoken_df.apply(
            lambda run: (run["label"] if run["variant"] == "no_metatk" else
                         f"{'Metatoken' if run['category'] == 'base' else run['label']} "
                         f"({MODE_NAMES[run['metatoken_mode']]})"), axis=1)
    return metatoken_df


def metatoken_contrasts(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Seed-paired differences of METATOKEN_CONTRASTS, one row per replicate with both runs."""
    records = []
    for label, (variant_a, mode_a), (variant_b, mode_b) in METATOKEN_CONTRASTS:
        runs_a = df[(df["variant"] == variant_a) & (df["metatoken_mode"] == mode_a)].set_index("replicate")
        runs_b = df[(df["variant"] == variant_b) & (df["metatoken_mode"] == mode_b)].set_index("replicate")
        for replicate in runs_a.index.intersection(runs_b.index):
            run_a, run_b = runs_a.loc[replicate], runs_b.loc[replicate]
            records.append({"label": label, "replicate": replicate, "run_id": run_a["run_id"],
                            metric: run_a[metric] - run_b[metric],
                            "collapse_flag": bool(run_a["collapse_flag"] or run_b["collapse_flag"])})
    return pd.DataFrame(records, columns=["label", "replicate", "run_id", metric, "collapse_flag"]).dropna()


def category_runs(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Runs of a category panel in the metatoken mode of the category panels, with the base run as reference bar."""
    df = df[df["category_panel"]]
    category_df = df[df["category"] == category]
    base_df = df[df["category"] == "base"]
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


def bar_statistics(points_df: pd.DataFrame, metric: str, error_bars: str) -> pd.DataFrame:
    """
    Per label (in order of appearance): the bar (mean), its error bar (`error_bars`: sd or ci95 over the runs, empty for
    one run; published: SD with ddof = 0 over all rows) and the number of runs.
    """
    grouped = points_df.groupby("label", sort=False)
    summary_df = grouped[metric].agg(mean="mean", sd="std", count="count")
    summary_df["n"] = grouped["run_id"].nunique()
    if error_bars == "published":
        summary_df["error"] = grouped[metric].agg(published_sd)
    elif error_bars == "ci95":
        summary_df["error"] = ci95(summary_df["sd"], summary_df["count"])
    else:
        summary_df["error"] = summary_df["sd"]
    return summary_df


def plot_category(points_df: pd.DataFrame, metric: str, title: str, caption: str, best: str = "higher",
                  error_bars: str = "sd", band: tuple = None, notes: dict = None, sort: bool = True,
                  xlabel: str = None, reference: float = None):
    """
    Horizontal bars per label (bar_statistics of `points_df`: one row per run, or per run and resolution for
    `error_bars` published) with error bars and dots, sorted best first (`best`: higher, lower or zero, i.e. closest
    to zero) unless `sort` is False. Tick labels give the number of runs, dots of collapse-flagged runs are crosses,
    `band` (center, half-width) is shaded behind the bars, `reference` is a dashed line, `notes` (per label) follow the
    values and `caption` (what bars, error bars and band show) is printed under the axes.
    """
    summary_df = bar_statistics(points_df, metric, error_bars)
    if sort:
        order = {"higher": summary_df["mean"].sort_values(ascending=False).index,
                 "lower": summary_df["mean"].sort_values().index,
                 "zero": summary_df["mean"].abs().sort_values().index}[best]
        summary_df = summary_df.loc[order]
    labels = list(summary_df.index)
    tick_labels = [f"{label}, n={summary_df.loc[label, 'n']}" for label in labels]
    # The axes keep their width; the figure is as wide as the longest label needs, so labels never squeeze the axes
    fontsize = plt.rcParams["font.size"]
    width = (tick_label_width(tick_labels, fontsize) + AXES_WIDTH + VALUE_LABEL_WIDTH +
             (NOTE_LABEL_WIDTH if notes else 0) + 0.3)
    fig, ax = plt.subplots(figsize=(width, 0.24 * len(summary_df) + 0.7), constrained_layout=True)
    y = np.arange(len(summary_df))
    if band is not None:
        ax.axvspan(band[0] - band[1], band[0] + band[1], color="0.88", linewidth=0, zorder=0)
    ax.barh(y, summary_df["mean"], height=0.55, color="#56B4E9", edgecolor="white", alpha=0.8, zorder=2)
    has_error = summary_df["error"].notna().to_numpy()
    ax.errorbar(summary_df["mean"][has_error], y[has_error], xerr=summary_df["error"][has_error], fmt="none",
                ecolor="0.25", elinewidth=0.8, capsize=2.0, zorder=3)
    if best == "zero" or points_df[metric].min() < 0:
        ax.axvline(0, color="0.25", linewidth=0.6, zorder=1)
    if reference is not None:
        ax.axvline(reference, color="0.25", linewidth=0.6, linestyle="--", zorder=1)
    # Value labels right of the bars, dots, error bars and band (and of zero, for negative bars)
    band_max = band[0] + band[1] if band is not None else 0
    x_min, x_max = min(points_df[metric].min(), 0), max(points_df[metric].max(), band_max, 0)
    offset = 0.02 * (x_max - x_min) if x_max > x_min else 0.02
    rng = np.random.default_rng(42)
    for idx, label in enumerate(labels):
        label_df = points_df[points_df["label"] == label]
        jitter = idx + rng.uniform(-0.15, 0.15, len(label_df))
        flagged = label_df["collapse_flag"].to_numpy(dtype=bool)
        colors = [DOT_COLORS[(replicate - 1) % len(DOT_COLORS)] for replicate in label_df["replicate"][~flagged]]
        ax.scatter(label_df[metric][~flagged], jitter[~flagged], s=8, color=colors, edgecolor="0.25", linewidth=0.4,
                   zorder=4)
        ax.scatter(label_df[metric][flagged], jitter[flagged], s=18, marker="x", color=FLAG_COLOR, linewidth=0.9,
                   zorder=5)
        mean, error = summary_df.loc[label, "mean"], summary_df.loc[label, "error"]
        value = f"{mean:.3g}" if 0 < abs(mean) < 0.01 else f"{mean:.3f}"
        ax.text(max(label_df[metric].max(), mean + (error if np.isfinite(error) else 0), 0) + offset, idx,
                value + (f" | {notes[label]}" if notes else ""), va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel or f"{metric} ({BEST_SYMBOLS[best]})")
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=MAX_X_TICKS - 1))
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    if points_df["collapse_flag"].any():
        caption += " Red crosses: runs flagged for collapse (collapse_diagnostics.csv)."
    ax.annotate(textwrap.fill(caption, int(width * CAPTION_CHARS_PER_INCH)), xy=(0, 0), xycoords="axes fraction",
                xytext=(0, -30), textcoords="offset points", va="top", fontsize=6.5, color="0.3").set_in_layout(False)
    return fig


def plot_summary(summary_df: pd.DataFrame, x_metric: str, title: str, caption: str):
    """NMI against a batch-integration metric per arm, colored by category; arms with a flagged run as crosses."""
    fig, ax = plt.subplots(figsize=(4.6, 3.6), constrained_layout=True)
    for category, category_df in summary_df.groupby("category", sort=False):
        color = CATEGORY_COLORS[CATEGORY_ORDER.index(category) % len(CATEGORY_COLORS)]
        flagged = category_df["n_collapse_flagged"].to_numpy() > 0
        ax.scatter(category_df[f"{x_metric}_mean"][~flagged], category_df["NMI_mean"][~flagged], s=16, color=color,
                   edgecolor="0.25", linewidth=0.4, label=category.replace("_", " "), zorder=3)
        ax.scatter(category_df[f"{x_metric}_mean"][flagged], category_df["NMI_mean"][flagged], s=24, marker="x",
                   color=FLAG_COLOR, linewidth=0.9, zorder=4)
        for _, arm in category_df.iterrows():
            ax.annotate("Base" if category == "base" else arm["label"], (arm[f"{x_metric}_mean"], arm["NMI_mean"]),
                        xytext=(2, 2), textcoords="offset points", fontsize=4.5, color="0.3")
    ax.set_xlabel(f"{x_metric} ({BEST_SYMBOLS[INTEGRATION_METRICS.get(x_metric, 'higher')]}, label-free)")
    ax.set_ylabel("NMI (↑)")
    ax.set_title(title, fontsize=10)
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=5.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    if (summary_df["n_collapse_flagged"] > 0).any():
        caption += " Red crosses: arms with a run flagged for collapse."
    ax.annotate(textwrap.fill(caption, int(6.0 * CAPTION_CHARS_PER_INCH)), xy=(0, 0), xycoords="axes fraction",
                xytext=(0, -30), textcoords="offset points", va="top", fontsize=6.5, color="0.3").set_in_layout(False)
    return fig


def write_table(df: pd.DataFrame, file_path: Path):
    """Writes a table with its index, unless it is empty (e.g. no run with both metatoken modes yet)."""
    if not df.empty:
        df.to_csv(file_path)


def save_figure(fig, figures_folder_path: Path, name: str):
    # The layout is fixed without the captions under the axes (constrained layout would shrink the axes to fit them),
    # then the saved area includes them
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for annotation in fig.findobj(mpl.text.Annotation):
        annotation.set_in_layout(True)
    for extension in ["svg", "pdf", "png"]:
        fig.savefig(figures_folder_path / f"{name}.{extension}")
    plt.close(fig)


def caption_text(args, eval_data_text: str, n_resolutions: int = None, band: tuple = None, band_dof: int = 0,
                 mode_text: str = None, notes: bool = False, metric_text: str = None) -> str:
    """
    What the bars, error bars, dots, band, metatoken mode, value labels, metric and evaluation data of a panel are.
    """
    averaged = f", each averaged over {n_resolutions} Leiden resolutions" if n_resolutions else ""
    if args.error_bars == "published":
        unit = "resolution x replicate values" if n_resolutions else "runs"
        text = (f"Bars: mean, error bars: SD (ddof 0) over all {unit} (published style; the SD includes the "
                f"resolution sweep); dots: these values; n: runs.")
    else:
        error_text = "SD (ddof 1)" if args.error_bars == "sd" else "95% t confidence interval"
        text = (f"Bars: mean of the runs{averaged}; error bars: {error_text} over the runs (none for n=1); "
                f"dots: runs by replicate; n: runs.")
    if band is not None:
        text += (f" Grey band: base mean +/- pooled seed SD (df {band_dof})." if args.error_bars == "sd" else
                 f" Grey band: 95% interval of one run around the base mean (pooled seed SD, df {band_dof}).")
    if mode_text:
        text += f" {mode_text}"
    if notes:
        text += " k: mean number of Leiden clusters."
    if metric_text:
        text += f" {metric_text}"
    return text + f" Evaluation: {eval_data_text}."


def category_band(category_df: pd.DataFrame, metric: str, pooled: tuple, error_bars: str) -> tuple | None:
    """
    Band around the base mean for panels with an arm of one run: the pooled seed SD, or with ci95 the 95% interval of
    one run around the base mean; none without base runs, pooled SD or with the published error bars.
    """
    sd, dof = pooled
    base_values = category_df.loc[category_df["category"] == "base", metric].dropna()
    n_runs = category_df[category_df["category"] != "base"].groupby("variant")["run_id"].nunique()
    if error_bars == "published" or base_values.empty or dof == 0 or not (n_runs == 1).any():
        return None
    if error_bars == "ci95":
        return float(base_values.mean()), float(stats.t.ppf(0.975, dof) * sd * np.sqrt(1 + 1 / len(base_values)))
    return float(base_values.mean()), sd


def plot_metric_panels(df: pd.DataFrame, points_df: pd.DataFrame, metric: str, best: str, name: str,
                       title_suffix: str, args, eval_data_text: str, figures_folder_path: Path,
                       n_resolutions: int = None, notes_key: str = None, metric_text: str = None,
                       direction: bool = True, xlabel: str = None, reference: float = None):
    """
    The panels of one metric: metatoken_inference, metatoken_contrasts and one per category. `df` has one row per run
    and mode; `points_df` the rows the bars and dots are computed from (with published error bars, the resolution
    rows); `metric_text` is added to the captions. Without `direction` (a readout with no better side), the bars are
    not sorted, `xlabel` replaces the axis label with its arrow and `reference` is drawn as a dashed line.
    """
    df = df[df[metric].notna()]
    points_df = points_df[points_df[metric].notna()]
    if df.empty:
        return
    category_mode_text = ("Metatoken kept at inference for runs trained with it (removed for No Metatoken)." if
                          args.category_mode == "training" else
                          "Metatoken removed at inference (mouse evaluation; out of distribution for runs trained "
                          "with it).")
    def notes(plotted_df: pd.DataFrame) -> dict | None:
        """Mean number of clusters per label, after the value."""
        if not notes_key:
            return None
        return plotted_df.groupby("label")[notes_key].mean().map(lambda k: f"k {k:.0f}").to_dict()

    # Metatoken at inference: removed (as in the mouse evaluation) vs kept, for runs trained with it
    metatoken_df = metatoken_inference_runs(points_df)
    if not metatoken_df.empty:
        caption = caption_text(args, eval_data_text, n_resolutions, notes=bool(notes_key), metric_text=metric_text)
        fig = plot_category(metatoken_df, metric, f"Metatoken at inference ({title_suffix})", caption, best,
                            args.error_bars, notes=notes(metatoken_df), sort=direction, xlabel=xlabel,
                            reference=reference)
        save_figure(fig, figures_folder_path, f"ablation_metatoken_inference_{name}_{metric}")
    contrast_df = metatoken_contrasts(df, metric)
    if not contrast_df.empty:
        error_text = "SD (ddof 1)" if args.error_bars != "ci95" else "95% t confidence interval"
        averaged = f" (runs averaged over {n_resolutions} Leiden resolutions)" if n_resolutions else ""
        caption = (f"Seed-paired differences, first minus second{averaged}; bars: mean, error bars: {error_text} over "
                   f"the replicates; dots: replicates. Dropout kept - removed: inference-time effect, both inputs "
                   f"occur in training. Base kept - removed: removing the metatoken is out of distribution for base "
                   f"(confounded). No Metatoken / Dropout - Base: effect of training with the token, with the "
                   f"training inputs. {metric_text + ' ' if metric_text else ''}Evaluation: {eval_data_text}.")
        fig = plot_category(contrast_df, metric, f"Metatoken contrasts ({title_suffix})", caption,
                            error_bars="ci95" if args.error_bars == "ci95" else "sd", sort=False,
                            xlabel=f"Difference in {metric}")
        save_figure(fig, figures_folder_path, f"ablation_metatoken_contrasts_{name}_{metric}")

    # Category panels in the mode of the training input (or drop), with the base run as reference bar
    pooled = pooled_seed_sd(df[df["category_panel"]], metric)
    for category in CATEGORY_ORDER[1:]:
        category_df = category_runs(df, category)
        if category_df.empty:
            continue
        band = category_band(category_df, metric, pooled, args.error_bars)
        category_points_df = category_runs(points_df, category)
        caption = caption_text(args, eval_data_text, n_resolutions, band, pooled[1], category_mode_text,
                               bool(notes_key), metric_text)
        fig = plot_category(category_points_df, metric, f"{category.replace('_', ' ').capitalize()} ({title_suffix})",
                            caption, best, args.error_bars, band, notes(category_points_df), sort=direction,
                            xlabel=xlabel, reference=reference)
        save_figure(fig, figures_folder_path, f"ablation_{category}_{name}_{metric}")


def main(args):
    # The published panels removed the metatoken for all runs
    if args.category_mode is None:
        args.category_mode = "drop" if args.error_bars == "published" else "training"
    artifact_folder_path = Path(args.artifact_folder_path)
    results_folder_path = artifact_folder_path / "results"
    figures_folder_path = artifact_folder_path / "figures"
    results_folder_path.mkdir(parents=True, exist_ok=True)
    figures_folder_path.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(artifact_folder_path / "manifest.tsv", sep="\t")
    manifest_df = manifest_df.join(pd.DataFrame([run_settings(file_path) for file_path in
                                                 manifest_df["config_file_path"]], index=manifest_df.index))
    eval_gene_set = read_dataset_env(artifact_folder_path / "dataset.env").get("EVAL_GENE_SET", "unknown")
    eval_data_text = EVAL_DATA_TEXTS.get(eval_gene_set, f"EVAL_GENE_SET {eval_gene_set}")
    print(f"Evaluation data (EVAL_GENE_SET {eval_gene_set}): {eval_data_text}.")
    manifest_df = evaluated_runs(manifest_df)

    # Collapse flags per run and mode, used by every table and figure
    collapse_df = collapse_flags(collect_collapse_statistics(manifest_df, args.dataset_name), args.collapse_fraction,
                                 args.category_mode)
    collapse_df.to_csv(results_folder_path / "collapse_diagnostics.csv", index=False)
    print_pairs(f"Runs flagged for collapse (participation ratio or between-cell variance fraction below "
                f"{args.collapse_fraction:.0%} of base, or flagged by the evaluation)",
                list(collapse_df.loc[collapse_df["collapse_flag"], ["run_id", "metatoken_mode", "collapse_reason"]]
                     .itertuples(index=False)))

    # Niche identification: per run (mean over resolutions) and per arm
    all_metrics_df = add_run_flags(collect_metrics(manifest_df, args.dataset_name, args.label_keys), collapse_df,
                                   args.category_mode)
    all_metrics_df.to_csv(results_folder_path / "niche_identification_ablations.csv", index=False)
    n_resolutions = all_metrics_df["resolution"].nunique()
    run_df = run_means(all_metrics_df)
    mixing_df = collect_cross_section_mixing(manifest_df, args.dataset_name, args.label_keys)
    if not mixing_df.empty:
        run_df = run_df.merge(mixing_df, on=RUN_KEYS + ["metatoken_mode", "region_key"], how="left")
    run_df = add_run_flags(run_df, collapse_df, args.category_mode)
    # Scores and readouts compared between arms and modes; the label counts are summarized only
    niche_value_keys = (NICHE_METRICS + [f"{metric}_matched" for metric in NICHE_METRICS] + ["n_clusters"] +
                        ([CROSS_SECTION_METRIC] if CROSS_SECTION_METRIC in run_df else []))
    run_df = add_differences_to_base(run_df, niche_value_keys, ["region_key"], args.category_mode)
    run_df["eval_gene_set"] = eval_gene_set
    run_df.to_csv(results_folder_path / "niche_identification_ablations_runs.csv", index=False)
    arm_keys = ["category", "variant", "label", "metatoken_mode"]
    summary_df = summarize(run_df, ["region_key"] + arm_keys,
                           niche_value_keys + CLUSTER_COLUMNS[1:] +
                           [f"{key}_diff_to_base" for key in niche_value_keys])
    for metric in NICHE_METRICS:
        summary_df[f"{metric}_sd_published"] = all_metrics_df.groupby(["region_key"] + arm_keys)[metric].agg(
            published_sd)
    summary_df["n_collapse_flagged"] = run_df.groupby(["region_key"] + arm_keys)["collapse_flag"].sum()
    summary_df["category_panel"] = run_df.groupby(["region_key"] + arm_keys)["category_panel"].first()
    summary_df["eval_gene_set"] = eval_gene_set
    summary_df.to_csv(results_folder_path / "niche_identification_ablations_summary.csv")
    write_table(mode_differences(run_df, ["region_key", "category", "variant", "label"], niche_value_keys),
                results_folder_path / "niche_identification_metatoken_mode_differences.csv")

    for label_key in args.label_keys:
        key_df = run_df[run_df["region_key"] == label_key]
        if key_df.empty:
            continue
        key_points_df = (all_metrics_df[all_metrics_df["region_key"] == label_key] if args.error_bars == "published"
                         else key_df)
        for metric in NICHE_METRICS:
            plot_metric_panels(key_df, key_points_df, metric, "higher", label_key, label_key, args, eval_data_text,
                               figures_folder_path, n_resolutions, notes_key="n_clusters")
        if CROSS_SECTION_METRIC in key_df:
            plot_metric_panels(key_df, key_df, CROSS_SECTION_METRIC, "higher", label_key,
                               f"{label_key}, label-aware mixing", args, eval_data_text, figures_folder_path,
                               metric_text=METRIC_TEXTS[CROSS_SECTION_METRIC], direction=False,
                               xlabel="Cross-section neighbors / random mixing", reference=1.0)

    # Batch integration between the sections, if evaluated: the same panels
    integration_df = collect_integration_metrics(manifest_df, args.dataset_name)
    integration_summary_df = pd.DataFrame()
    if not integration_df.empty:
        integration_df = add_run_flags(integration_df, collapse_df, args.category_mode)
        metrics = [metric for metric in integration_df.columns
                   if (metric in INTEGRATION_METRICS or metric.startswith(ASW_LABEL_PREFIX)) and
                   integration_df[metric].notna().any()]
        integration_df = add_differences_to_base(integration_df, metrics, [], args.category_mode)
        integration_df.melt(id_vars=RUN_KEYS + ["metatoken_mode", "category_panel", "collapse_flag"],
                            value_vars=metrics, var_name="metric", value_name="score").to_csv(
            results_folder_path / "batch_integration_ablations.csv", index=False)
        integration_summary_df = summarize(integration_df, arm_keys,
                                           metrics + [f"{metric}_diff_to_base" for metric in metrics])
        integration_summary_df["n_collapse_flagged"] = integration_df.groupby(arm_keys)["collapse_flag"].sum()
        integration_summary_df["category_panel"] = integration_df.groupby(arm_keys)["category_panel"].first()
        integration_summary_df["eval_gene_set"] = eval_gene_set
        integration_summary_df.to_csv(results_folder_path / "batch_integration_ablations_summary.csv")
        write_table(mode_differences(integration_df, ["category", "variant", "label"], metrics),
                    results_folder_path / "batch_integration_metatoken_mode_differences.csv")
        # iLISI relative to random mixing replaces iLISI in the panels, unless no run has it
        plot_metrics = [metric for metric in metrics if not (metric == "iLISI" and "iLISI_relative" in metrics)]
        for metric in plot_metrics:
            plot_metric_panels(integration_df, integration_df, metric, INTEGRATION_METRICS.get(metric, "zero"),
                               "batch_integration", "batch integration", args, eval_data_text, figures_folder_path,
                               metric_text=METRIC_TEXTS.get(metric, METRIC_TEXTS[ASW_LABEL_PREFIX]))

    # Summary per arm and label key in the mode of the category panels: niche scores, granularity, integration and
    # collapse
    summary_columns = (["n_runs", "n_collapse_flagged"] +
                       [f"{metric}_{statistic}" for metric in NICHE_METRICS for statistic in ["mean", "sd"]] +
                       [f"{column}_mean" for column in ["n_clusters", "n_clusters_labelled", "n_label_categories"] +
                        [f"{metric}_matched" for metric in NICHE_METRICS] +
                        ([CROSS_SECTION_METRIC] if CROSS_SECTION_METRIC in run_df else [])])
    ablation_summary_df = summary_df.loc[summary_df["category_panel"], summary_columns].reset_index()
    if not integration_summary_df.empty:
        integration_columns = [f"{metric}_mean" for metric in ["iLISI_relative", "iLISI", "ASW", "MMD"]
                               if f"{metric}_mean" in integration_summary_df]
        ablation_summary_df = ablation_summary_df.merge(
            integration_summary_df.loc[integration_summary_df["category_panel"], integration_columns].reset_index(),
            on=arm_keys, how="left")
    panel_collapse_df = collapse_df[in_category_panel(collapse_df, args.category_mode)]
    ablation_summary_df = ablation_summary_df.merge(
        panel_collapse_df.groupby(arm_keys)[RELATIVE_COLLAPSE_COLUMNS].mean().add_suffix("_mean").reset_index(),
        on=arm_keys, how="left")
    ablation_summary_df["eval_gene_set"] = eval_gene_set
    # Label keys as given, categories in figure order, arms in manifest order
    variant_order = {variant: index for index, variant in enumerate(manifest_df["variant"].unique())}
    ablation_summary_df = ablation_summary_df.sort_values(
        ["region_key", "category", "variant"],
        key=lambda column: column.map({"region_key": {key: index for index, key in enumerate(args.label_keys)},
                                       "category": {category: index for index, category in enumerate(CATEGORY_ORDER)},
                                       "variant": variant_order}[column.name]))
    ablation_summary_df.to_csv(results_folder_path / "ablation_summary.csv", index=False)
    mode_name = "training input" if args.category_mode == "training" else "metatoken removed"
    print(f"Summary per arm ({mode_name}; evaluation: {eval_data_text}):")
    print(ablation_summary_df[[column for column in PRINTED_SUMMARY_COLUMNS if column in ablation_summary_df]]
          .round(3).to_string(index=False))

    x_metric = next((metric for metric in ["iLISI_relative", "iLISI"] if f"{metric}_mean" in ablation_summary_df and
                     ablation_summary_df[f"{metric}_mean"].notna().any()), None)
    if x_metric:
        for label_key in args.label_keys:
            key_summary_df = ablation_summary_df[ablation_summary_df["region_key"] == label_key]
            key_summary_df = key_summary_df[key_summary_df[[f"{x_metric}_mean", "NMI_mean"]].notna().all(axis=1)]
            if key_summary_df.empty:
                continue
            caption = (f"Arm means over the runs ({mode_name}). The integration metric is label-free: a high value "
                       f"without a high NMI is mixing without niche structure. Evaluation: {eval_data_text}.")
            fig = plot_summary(key_summary_df, x_metric, f"Niche identification vs integration ({label_key})", caption)
            save_figure(fig, figures_folder_path, f"ablation_summary_{label_key}")
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
    parser.add_argument('--error_bars',
                        type=str,
                        choices=["sd", "ci95", "published"],
                        default="sd",
                        help='Error bars over the runs of an arm, each averaged over the resolutions: SD (ddof 1) or '
                             'the 95%% t confidence interval; published: SD (ddof 0) over all resolution x replicate '
                             'values, as in the published panels (with --category_mode drop, its default then).')
    parser.add_argument('--category_mode',
                        type=str,
                        choices=["training", "drop"],
                        default=None,
                        help='Metatoken mode of the category panels: the training input (kept for runs trained with '
                             'the metatoken; the default) or removed for all runs (mouse evaluation; the default with '
                             '--error_bars published).')
    parser.add_argument('--collapse_fraction',
                        type=float,
                        default=0.25,
                        help='Flag runs whose participation ratio or between-cell variance fraction is below this '
                             'fraction of the base mean (same metatoken mode; No Metatoken: the category-panel mode).')
    args = parser.parse_args()

    main(args)
