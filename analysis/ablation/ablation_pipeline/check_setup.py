"""
This script checks the setup of an ablation campaign on the farm before launching the job array:
0. Pinned code and campaign: the pinned TERRA copy (CODE_PATH) is the one imported and has the options of the plan,
   the metatoken inference option and meta.zero_fully_masked_attention_rows, its CODE_VERSION is the `code_version` of
   the configs, and `dataset.env` of the artifact folder (what the job array reads) holds the values of the arguments,
   i.e. the configs were regenerated after the last change of the env file.
1. Config parity: every variant's generated config is compared with the params of its mouse runs (model_config.yaml
   or params.yaml under --mouse_artifact_folder_path). Expected differences (data paths, special value rows, run
   folder and resumption, seed, step-matched epochs, code version, the TERRA-only options) are ignored; any other
   difference means the plan does not reproduce the mouse experiment and should be resolved first. Inferred overrides
   (cell-neighborhood layout, dimension ablations) are verified here. Fails if the model layout of a cell-neighborhood
   arm (seq_len_cell, seq_len_neighborhood, and model_input_size unless the mouse data held the same tokens at the
   positions the model reads) differs from its mouse runs or cannot be checked, if a config does not set
   meta.zero_fully_masked_attention_rows: true, or if the replicates of a variant differ in more than seed and run
   folder (replicates beyond the mouse runs, e.g. of the normalization arms, are compared through replicate 1).
2. Special values and vocabulary genes: the batch special value of every silver file fits into the special value
   embedding of the configs. TERRA maps the special values of `gt_type: rank` models to tokens with an offset
   hard-coded by vocabulary size: fails if it differs from the token dictionary's offset while the plan has a rank arm
   (the mouse dictionary is reported too: whether the mouse rank arms trained with the right metatoken). Fails if the
   batch metatoken's token id hard-coded in TERRA's datasets is not 'spt_batch' of the token dictionary, if the
   configs' vocab_size is not the size of the token dictionary, if two silver files share a batch special value (the
   metatoken would not tell the sections apart), if the median nearest-neighbour distances of the sections'
   obsm['spatial'] differ by more than 2x (coordinates in different units; printed for every section), if a silver cell
   has no detected gene of the token dictionary (its ranked gene lists are empty, which shifts the segments of the cells
   around it on the sparse ranking path), or if a silver X stores explicit zeros (zero-count gene tokens on the sparse
   ranking path); filter_silver_cells.py removes both.
3. Labels and sections: every label file (LABEL_FILE_PATHS, default: the silver files) is one section, the batch of the
   integration metrics. Prints its cells, uns dataset id and batch, label keys and their most frequent values (to spot
   the unassigned value) and how many cells carry the unassigned label. Fails if a file is missing or has no
   obs['cell_id'], a label key is in no file, a cell id occurs twice, UNASSIGNED_LABEL is still the placeholder of the
   env file, or, with two or more sections, scib_metrics does not import.
4. Tokenized datasets: each exists, was made from the current inputs (its ablation_provenance.json: TERRA code version,
   tokenization settings, md5 of the token dictionary, the factor file and the silver files), has one row per silver
   cell and all its cells are in the label files (with the cells per section). With --eval_gene_set shared_genes, the
   same for the shared-gene evaluation data: its silver files (filter_shared_genes.py) were made from the current
   silver files, and its tokenized datasets (every tokenization of the plan, same token dictionary and factor file).
5. Normalization factors: every feature of the silver files has an Ensembl id, a row of the factor file with finite
   positive factors and counts in at least one section (the tokenizer looks up the factors of every feature by id);
   the gene-corrected read depth columns of the dataset's factor file match the tokenizer's own normalizer (same
   constants), and every silver gene has factors (`norm_factors.check_norm_factors`). The column of
   the plan's `rank_tie_break_norm_factor` has a finite value for every vocabulary gene of the silver files. Reports,
   for each per-cell sequence length of the plan, how many cells have more detected genes and how many of them a
   raw-count ranking (the base and every arm whose rank is not normalized) cuts inside tied counts, i.e. where the
   tie order decides which genes are kept.
6. Schedule end values (report): the LR, weight decay and EMA schedules span `ipe_scale` x epochs (1.25, as in
   V-JEPA), but training stops after `epochs`, so they end at 80% of their path; the values to state in Methods.
7. Cells with few detected genes (report for Methods): with one target block, the targets are ceil(r * K) of the K gene
   tokens of the kept segments, so in index-cell-only batches (1 in 11) a cell has no context gene if ceil(r * K) = K
   (K <= 2 at the base ratio 0.6, K <= 4 at 0.8, K <= 9 at 0.9); per section, the share of cells with at most that
   many detected vocabulary genes for every mask ratio of the plan.
Exits with an error if check 0, 1, 2, 3, 4 or 5 fails; the other check 1 differences have to be reviewed by eye.
--skip_datasets skips check 4 (before tokenization).

Usage (`source datasets/<dataset>.env` first, see README.md; add --skip_datasets before tokenization):
```bash
python check_setup.py --plan_file_path "${PLAN_FILE_PATH}" --artifact_folder_path "${ARTIFACT_FOLDER_PATH}" \
    --data_directory_path "${DATA_DIRECTORY_PATH}" --dataset_name "${DATASET_NAME}" \
    --silver_file_paths ${SILVER_FILE_PATHS} --label_file_paths ${LABEL_FILE_PATHS:-${SILVER_FILE_PATHS}} \
    --label_keys ${LABEL_KEYS} --unassigned_label "${UNASSIGNED_LABEL}" \
    --norm_factor_file_path "${NORM_FACTOR_FILE_PATH}" \
    --token_dictionary_file_path "${TOKEN_DICTIONARY_FILE_PATH}" --code_path "${CODE_PATH}" \
    --mouse_token_dictionary_file_path "${MOUSE_TOKEN_DICTIONARY_FILE_PATH}" \
    --mouse_artifact_folder_path "${MOUSE_ARTIFACT_FOLDER_PATH}" --eval_gene_set "${EVAL_GENE_SET}"
```
"""

from __future__ import annotations

import argparse
import glob
import math
import pickle
import re
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ablation_utils import (EVAL_GENE_SETS, cells_without_vocabulary_genes, count_detected_vocabulary_genes,
                            count_stored_zeros, get_code_version, get_gene_set_dataset_name,
                            get_gene_set_silver_file_paths, get_section_names, get_seq_len_cell,
                            get_shared_gene_provenance, get_tokenization_provenance, get_tokenized_dataset_path,
                            load_plan, provenance_differences, ties_at_cut)

# UNASSIGNED_LABEL of datasets/*.env until it is set from check 3's output (README step 5)
UNASSIGNED_LABEL_PLACEHOLDER = "PLACEHOLDER_SET_FROM_CHECK_SETUP"

# Keys that differ between the mouse runs and the reruns by design
IGNORED_KEYS = {"code_version",
                "data.dataset_name",
                "data.raw_data_folder_path",
                "data.token_dict_folder_path",
                "data.tokenized_data_folder_path",
                "data.n_special_values",
                "data.vocab_size",
                "seed",
                # Opt-in fixes (`training_options` and `tokenizer_options` of the plan; off in the mouse runs)
                "mask.pad_masks_to_batch_max",
                "data.stable_rank_ties",
                "data.rank_tie_break_norm_factor",
                "data.zero_expr_from_raw_counts",
                "data.seurat_loess_span",
                "data.aggregate_own_neighbors",
                # TERRA option that the v2 code did not have, on in the reruns (checked separately)
                "meta.zero_fully_masked_attention_rows",
                "state.folder_path",
                "state.load_checkpoint",
                "state.read_checkpoint",
                "state.read_timestamp",
                "state.checkpoint_freq",
                # TERRA: checkpoint names (terra-latest.pth.tar) and the SDPA kernel choice ('torch' = PyTorch's choice,
                # the behaviour of the v2 code)
                "state.write_tag",
                "meta.attention_backend",
                # Matched to the mouse runs in optimizer steps instead of epochs
                "optimization.epochs",
                "optimization.warmup"}
# RoPE keys (added on 21 Apr, after the plotted runs; unused with the 'segment' and 'coord' encodings) and the TERRA
# options that the v2 code did not have (all off in the base config)
IGNORED_KEY_PREFIXES = ("meta.rope_", "meta.laplacian_", "protein_init.", "special_token_correction.",
                        "optimization.vicreg_granularity", "data.pf_targets_file_path")
# Keys in which the replicates of a variant differ (check 1)
REPLICATE_KEYS = {"seed", "state.folder_path"}
# Ratio of the sections' median nearest-neighbour distances above which their coordinates are taken to be in different
# units (check 2; the mouse sections are both in um, about 8-15)
MAX_NN_DISTANCE_RATIO = 2.0
# Label values listed per key (check 3), and values that may mark unassigned cells (listed in addition)
TOP_VALUES = 10
UNASSIGNED_PATTERN = re.compile(r"unassign|unknown|undefined|unlabel|^(none|nan|na|n/a|other|-|)$", re.IGNORECASE)


def flatten(config: dict, prefix: str = "") -> dict:
    flat_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat_config.update(flatten(value, f"{prefix}{key}."))
        else:
            flat_config[f"{prefix}{key}"] = value
    return flat_config


def find_mouse_params(mouse_artifact_folder_path: str, timestamp: str) -> Path | None:
    for pattern in [f"{mouse_artifact_folder_path}/models/{timestamp}/model_config.yaml",
                    f"{mouse_artifact_folder_path}/*/{timestamp}/params.yaml"]:
        file_paths = sorted(glob.glob(pattern))
        if file_paths:
            return Path(file_paths[0])
    return None


def check_cell_neighborhood_layout(variant: str, timestamp: str, config: dict, mouse_config: dict) -> bool:
    """
    Check 1 for a cell-neighborhood arm, whose layout is inferred from the mouse run names (ablation_plan.yaml): the
    model must read the same index-cell and neighborhood tokens as the mouse run. seq_len_cell and seq_len_neighborhood
    must be equal. The datasets read the first seq_len_cell tokens as the index cell and the next seq_len_neighborhood
    as the neighborhood, so model_input_size may differ only if the mouse data put the index cell into exactly the
    first seq_len_cell tokens (the v2 split into n_neighs + 1 blocks, e.g. 704 / 11 = 64) and holds the tokens the model
    reads: these are then the same top-ranked genes of the aggregated neighborhood. n_segments does not change the
    cell-neighborhood batches (the collator output is the same for 11 and 2).
    """
    keys = ["data.seq_len_cell", "data.seq_len_neighborhood", "data.model_input_size", "data.n_segments"]
    layout = {key.split(".")[1]: (mouse_config.get(key, "<missing>"), config.get(key, "<missing>")) for key in keys}
    seq_len_cell, seq_len_neighborhood = config["data.seq_len_cell"], config["data.seq_len_neighborhood"]
    ok = all(layout[key][0] == layout[key][1] for key in ["seq_len_cell", "seq_len_neighborhood"])
    mouse_input_size, input_size = layout["model_input_size"]
    if ok and mouse_input_size != input_size:
        mouse_n_neighs = mouse_config.get("data.n_neighs", config["data.n_neighs"])
        ok = (isinstance(mouse_input_size, int) and mouse_input_size / (mouse_n_neighs + 1) == seq_len_cell
              and mouse_input_size >= seq_len_cell + seq_len_neighborhood)
    print(f"{variant} | mouse run {timestamp}: cell-neighborhood layout (mouse, rerun) {layout}: "
          f"{'same model input' if ok else 'DIFFERENT'}")
    if not ok:
        print(f"ERROR: {variant} does not read the tokens of mouse run {timestamp}.", end=" ")
        if layout["seq_len_cell"][0] == 64 and layout["seq_len_neighborhood"][0] == 640:
            print("The mouse cell-neighborhood runs were 64/640: in ablation_plan.yaml, set the tokenization to "
                  "`cell_neigh: {tokenizer: cell-neighborhood-tokenizer, model_input_size: 704, n_neighs: 10, "
                  "seq_len_cell: null}` and, in the overrides of cell_neigh_tokenizer_combined, "
                  "cell_neigh_tokenizer_rank and cell_neigh_tokenizer_counts, `data.seq_len_neighborhood: 640` (keep "
                  "data.seq_len_cell: 64); then regenerate the configs, tokenize (new dataset name) and rerun this "
                  "check.")
        else:
            print("Set the cell_neigh tokenization and the data.seq_len_cell / data.seq_len_neighborhood overrides of "
                  "the cell-neighborhood variants in ablation_plan.yaml to the mouse layout, regenerate the configs "
                  "and tokenize.")
    return ok


def check_config_parity(variants: dict, tokenizations: dict, manifest_df: pd.DataFrame,
                        mouse_artifact_folder_path: str) -> bool:
    print("== 1. Config parity with the mouse runs (differences other than data, run folder, seed and code version)")
    ok = True
    for variant, spec in variants.items():
        configs = {}
        for replicate, config_file_path in (manifest_df[manifest_df["variant"] == variant]
                                            .set_index("replicate")["config_file_path"].items()):
            with open(config_file_path) as file:
                configs[replicate] = flatten(yaml.safe_load(file))
        config = configs[1]
        if config.get("meta.zero_fully_masked_attention_rows") is not True:
            print(f"ERROR: {variant}: meta.zero_fully_masked_attention_rows is "
                  f"{config.get('meta.zero_fully_masked_attention_rows', '<missing>')}, not true: set it in the base "
                  f"config and regenerate the configs")
            ok = False
        # Replicates differ only in their seed and run folder
        for replicate, replicate_config in configs.items():
            differences = sorted(key for key in set(config) | set(replicate_config)
                                 if key not in REPLICATE_KEYS and config.get(key) != replicate_config.get(key))
            if differences:
                print(f"ERROR: {variant}: replicate {replicate} differs from replicate 1 in {differences}")
                ok = False
        if len(configs) > len(spec["mouse_runs"]):
            print(f"{variant}: {len(configs)} replicates, {len(spec['mouse_runs'])} mouse runs; replicate 1 is "
                  f"compared")
        cell_neighborhood = tokenizations[spec["tokenization"]]["tokenizer"] == "cell-neighborhood-tokenizer"
        for timestamp in spec["mouse_runs"]:
            mouse_params_file_path = find_mouse_params(mouse_artifact_folder_path, timestamp)
            if mouse_params_file_path is None:
                print(f"{variant} | mouse run {timestamp}: params not found")
                if cell_neighborhood:
                    print(f"ERROR: {variant}: the inferred cell-neighborhood layout cannot be checked without the "
                          f"params of mouse run {timestamp}")
                    ok = False
                continue
            with open(mouse_params_file_path) as file:
                mouse_config = flatten(yaml.safe_load(file))
            differences = {key: (mouse_config.get(key, "<missing>"), config.get(key, "<missing>"))
                           for key in sorted(set(mouse_config) | set(config))
                           if key not in IGNORED_KEYS and not key.startswith(IGNORED_KEY_PREFIXES)
                           and mouse_config.get(key) != config.get(key)}
            print(f"{variant} | mouse run {timestamp}: "
                  f"{'identical' if not differences else f'mouse vs rerun {differences}'}")
            if cell_neighborhood:
                ok &= check_cell_neighborhood_layout(variant, timestamp, config, mouse_config)
    return ok


def check_pinned_code(code_path: str, config_file_path: str) -> bool:
    print("== 0. Pinned code and campaign")
    import inspect

    sys.path.insert(0, f"{code_path}/src")
    import terra
    from terra.inference.embed import embed_dataset
    from terra.tokenizers.cell_tokenizers import CellBaseTokenizer, CellNeighborhoodTokenizer

    version_file_path = Path(code_path) / "CODE_VERSION"
    version = version_file_path.read_text().strip() if version_file_path.exists() else "?"
    # The pinned copy, with the options of the plan and the metatoken inference option of the evaluation
    pinned = terra.__file__.startswith(f"{Path(code_path).resolve()}/") or terra.__file__.startswith(f"{code_path}/")
    tokenizer_options = set(inspect.signature(CellBaseTokenizer.__init__).parameters)
    has_options = ({"stable_rank_ties", "rank_tie_break_norm_factor", "zero_expr_from_raw_counts",
                    "seurat_loess_span"} <= tokenizer_options
                   and "aggregate_own_neighbors" in inspect.signature(CellNeighborhoodTokenizer.__init__).parameters
                   and "keep_special_tokens" in inspect.signature(embed_dataset).parameters)
    print(f"{terra.__file__}: {version}; "
          f"{'pinned copy' if pinned else 'NOT THE PINNED COPY (check PYTHONPATH)'}, "
          f"{'with the options of the plan' if has_options else 'WITHOUT the options of the plan (rerun setup_code.sh)'}")
    # TERRA ignores config keys it does not read, so a pin without the option would train without it
    zero_rows = any("zero_fully_masked_attention_rows" in file_path.read_text()
                    for file_path in Path(terra.__file__).parent.rglob("*.py"))
    print(f"meta.zero_fully_masked_attention_rows (set in the configs): "
          f"{'supported' if zero_rows else 'NOT SUPPORTED by this TERRA version (pin one that has it, setup_code.sh)'}")
    with open(config_file_path) as file:
        config_version = yaml.safe_load(file).get("code_version")
    same_version = config_version == version
    print(f"Code version of the configs: {config_version} "
          f"({'same' if same_version else 'DIFFERENT: regenerate the configs or set up that code version'})")
    return pinned and has_options and zero_rows and same_version


def check_dataset_env(artifact_folder_path: str, expected: dict) -> bool:
    """dataset.env (generate_ablation_runs.py; read by the job array) holds the values of the arguments."""
    dataset_env_file_path = Path(artifact_folder_path) / "dataset.env"
    if not dataset_env_file_path.is_file():
        print(f"{dataset_env_file_path} is MISSING: run generate_ablation_runs.py")
        return False
    values = {}
    for line in dataset_env_file_path.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key] = " ".join(shlex.split(value))
    differences = {key: (values.get(key, "<missing>"), value) for key, value in expected.items()
                   if values.get(key) != value}
    if differences:
        print(f"{dataset_env_file_path} (read by the job array) is DIFFERENT (dataset.env vs arguments): "
              f"{differences}; regenerate the configs with generate_ablation_runs.py")
    else:
        print(f"{dataset_env_file_path} (read by the job array): same values as the arguments")
    return not differences


def spv_token_offset(token_dict: dict) -> int:
    """
    Token id offset of the special values in the token dictionary (token id = special value + offset), as the
    tokenizers store them: the first special value token - 2 - the number of cls tokens.
    """
    spv_token_ids = [token_id for key, token_id in token_dict.items() if str(key).startswith("spv_")]
    n_cls_tokens = sum(1 for key in token_dict if "cls" in str(key))
    return min(spv_token_ids) - 2 - n_cls_tokens


def hard_coded_batch_token_ids() -> list[int]:
    """Token ids that TERRA's datasets hard-code for the batch metatoken (cell_datasets.py: `item['batch_token']`)."""
    import inspect
    from terra.datasets import cell_datasets

    return sorted({int(token_id) for token_id in re.findall(r"\['batch_token'\]\s*=\s*torch\.tensor\(\[(\d+)\]\)",
                                                             inspect.getsource(cell_datasets))})


def median_nearest_neighbor_distance(adata) -> float:
    """Median distance of each cell to its nearest other cell in obsm['spatial'] (the coordinates of the kNN graph)."""
    from sklearn.neighbors import NearestNeighbors

    coordinates = np.asarray(adata.obsm["spatial"], dtype=float)
    distances, _ = NearestNeighbors().fit(coordinates).kneighbors(n_neighbors=1)
    return float(np.median(distances))


def check_special_values(config_file_path: Path, token_dictionary_file_path: str, silver_file_paths: list,
                         mouse_token_dictionary_file_path: str, has_rank_variant: bool) -> bool:
    print("== 2. Batch special values and vocabulary genes")
    import scanpy as sc
    from terra.datasets.cell_datasets import rank_special_value_offset

    with open(token_dictionary_file_path, "rb") as file:
        token_dict = pickle.load(file)
    spv_offset = spv_token_offset(token_dict)
    # TERRA maps the special values of `gt_type: rank` models to tokens with an offset hard-coded by vocabulary size
    rank_offset = rank_special_value_offset(len(token_dict))
    with open(config_file_path) as file:
        data_config = yaml.safe_load(file)["data"]
    n_special_values = data_config["n_special_values"]
    # TERRA builds the models from the token dictionary; the configs record its size
    vocab_size_ok = data_config.get("vocab_size") == len(token_dict)
    print(f"vocab_size of the configs {data_config.get('vocab_size')}, token dictionary {len(token_dict)} tokens: "
          f"{'same' if vocab_size_ok else 'DIFFERENT (regenerate the configs with this token dictionary)'}")
    # The batch metatoken's token id is hard-coded in TERRA's datasets, not read from the token dictionary: a gene's id
    # would make the metatoken share that gene's token embedding
    batch_token_ids = hard_coded_batch_token_ids()
    batch_token_ok = all(token_id == token_dict.get("spt_batch") for token_id in batch_token_ids)
    print(f"Batch metatoken: token id {batch_token_ids or 'not hard-coded'} in TERRA's datasets, 'spt_batch' "
          f"{token_dict.get('spt_batch', '<missing>')} in the token dictionary: "
          f"{'OK' if batch_token_ok else 'DIFFERENT'}")
    if not batch_token_ok:
        print("ERROR: the batch metatoken would train with the token embedding of another token; make TERRA read the "
              "metatoken's token id from the token dictionary before launching.")
    fits = True
    n_empty_cells = 0
    n_stored_zeros = 0
    batch_values = []
    nn_distances = []
    for silver_file_path in silver_file_paths:
        adata = sc.read_h5ad(silver_file_path, backed="r")
        uns = adata.uns
        # Cells without a detected vocabulary gene, among those the tokenizer keeps (obs['filter_pass'], filters.py)
        empty = cells_without_vocabulary_genes(adata, token_dict)
        if "filter_pass" in adata.obs.columns:
            empty &= np.array([filter_pass == 1 for filter_pass in adata.obs["filter_pass"]])
        n_empty_cells += int(empty.sum())
        print(f"{silver_file_path}: {int(empty.sum())} of {adata.n_obs} cells without a detected gene of the token "
              f"dictionary{' (OK)' if not empty.any() else ', e.g. ' + str(list(adata.obs_names[empty][:3]))}")
        n_file_stored_zeros = count_stored_zeros(adata)
        n_stored_zeros += n_file_stored_zeros
        print(f"{silver_file_path}: {n_file_stored_zeros} explicitly stored zeros in X"
              f"{' (OK)' if n_file_stored_zeros == 0 else ''}")
        nn_distances.append(median_nearest_neighbor_distance(adata))
        print(f"{silver_file_path}: median nearest-neighbour distance in obsm['spatial'] {nn_distances[-1]:.4g}")
        batch_value_token = token_dict[f"spv_{uns['dataset_id']}_{uns['batch']}"]
        batch_value = batch_value_token - spv_offset
        batch_values.append(batch_value)
        fits &= batch_value <= n_special_values + 1
        print(f"{silver_file_path}: batch_value {batch_value}, special value embedding rows {n_special_values + 2}: "
              f"{'OK' if batch_value <= n_special_values + 1 else 'TOO SMALL'}")
    distinct = len(set(batch_values)) == len(batch_values)
    if not distinct:
        print(f"ERROR: silver files share a batch special value ({batch_values}), so the metatoken does not tell their "
              f"sections apart; check uns['dataset_id'] and uns['batch'] of the silver files.")
    # Sections in the same unit (um) have similar cell spacings; the coord arm feeds raw coordinate differences
    scales_ok = max(nn_distances) <= MAX_NN_DISTANCE_RATIO * min(nn_distances)
    if not scales_ok:
        print(f"ERROR: the median nearest-neighbour distances of the sections "
              f"({', '.join(f'{distance:.4g}' for distance in nn_distances)}) differ by more than "
              f"{MAX_NN_DISTANCE_RATIO:g}x: check that obsm['spatial'] of every section is in um.")
    if n_stored_zeros > 0:
        print(f"ERROR: the silver X stores {n_stored_zeros} explicit zeros, which the sparse ranking path turns into "
              f"zero-count gene tokens. Remove them with filter_silver_cells.py (README.md) and use the filtered files.")
    print(f"`gt_type: rank` metatoken: the token dictionary's special value offset is {spv_offset}, TERRA's (hard-coded "
          f"for {len(token_dict)} tokens) {rank_offset}: {'same' if rank_offset == spv_offset else 'DIFFERENT'}")
    rank_ok = rank_offset == spv_offset or not has_rank_variant
    if not rank_ok:
        print("ERROR: the rank arm would train with a wrong metatoken token; derive TERRA's rank special value offset "
              "from the token dictionary (cell_datasets.rank_special_value_offset) before launching.")

    # Mouse dictionary: did the mouse rank arms (v2 code, same hard-coded offsets) get the right metatoken?
    if Path(mouse_token_dictionary_file_path).exists():
        with open(mouse_token_dictionary_file_path, "rb") as file:
            mouse_token_dict = pickle.load(file)
        mouse_offset = spv_token_offset(mouse_token_dict)
        mouse_rank_offset = rank_special_value_offset(len(mouse_token_dict))
        print(f"Mouse token dictionary ({len(mouse_token_dict)} tokens): offset {mouse_offset}, hard-coded "
              f"{mouse_rank_offset} -> the mouse rank arms trained with "
              f"{'the right' if mouse_offset == mouse_rank_offset else 'a WRONG'} metatoken")
    else:
        print(f"Mouse token dictionary not found ({mouse_token_dictionary_file_path}).")
    if n_empty_cells > 0:
        print(f"ERROR: {n_empty_cells} cells have no detected gene of the token dictionary, so their ranked gene lists "
              f"are empty; on the sparse ranking path the cells around them get shifted segments and lose their "
              f"farthest neighbors. Remove them with filter_silver_cells.py (README.md) and use the filtered files.")
    return (fits and rank_ok and distinct and n_empty_cells == 0 and n_stored_zeros == 0 and vocab_size_ok
            and batch_token_ok and scales_ok)


def check_labels(label_file_paths: list, label_keys: list, unassigned_label: str) -> tuple[bool, pd.Series]:
    """Check 3; returns whether it passed and the section of every cell id of the label files."""
    print("== 3. Labels and sections (one label file per section; the sections are the batches of the integration "
          "metrics)")
    import scanpy as sc

    ok = True
    cell_sections = []
    files_with_key = {key: [] for key in label_keys}
    n_unassigned = 0
    for file_path, section_name in zip(label_file_paths, get_section_names(label_file_paths)):
        if not Path(file_path).is_file():
            print(f"{file_path}: MISSING")
            ok = False
            continue
        adata = sc.read_h5ad(file_path, backed="r")
        obs, uns = adata.obs, adata.uns
        keys = [key for key in label_keys if key in obs.columns]
        print(f"{file_path}: section {section_name}; {adata.n_obs} cells; uns dataset_id "
              f"{uns.get('dataset_id', '<missing>')}, batch {uns.get('batch', '<missing>')}; label keys {keys}")
        if "cell_id" in obs.columns:
            cell_sections.append(pd.Series(section_name, index=obs["cell_id"].astype(str).to_numpy()))
        else:
            print("  ERROR: no obs['cell_id'] (the tokenizer cell ids)")
            ok = False
        for key in keys:
            files_with_key[key].append(file_path)
            # Values as the evaluation compares them (strings; missing values are unlabelled)
            labelled = obs[key].notna().to_numpy()
            counts = pd.Series(obs[key].astype(str).to_numpy()[labelled]).value_counts()
            n_key_unassigned = int(counts.get(unassigned_label, 0))
            n_unassigned += n_key_unassigned
            print(f"  {key}: {len(counts)} values, {int((~labelled).sum())} cells without a value; top "
                  f"{min(TOP_VALUES, len(counts))}: "
                  f"{', '.join(f'{value} ({count})' for value, count in counts.head(TOP_VALUES).items())}")
            candidates = [f"{value} ({count})" for value, count in counts.items() if UNASSIGNED_PATTERN.search(value)]
            if candidates:
                print(f"    values that may mark unassigned cells: {', '.join(candidates)}")
            print(f"    UNASSIGNED_LABEL {unassigned_label!r}: {n_key_unassigned} cells")
    for key, file_paths in files_with_key.items():
        if file_paths:
            print(f"Label key {key}: in {len(file_paths)} of {len(label_file_paths)} label files")
        else:
            print(f"ERROR: label key {key} is in none of the label files")
            ok = False
    if unassigned_label == UNASSIGNED_LABEL_PLACEHOLDER:
        print(f"ERROR: UNASSIGNED_LABEL is still the placeholder {unassigned_label!r}: set it in the env file to the "
              f"unassigned value above, or to a value that occurs in no label file to score every labelled cell (as in "
              f"the mouse evaluation), then regenerate the configs.")
        ok = False
    elif n_unassigned == 0:
        print(f"WARNING: UNASSIGNED_LABEL {unassigned_label!r} occurs in no label file, so every labelled cell is "
              f"scored (as in the mouse evaluation); set it to the unassigned value above if there is one.")
    cell_section = pd.concat(cell_sections) if cell_sections else pd.Series(dtype=object)
    duplicated = cell_section.index[cell_section.index.duplicated()].unique()
    if len(duplicated) > 0:
        print(f"ERROR: {len(duplicated)} cell ids occur more than once in the label files, e.g. "
              f"{list(duplicated[:3])}")
        ok = False
        cell_section = cell_section[~cell_section.index.duplicated()]
    else:
        print(f"Cell ids: {len(cell_section)}, unique over the label files")
    if len(label_file_paths) >= 2:
        try:
            import scib_metrics
            print(f"{len(label_file_paths)} sections: the evaluation computes the batch-integration metrics (iLISI, "
                  f"ASW, MMD) between them (scib_metrics {scib_metrics.__version__})")
        except ImportError as error:
            print(f"ERROR: {len(label_file_paths)} sections, but scib_metrics (batch-integration metrics) does not "
                  f"import: {error}")
            ok = False
    else:
        print("1 section: no batch-integration metrics")
    return ok, cell_section


def report_schedule_end_values(config_file_paths: dict):
    print("== 6. Schedule end values (schedules span ipe_scale x epochs; training stops after epochs)")
    schedules = {}
    for variant, config_file_path in config_file_paths.items():
        with open(config_file_path) as file:
            optimization = yaml.safe_load(file)["optimization"]
        schedules.setdefault(yaml.safe_dump(optimization, sort_keys=True), []).append(variant)
    for optimization, variants in schedules.items():
        opt = yaml.safe_load(optimization)
        epochs, warmup, scale = opt["epochs"], opt["warmup"], opt["ipe_scale"]
        # Progress of WarmupCosineSchedule (after warmup), CosineWDSchedule and the EMA schedule at the last step
        lr_end = opt["final_lr"] + (opt["lr"] - opt["final_lr"]) * 0.5 * (
            1 + math.cos(math.pi * (epochs - warmup) / (scale * epochs - warmup)))
        wd_end = opt["final_weight_decay"] + (opt["weight_decay"] - opt["final_weight_decay"]) * 0.5 * (
            1 + math.cos(math.pi / scale))
        ema_end = opt["ema"][0] + (opt["ema"][1] - opt["ema"][0]) / scale
        print(f"{len(variants)} variants (e.g. {variants[0]}), {epochs} epochs, ipe_scale {scale}: "
              f"LR {opt['lr']:g} -> {lr_end:.3g} (configured {opt['final_lr']:g}); "
              f"WD {opt['weight_decay']:g} -> {wd_end:.3g} (configured {opt['final_weight_decay']:g}); "
              f"EMA {opt['ema'][0]:g} -> {ema_end:.4g} (configured {opt['ema'][1]:g})")


def check_gene_ids(norm_factor_file_path: str, silver_file_paths: list) -> bool:
    """
    Check 5 (gene ids), before the factor checks: the tokenizer looks up the factors of every feature of the silver
    files by its var['ensembl_id'] in the factor file as pandas reads it (normalize_by_factor), and fails late
    (IndexError in the factor-based tokenizations) on a feature without an Ensembl id, which compute_norm_factors.py
    writes as 'nan' and pandas reads back as NaN. A gene without counts in any section has no factors.
    """
    import scanpy as sc

    norm_factor_df = pd.read_csv(norm_factor_file_path)
    factor_columns = [column for column in ["mean", "nonzero_mean", "gene_corrected_read_depth_mean",
                                            "gene_corrected_read_depth_nonzero_mean"] if column in norm_factor_df]
    ok = True
    n_rows_without_id = int(norm_factor_df["gene_id"].isna().sum())
    if n_rows_without_id:
        print(f"ERROR: {n_rows_without_id} rows of {norm_factor_file_path} have no gene id (features of the silver "
              f"files without an Ensembl id, written as 'nan')")
        ok = False
    factors = norm_factor_df.dropna(subset=["gene_id"]).drop_duplicates("gene_id").set_index("gene_id")[factor_columns]
    total_counts = {}
    for silver_file_path in silver_file_paths:
        adata = sc.read_h5ad(silver_file_path, backed="r")
        gene_ids = adata.var["ensembl_id"]
        without_id = (gene_ids.isna()
                      | gene_ids.astype(str).str.strip().str.lower().isin(["", "nan", "none"])).to_numpy()
        if without_id.any():
            print(f"ERROR: {silver_file_path}: {int(without_id.sum())} features without an Ensembl id, e.g. "
                  f"{list(adata.var_names[without_id][:3])}; the factor-based tokenizations would fail on them: remove "
                  f"them from the silver files")
            ok = False
        counts = np.zeros(adata.n_vars)
        for start in range(0, adata.n_obs, 100_000):
            counts += np.asarray(adata.X[start:start + 100_000].sum(axis=0)).ravel()
        for gene_id, count in zip(gene_ids[~without_id].astype(str), counts[~without_id]):
            total_counts[gene_id] = total_counts.get(gene_id, 0.0) + count
    silver_gene_ids = pd.Index(sorted(total_counts))
    missing = silver_gene_ids.difference(factors.index)
    with_factors = silver_gene_ids.intersection(factors.index)
    values = factors.loc[with_factors].to_numpy(dtype=float)
    invalid = with_factors[~(np.isfinite(values) & (values > 0)).all(axis=1)]
    without_counts = [gene_id for gene_id in silver_gene_ids if total_counts[gene_id] == 0]
    print(f"Gene ids: {len(silver_gene_ids)} genes with an Ensembl id in the silver files; {len(missing)} without a "
          f"row of the factor file, {len(invalid)} with a non-finite or non-positive factor, {len(without_counts)} "
          f"without counts in any section{' (OK)' if not (len(missing) or len(invalid) or without_counts) else ''}")
    for name, genes in [("without a row of the factor file", missing), ("with invalid factors", invalid),
                        ("without counts in any section (no factors; remove them from the silver files)",
                         without_counts)]:
        if len(genes):
            print(f"ERROR: genes {name}, e.g. {list(genes[:3])}")
            ok = False
    return ok


def report_few_gene_cells(variants: dict, config_file_paths: dict, token_dictionary_file_path: str,
                          silver_file_paths: list):
    """Check 7: per section, the share of cells that index-cell-only batches leave without context gene, per ratio."""
    print("== 7. Cells with few detected genes (no context gene in index-cell-only batches when ceil(r * K) = K)")
    import scanpy as sc

    with open(token_dictionary_file_path, "rb") as file:
        token_dict = pickle.load(file)
    mask_ratios = set()
    for variant in variants:
        with open(config_file_paths[variant]) as file:
            mask_ratios.add(float(yaml.safe_load(file)["mask"]["per_block_mask_ratio"]))
    # Largest K whose targets (BlockMaskCollator: min(int(K * r + 0.9999), K)) take every gene token, per ratio
    max_genes = {ratio: max(K for K in range(1, 1000) if min(int(K * ratio + 0.9999), K) == K)
                 for ratio in sorted(mask_ratios)}
    thresholds = sorted(set(max_genes.values()))
    for silver_file_path in silver_file_paths:
        n_detected = count_detected_vocabulary_genes(sc.read_h5ad(silver_file_path, backed="r"), token_dict)
        shares = ", ".join(f"<= {K} genes (ratio {', '.join(f'{r:g}' for r, k in max_genes.items() if k == K)}): "
                           f"{int((n_detected <= K).sum())} ({(n_detected <= K).mean():.2%})" for K in thresholds)
        print(f"{silver_file_path}: {len(n_detected)} cells; {shares}")


def check_rank_tie_break(tokenizations: dict, norm_factor_file_path: str, token_dictionary_file_path: str,
                         silver_file_paths: list) -> bool:
    """Check 5 (rank ties): the tie-break column covers the vocabulary genes; report how often the tie order matters."""
    import scanpy as sc

    with open(token_dictionary_file_path, "rb") as file:
        token_dict = pickle.load(file)
    adatas = [sc.read_h5ad(silver_file_path, backed="r") for silver_file_path in silver_file_paths]
    ok = True
    factors = sorted({tokenization["rank_tie_break_norm_factor"] for tokenization in tokenizations.values()} - {None})
    if factors:
        norm_factor_df = pd.read_csv(norm_factor_file_path).drop_duplicates("gene_id").set_index("gene_id")
        gene_ids = pd.Index(sorted({gene_id for adata in adatas for gene_id in adata.var["ensembl_id"].astype(str)
                                    if token_dict.get(gene_id, False)}))
        for factor in factors:
            if factor not in norm_factor_df.columns:
                print(f"ERROR: rank_tie_break_norm_factor {factor!r} is not a column of {norm_factor_file_path}")
                ok = False
                continue
            invalid = ~np.isfinite(norm_factor_df[factor].reindex(gene_ids).to_numpy(dtype=float))
            print(f"Rank ties ordered by {factor!r} (ascending), then token id: {len(gene_ids) - invalid.sum()} of "
                  f"{len(gene_ids)} vocabulary genes of the silver files have a value"
                  f"{' (OK)' if not invalid.any() else ', MISSING for e.g. ' + str(list(gene_ids[invalid][:3]))}")
            ok &= not invalid.any()
    else:
        print("Rank ties ordered by token id (no rank_tie_break_norm_factor)")
    seq_lens = sorted({get_seq_len_cell(tokenization) for tokenization in tokenizations.values()})
    n_cells = sum(adata.n_obs for adata in adatas)
    n_cut, n_tied = dict.fromkeys(seq_lens, 0), dict.fromkeys(seq_lens, 0)
    for adata in adatas:
        for k, (n_file_cut, n_file_tied) in ties_at_cut(adata, token_dict, seq_lens).items():
            n_cut[k] += n_file_cut
            n_tied[k] += n_file_tied
    for k in seq_lens:
        print(f"{k} genes per cell: {n_cut[k]} of {n_cells} cells ({n_cut[k] / n_cells:.1%}) have more detected "
              f"genes; a raw-count ranking cuts {n_tied[k]} of them ({n_tied[k] / n_cells:.1%} of all cells) inside "
              f"tied counts, where the tie order decides which genes are kept")
    return ok


def check_tokenized_datasets(tokenizations: dict, silver_file_paths: list, data_directory_path: str,
                             dataset_name: str, cell_section: pd.Series, code_path: str, norm_factor_file_path: str,
                             token_dictionary_file_path: str) -> bool:
    import scanpy as sc
    from datasets import load_from_disk

    n_cells = sum(sc.read_h5ad(silver_file_path, backed="r").n_obs for silver_file_path in silver_file_paths)
    code_version = get_code_version(code_path)
    md5_cache = {}
    complete = True
    for name, tokenization in tokenizations.items():
        dataset_path = get_tokenized_dataset_path(tokenization, data_directory_path, dataset_name)
        if not dataset_path.exists():
            print(f"{name}: missing ({dataset_path})")
            complete = False
            continue
        differences = provenance_differences(
            dataset_path, get_tokenization_provenance(tokenization, code_version, silver_file_paths,
                                                      norm_factor_file_path, token_dictionary_file_path, md5_cache))
        if differences:
            print(f"{name}: NOT made from the current inputs (differs in: {', '.join(differences)}); move or delete "
                  f"{dataset_path} and tokenize again")
            complete = False
            continue
        dataset = load_from_disk(str(dataset_path))
        # Cell ids as the evaluation joins them to the labels
        cell_ids = pd.Index([str(cell_id) for cell_id in dataset["cell_id"]])
        sections = cell_section.reindex(cell_ids)
        unlabelled = sections.isna().to_numpy()
        coverage = ("all in the label files" if not unlabelled.any() else
                    f"{unlabelled.sum()} NOT in the label files, e.g. {list(cell_ids[unlabelled][:3])}")
        print(f"{name}: {dataset.num_rows} rows ({'OK' if dataset.num_rows == n_cells else f'expected {n_cells}'}); "
              f"cells {coverage}; per section {sections.value_counts().to_dict()}")
        complete &= dataset.num_rows == n_cells and not unlabelled.any()
    return complete


def check_shared_gene_data(tokenizations: dict, silver_file_paths: list, data_directory_path: str,
                           dataset_name: str, cell_section: pd.Series, code_path: str, norm_factor_file_path: str,
                           token_dictionary_file_path: str) -> bool:
    """
    Check 4 for the shared-gene evaluation data: its silver files exist, are all the files of its folder (the tokenizer
    reads every h5ad file) and were made from the current silver files (filter_shared_genes.py), and its tokenized
    datasets pass the checks of the training datasets.
    """
    shared_dataset_name = get_gene_set_dataset_name(dataset_name, "shared_genes")
    shared_file_paths = get_gene_set_silver_file_paths(silver_file_paths, data_directory_path, dataset_name,
                                                       "shared_genes")
    shared_folder_path = Path(data_directory_path) / "silver" / shared_dataset_name
    print(f"Shared-gene evaluation data {shared_dataset_name}:")
    missing = [file_path for file_path in shared_file_paths if not Path(file_path).is_file()]
    if missing:
        print(f"ERROR: {missing} missing: run filter_shared_genes.py")
        return False
    other_files = sorted({str(file_path) for file_path in shared_folder_path.glob("**/*.h5ad")}
                         - set(shared_file_paths))
    if other_files:
        print(f"ERROR: {shared_folder_path} holds other h5ad files, which the tokenizer would also read: {other_files}")
        return False
    expected = get_shared_gene_provenance(silver_file_paths, None)
    differences = provenance_differences(shared_folder_path, {"source_silver_file_md5s":
                                                              expected["source_silver_file_md5s"]})
    if differences:
        print(f"ERROR: {shared_folder_path} was NOT made from the current silver files (differs in: "
              f"{', '.join(differences)}); rerun filter_shared_genes.py, then tokenize again")
        return False
    print(f"{shared_folder_path}: made from the current silver files")
    return check_tokenized_datasets(tokenizations, shared_file_paths, data_directory_path, shared_dataset_name,
                                    cell_section, code_path, norm_factor_file_path, token_dictionary_file_path)


def main(args):
    label_file_paths = args.label_file_paths or args.silver_file_paths
    tokenizations, variants = load_plan(args.plan_file_path)
    manifest_df = pd.read_csv(Path(args.artifact_folder_path) / "manifest.tsv", sep="\t")
    config_file_paths = (manifest_df[manifest_df["replicate"] == 1]
                         .set_index("variant")["config_file_path"].to_dict())
    ok = check_pinned_code(args.code_path, config_file_paths["base"])
    ok &= check_dataset_env(args.artifact_folder_path,
                            {"DATASET_NAME": args.dataset_name,
                             "SILVER_FILE_PATHS": " ".join(args.silver_file_paths),
                             "LABEL_FILE_PATHS": " ".join(label_file_paths),
                             "LABEL_KEYS": " ".join(args.label_keys),
                             "UNASSIGNED_LABEL": args.unassigned_label,
                             "NORM_FACTOR_FILE_PATH": args.norm_factor_file_path,
                             "CODE_PATH": args.code_path,
                             "EVAL_GENE_SET": args.eval_gene_set})
    ok &= check_config_parity(variants, tokenizations, manifest_df, args.mouse_artifact_folder_path)
    has_rank_variant = any(variant["overrides"].get("meta.gt_type") == "rank" for variant in variants.values())
    ok &= check_special_values(config_file_paths["base"], args.token_dictionary_file_path, args.silver_file_paths,
                               args.mouse_token_dictionary_file_path, has_rank_variant)
    labels_ok, cell_section = check_labels(label_file_paths, args.label_keys, args.unassigned_label)
    ok &= labels_ok
    if not args.skip_datasets:
        print("== 4. Tokenized datasets (made from the current inputs; one row per silver cell; every cell in the "
              "label files)")
        ok &= check_tokenized_datasets(tokenizations, args.silver_file_paths, args.data_directory_path,
                                       args.dataset_name, cell_section, args.code_path, args.norm_factor_file_path,
                                       args.token_dictionary_file_path)
        if args.eval_gene_set == "shared_genes":
            ok &= check_shared_gene_data(tokenizations, args.silver_file_paths, args.data_directory_path,
                                         args.dataset_name, cell_section, args.code_path, args.norm_factor_file_path,
                                         args.token_dictionary_file_path)
    print("== 5. Normalization factors")
    from norm_factors import check_norm_factors
    from terra.preprocessors.normalizers import normalize_by_gene_corrected_read_depth
    gene_ids_ok = check_gene_ids(args.norm_factor_file_path, args.silver_file_paths)
    # The factor checks look genes up by id, which fails on the errors above
    ok &= gene_ids_ok and check_norm_factors(args.norm_factor_file_path, args.silver_file_paths,
                                             normalize_by_gene_corrected_read_depth)
    ok &= check_rank_tie_break(tokenizations, args.norm_factor_file_path, args.token_dictionary_file_path,
                               args.silver_file_paths)
    report_schedule_end_values(config_file_paths)
    report_few_gene_cells(variants, config_file_paths, args.token_dictionary_file_path, args.silver_file_paths)
    if not ok:
        sys.exit("Setup check failed, see above.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--plan_file_path',
                        type=str,
                        required=True,
                        help='Ablation plan.')
    parser.add_argument('--artifact_folder_path',
                        type=str,
                        required=True,
                        help='Folder with the generated configs and manifest.tsv.')
    parser.add_argument('--data_directory_path',
                        type=str,
                        required=True,
                        help='Data directory with the gold subdirectory (tokenized datasets).')
    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        help='Name of the dataset (tokenized dataset names).')
    parser.add_argument('--silver_file_paths',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Silver AnnData files of the dataset.')
    parser.add_argument('--label_file_paths',
                        type=str,
                        nargs='+',
                        help='Label files of the evaluation, one per section (LABEL_FILE_PATHS; default: the silver '
                             'files).')
    parser.add_argument('--label_keys',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Keys of the niche labels in the obs of the label files.')
    parser.add_argument('--unassigned_label',
                        type=str,
                        required=True,
                        help='Label of cells outside annotated regions (excluded from the metrics).')
    parser.add_argument('--norm_factor_file_path',
                        type=str,
                        required=True,
                        help='Per-gene normalization factors of the dataset.')
    parser.add_argument('--token_dictionary_file_path',
                        type=str,
                        required=True,
                        help='Token dictionary of the dataset\'s species.')
    parser.add_argument('--code_path',
                        type=str,
                        required=True,
                        help='Pinned code (setup_code.sh).')
    parser.add_argument('--mouse_token_dictionary_file_path',
                        type=str,
                        required=True,
                        help='Mouse token dictionary (rank metatoken check of the mouse runs).')
    parser.add_argument('--mouse_artifact_folder_path',
                        type=str,
                        required=True,
                        help='Artifact folder of the mouse ablation runs (models/<timestamp>/model_config.yaml).')
    parser.add_argument('--eval_gene_set',
                        type=str,
                        choices=EVAL_GENE_SETS,
                        required=True,
                        help='Evaluation data (EVAL_GENE_SET); with shared_genes, check 4 also checks the shared-gene '
                             'data.')
    parser.add_argument('--skip_datasets',
                        action='store_true',
                        help='Skip the tokenized dataset checks (check 4; before tokenization).')
    args = parser.parse_args()

    main(args)
