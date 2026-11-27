#!/usr/bin/env python
"""
Train TERRA models on gene panels: the training arm of the gene panel robustness experiment (R3C2.1, E12).

The panels are those of a reference run of gene_panel_subsampling.py (default: the full panel and random_1000_r0 to
random_4000_r0, nested panels of one random draw). For each panel, the sections of the reference run's dataset that
the reference run does not evaluate are restricted to the panel's genes and tokenized exactly as the embed step
tokenizes the evaluation sections (TERRA's harmonize_adata and tokenize_adata, the reference model folder with
data.pad_empty_neighbors). A model with the recipe of a TERRA training config is trained from scratch on each panel
with TERRA's launcher, packaged as a model folder, and embedded with gene_panel_subsampling.py on the reference run's
cells and panels; compare_panel_runs.py then compares all runs. The full panel is trained with two seeds (default),
whose difference is the training-seed ceiling of every comparison between models.

Steps (submit_panel_training_data.sh and submit_panel_model_runs.sh submit them):
  configure     <training folder>/training_config.json; prints the number of panels to tokenize
  prepare_data  (CPU, one job per panel) <data root>/<training name>/<panel>/dataset: one row per training cell with at
                least --min_panel_genes detected panel genes (all cells stay in the spatial neighbor graph), and
                data_record.json (cells per section, genes, tokenizer settings, provenance)
  make_configs  for every model, a model config and a job config in the TERRA checkout
                (configs/model|job/<config folder>/<training name>_<model>.yaml): the recipe of --recipe_job_config
                (its CONFIG_FILE) with the panel's dataset, the reference model's tokenizer settings, the seed and
                the run folder <training folder>/runs/<model>; prints the commands that submit the training jobs
  package       (after training) <training folder>/models/<model>: model_config.yaml (the run's params.yaml),
                model_checkpoint.pt (encoder, predictor and target encoder of the last epoch), the reference model's
                token and gene dictionaries and panel_model.json
  status        what is done for every panel and model
  runs          LABEL=RUN_FOLDER items of the reference run and the embedding runs of the models (compare_panel_runs)

Example (farm; see README.md, "Models trained on a panel"):
  bash submit_panel_training_data.sh xhs1000_skin_panels
  python panel_training.py make_configs --training_folder_path .../panel_training/xhs1000_skin_panels
  cd /nfs/team361/sb75/terra && bash scripts/bsub_submit_tiger.sh configs/job/panel_training/<config>.yaml
  bash submit_panel_model_runs.sh xhs1000_skin_panels
  bash submit_panel_comparison.sh xhs1000_skin_panels
"""

import argparse
import copy
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gene_panel_subsampling as gps  # noqa: E402

TRAINING_VERSION = 1
DEFAULT_REFERENCE_RUN = '/nfs/team361/sb75/terra-reproducibility/artifacts/panel_robustness/xhs1000_3sections_terra96m'
DEFAULT_DATA_ROOT = '/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/panel_training'
DEFAULT_TERRA_DIR = '/nfs/team361/sb75/terra'
DEFAULT_RECIPE_JOB_CONFIG = 'configs/job/hst_corpus_110m/hst_corpus_110m_terra96m.yaml'
DEFAULT_PANELS = ['full', 'random_1000_r0', 'random_2000_r0', 'random_3000_r0', 'random_4000_r0']
# Data keys of the recipe that belong to its own corpus
DROP_DATA_KEYS = ('precomputed_split', 'precomputed_n_nonzero_tokens', 'precomputed_epoch_splits',
                  'precomputed_epoch_n_nonzero_tokens', 'val_batch_ids', 'test_batch_ids', 'pf_targets_file_path')
# Data keys of the recipe that configure the training data loader (all other data keys come from the reference model)
TRAINING_DATA_KEYS = ('batch_size', 'num_workers', 'pin_memory', 'use_sampler', 'sampling_strategy',
                      'mega_batch_mult_max', 'n_special_values', 'nz_spc', 'allow_heldout_in_split')
# Data keys that name files or the dataset (not tokenizer settings)
PATH_DATA_KEYS = ('dataset_name', 'tokenized_data_folder_path', 'raw_data_folder_path', 'token_dict_folder_path',
                  'norm_factor_file_path')
# Keys that TERRA dev needs to train the released TERRA-96M recipe as it was trained (set only where the recipe has
# none; the released model_config.yaml has none of them)
DEV_DEFAULTS = (('meta', 'target_enc_layer_norm', True), ('meta', 'attention_backend', 'torch'),
                ('data', 'n_special_values', 1000), ('data', 'nz_spc', False), ('data', 'use_sampler', True),
                ('data', 'sampling_strategy', None), ('mask', 'sample_segments', True))
NO_MID_EPOCH_CHECKPOINTS = 1_000_000_000
MODEL_FILES = ('token_dictionary.pkl', 'ensembl_dictionary.pkl', 'ensembl_dic.pkl', 'gene_count_dictionary.pkl',
               'norm_factors.csv', 'pf_targets.csv')

log = gps.log
read_json = gps.read_json
write_json = gps.write_json


# ----------------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------------

def config_file(training_folder) -> Path:
    return Path(training_folder) / 'training_config.json'


def load_training_config(training_folder) -> dict:
    path = config_file(training_folder)
    if not path.is_file():
        raise FileNotFoundError(f'{path} does not exist; run `panel_training.py configure` first.')
    return read_json(path)


def reference_config(tcfg: dict) -> dict:
    return gps.load_config(tcfg['reference_run_folder'])


def data_folder(tcfg: dict, panel: str) -> Path:
    return Path(tcfg['data_root']) / tcfg['name'] / panel


def run_folder(tcfg: dict, model: dict) -> Path:
    return Path(tcfg['training_folder_path']) / 'runs' / model['model_id']


def model_folder(tcfg: dict, model: dict) -> Path:
    return Path(tcfg['training_folder_path']) / 'models' / model['model_id']


def find_model(tcfg: dict, model_id: str) -> dict:
    for model in tcfg['models']:
        if model['model_id'] == model_id:
            return model
    raise ValueError(f"Unknown model '{model_id}'; the models are {[m['model_id'] for m in tcfg['models']]}.")


def section_name(path: Path) -> str:
    return path.stem[len('adata_'):] if path.stem.startswith('adata_') else path.stem


def dataset_sections(cfg: dict) -> dict:
    """Section name -> silver file of every section of the dataset."""
    folder = Path(cfg['silver_folder_path']) / cfg['dataset_name']
    files = sorted(folder.rglob('*.h5ad'))
    sections = {}
    for path in files:
        name = section_name(path)
        if name in sections:
            raise ValueError(f'Two silver files for section {name}: {sections[name]} and {path}.')
        sections[name] = path
    return sections


def cmd_configure(args):
    reference_folder = Path(args.reference_run_folder).resolve()
    cfg = gps.load_config(reference_folder)
    if not (reference_folder / 'prepare_done.json').is_file():
        raise FileNotFoundError(f'The reference run {reference_folder} has not finished its prepare step.')
    if not cfg['pad_empty_neighbors']:
        raise ValueError('The reference run must embed with pad_empty_neighbors, as the training data is tokenized.')
    panels_available = read_json(reference_folder / 'panels.json')['panels']
    unknown = [p for p in args.panels if p not in panels_available]
    if unknown:
        raise ValueError(f'Panels {unknown} are not panels of {reference_folder}: {sorted(panels_available)}.')
    sections = dataset_sections(cfg)
    if args.training_sections:
        training_sections = list(args.training_sections)
        missing = [s for s in training_sections if s not in sections]
        if missing:
            raise ValueError(f'No silver file for sections {missing} in {cfg["silver_folder_path"]}/'
                             f'{cfg["dataset_name"]}.')
    else:
        candidates = {s for s in sections if re.fullmatch(r'batch\d+', s)}
        ignored = sorted(set(sections) - candidates)
        if ignored:
            log(f'Silver files that are not batch<k> sections are not used for training: {ignored}.')
        training_sections = sorted(candidates - set(cfg['batches']), key=lambda s: (len(s), s))
    held_out = sorted(set(training_sections) & set(cfg['batches']))
    if held_out:
        raise ValueError(f'Sections {held_out} are evaluated by the reference run; they cannot be training sections.')
    if not training_sections:
        raise ValueError('No training section.')
    models = []
    for panel in dict.fromkeys(args.panels):
        seeds = list(args.seeds) + (list(args.full_extra_seeds) if panel == 'full' else [])
        for seed in dict.fromkeys(seeds):
            models.append({'model_id': f'{panel}_s{seed}', 'panel': panel, 'seed': int(seed)})
    tcfg = {
        'training_version': TRAINING_VERSION,
        'name': args.name or Path(args.training_folder_path).name,
        'training_folder_path': str(Path(args.training_folder_path).resolve()),
        'reference_run_folder': str(reference_folder),
        'reference_model_folder_path': cfg['model_folder_path'],
        'silver_folder_path': cfg['silver_folder_path'],
        'dataset_name': cfg['dataset_name'],
        'evaluation_sections': cfg['batches'],
        'training_sections': training_sections,
        'panels': list(dict.fromkeys(args.panels)),
        'models': models,
        'data_root': args.data_root,
        'min_panel_genes': args.min_panel_genes,
        'nproc': args.nproc,
    }
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', tcfg['name']):
        raise ValueError(f"The training name '{tcfg['name']}' may only contain letters, digits, '_', '.' and '-'.")
    path = config_file(tcfg['training_folder_path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = read_json(path)
        if existing != tcfg and not args.overwrite:
            changed = sorted(k for k in set(existing) | set(tcfg) if existing.get(k) != tcfg.get(k))
            raise ValueError(f'{path} exists with other parameters ({", ".join(changed)}); use another training '
                             'folder or --overwrite.')
    write_json(path, tcfg)
    log(f'Wrote {path}: {len(tcfg["panels"])} panels, {len(models)} models, {len(training_sections)} training '
        f'sections ({", ".join(training_sections)}), evaluation sections {", ".join(cfg["batches"])} held out.')
    print(len(tcfg['panels']))


# ----------------------------------------------------------------------------------------------------------------------
# Training data
# ----------------------------------------------------------------------------------------------------------------------

def load_training_section(path: Path, section: str):
    """All cells of a silver section, named <section>:<silver cell_id> (or :row<i>), with obs['section']."""
    import anndata as ad
    import pandas as pd

    adata = ad.read_h5ad(path)
    if 'cell_id' in adata.obs and adata.obs['cell_id'].astype(str).is_unique:
        names = adata.obs['cell_id'].astype(str).to_numpy()
    else:
        names = np.array([f'row{i}' for i in range(adata.n_obs)])
        adata.obs['cell_id'] = [f'{section}_{name}' for name in names]
    adata.obs['silver_name'] = adata.obs_names.astype(str)
    adata.obs_names = pd.Index([f'{section}:{name}' for name in names])
    adata.obs['section'] = section
    adata.uns = {k: v for k, v in adata.uns.items() if k in ('batch', 'dataset_id')}
    return adata


def tokenize_section(adata, genes: list, ref_cfg: dict, token_dict: dict, model_folder_path: Path, cache: Path,
                     nproc: int, min_panel_genes: int):
    """
    Tokenized dataset of one section restricted to the panel genes, as harmonize_tokenize_embed_pipeline tokenizes
    (harmonization of the whole section, then of each sample, then tokenize_adata per sample); rows of center cells
    with fewer than min_panel_genes detected vocabulary genes are dropped afterwards, so that every cell stays in the
    neighbor graph. Returns the dataset and per-section counts.
    """
    import scipy.sparse as sp
    from datasets import concatenate_datasets
    from terra.inference import harmonize_adata
    from terra.inference.tokenize import tokenize_adata

    mapping_path, occurrence_path = gps.gene_reference_files(ref_cfg)
    present = [g for g in genes if g in adata.var_names]
    adata = adata[:, present].copy()
    harmonize = dict(gene_mapping_dict_file_path=mapping_path, gene_occurrence_count_file_path=occurrence_path,
                     gene_occurrence_count_filter_value=ref_cfg['gene_occurrence_count_filter_value'],
                     min_cells_per_gene=0, min_genes_per_cell=0)
    harmonized = harmonize_adata(adata, **harmonize)
    sample_key = ref_cfg['sample_key']
    samples = harmonized.obs[sample_key].astype(str).unique().tolist() if sample_key in harmonized.obs else [None]
    n_neighs = int(gps.read_model_config(ref_cfg)['data']['n_neighs'])
    vocabulary = {key for key in token_dict if str(key).startswith('ENS')}
    parts, counts = [], {'n_cells': 0, 'n_rows': 0, 'n_dropped_few_genes': 0, 'n_dropped_qc': 0,
                         'n_genes_present': len(present), 'n_genes_tokenizable': 0}
    for sample in samples:
        part = harmonized if sample is None else harmonized[harmonized.obs[sample_key].astype(str) == sample]
        part = harmonize_adata(part, **harmonize)
        if part.n_obs <= n_neighs:
            log(f'WARNING: sample {sample} has {part.n_obs} cells (fewer than {n_neighs + 1}); left out.')
            continue
        dataset = tokenize_adata(adata=part, model_folder_path=str(model_folder_path),
                                 cache_directory_path=str(cache), nproc=nproc)
        # Rows are the cells that pass QC, in order
        passes = (part.obs['filter_pass'].to_numpy() == 1) if 'filter_pass' in part.obs else np.ones(part.n_obs, bool)
        expected = part.obs['cell_id'].astype(str).to_numpy()[passes]
        row_ids = np.asarray(dataset['cell_id'], dtype=str)
        if not np.array_equal(row_ids, expected):
            raise RuntimeError(f'Tokenized rows do not match the cells of sample {sample} '
                               f'({len(row_ids)} rows, {len(expected)} cells).')
        in_vocabulary = part.var['ensembl_id'].isin(vocabulary).to_numpy()
        counts['n_genes_tokenizable'] = max(counts['n_genes_tokenizable'], int(in_vocabulary.sum()))
        X = sp.csr_matrix(part.X)[:, np.flatnonzero(in_vocabulary)]
        n_detected = np.asarray((X > 0).sum(axis=1)).ravel()[passes]
        keep = np.flatnonzero(n_detected >= min_panel_genes)
        counts['n_cells'] += int(part.n_obs)
        counts['n_dropped_qc'] += int((~passes).sum())
        counts['n_dropped_few_genes'] += int(len(row_ids) - len(keep))
        counts['n_rows'] += int(len(keep))
        parts.append(dataset.select(keep))
    if not parts:
        return None, counts
    return (parts[0] if len(parts) == 1 else concatenate_datasets(parts)), counts


def cmd_prepare_data(args):
    import pickle

    import yaml
    from datasets import concatenate_datasets, load_from_disk

    tcfg = load_training_config(args.training_folder_path)
    if not 1 <= args.panel_index <= len(tcfg['panels']):
        raise ValueError(f'--panel_index must be 1 to {len(tcfg["panels"])}.')
    panel = tcfg['panels'][args.panel_index - 1]
    out = data_folder(tcfg, panel)
    record_path = out / 'data_record.json'
    if record_path.is_file() and not args.force:
        log(f'{record_path} exists; the data of {panel} is done (use --force to tokenize again).')
        return
    ref_cfg = reference_config(tcfg)
    genes = read_json(Path(tcfg['reference_run_folder']) / 'panels.json')['panels'][panel]['genes']
    model_folder_path = gps.embedding_model_folder(ref_cfg)
    with open(model_folder_path / 'model_config.yaml') as f:
        tokenizer_config = yaml.safe_load(f)['data']
    if not tokenizer_config.get('pad_empty_neighbors'):
        raise ValueError(f'{model_folder_path} does not set data.pad_empty_neighbors; rerun the reference prepare step.')
    with open(Path(ref_cfg['model_folder_path']) / 'token_dictionary.pkl', 'rb') as f:
        token_dict = pickle.load(f)
    sections = dataset_sections(tcfg)
    parts_folder = out / 'parts'
    cache_root = out / 'cache'
    if args.force and out.exists():
        for name in ('dataset', 'parts', 'cache'):
            shutil.rmtree(out / name, ignore_errors=True)
        record_path.unlink(missing_ok=True)
    parts_folder.mkdir(parents=True, exist_ok=True)
    log(f'Panel {panel}: {len(genes)} genes, {len(tcfg["training_sections"])} training sections -> {out}.')
    section_counts = {}
    for section in tcfg['training_sections']:
        part_path = parts_folder / section
        counts_path = parts_folder / f'{section}.json'
        if counts_path.is_file() and part_path.is_dir():
            section_counts[section] = read_json(counts_path)
            log(f'{section}: done before ({section_counts[section]["n_rows"]} rows).')
            continue
        start = time.time()
        adata = load_training_section(sections[section], section)
        cache = cache_root / f'{section}-{os.environ.get("LSB_JOBID", os.getpid())}'
        shutil.rmtree(cache, ignore_errors=True)
        cache.mkdir(parents=True)
        try:
            dataset, counts = tokenize_section(adata, genes, ref_cfg, token_dict, model_folder_path, cache,
                                               tcfg['nproc'], tcfg['min_panel_genes'])
        finally:
            shutil.rmtree(cache, ignore_errors=True)
        counts['seconds'] = round(time.time() - start, 1)
        if dataset is not None:
            shutil.rmtree(part_path, ignore_errors=True)
            dataset.save_to_disk(str(part_path))
        else:
            part_path.mkdir(parents=True, exist_ok=True)
        write_json(counts_path, counts)
        section_counts[section] = counts
        log(f'{section}: {counts["n_cells"]} cells, {counts["n_rows"]} training rows ({counts["n_dropped_few_genes"]} '
            f'with fewer than {tcfg["min_panel_genes"]} panel genes, {counts["n_dropped_qc"]} failing QC), '
            f'{counts["n_genes_present"]} panel genes in the file ({counts["seconds"] / 60:.1f} min).')
    datasets = [load_from_disk(str(parts_folder / s)) for s in tcfg['training_sections']
                if section_counts[s]['n_rows'] > 0]
    dataset = concatenate_datasets(datasets)
    n_rows = sum(c['n_rows'] for c in section_counts.values())
    if len(dataset) != n_rows:
        raise RuntimeError(f'The concatenated dataset has {len(dataset)} rows, the sections {n_rows}.')
    tmp_path = out / f'.dataset.tmp-{os.environ.get("LSB_JOBID", os.getpid())}'
    shutil.rmtree(tmp_path, ignore_errors=True)
    dataset.save_to_disk(str(tmp_path), num_shards=max(1, min(64, n_rows // 50000)))
    shutil.rmtree(out / 'dataset', ignore_errors=True)
    os.replace(tmp_path, out / 'dataset')
    record = {
        'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'training_name': tcfg['name'], 'panel': panel,
        'n_panel_genes': len(genes), 'n_rows': n_rows, 'min_panel_genes': tcfg['min_panel_genes'],
        'n_cells': sum(c['n_cells'] for c in section_counts.values()),
        'sections': section_counts, 'evaluation_sections_held_out': tcfg['evaluation_sections'],
        'tokenizer_model_folder': str(model_folder_path), 'tokenizer_data_config': tokenizer_config,
        'reference_run_folder': tcfg['reference_run_folder'], **gps.terra_provenance()}
    write_json(record_path, record)
    shutil.rmtree(parts_folder, ignore_errors=True)
    shutil.rmtree(cache_root, ignore_errors=True)
    log(f'Wrote {out / "dataset"} ({n_rows} rows) and {record_path}.')


# ----------------------------------------------------------------------------------------------------------------------
# Training configs
# ----------------------------------------------------------------------------------------------------------------------

def expand(value: str, terra_dir: Path) -> str:
    return os.path.expandvars(str(value).replace('${TERRA_REPO_DIR}', str(terra_dir)))


def training_model_config(recipe: dict, reference: dict, tcfg: dict, model: dict, record: dict, args) -> tuple:
    """Model config of one model: the recipe, with the panel's data and the reference model's tokenizer settings."""
    config = copy.deepcopy(recipe)
    for section in ('data', 'meta', 'mask', 'optimization', 'state'):
        if section not in config:
            raise ValueError(f"The recipe has no '{section}' section; give a training config with "
                             '--recipe_model_config.')
    # Data section: the reference model's (the settings the training data was tokenized with, which the packaged model
    # also tokenizes new data with), plus the recipe's data loader keys. Tokenizer options of the recipe that the
    # reference does not set (e.g. stable_rank_ties of a newer corpus) must not reach the model config.
    data = copy.deepcopy(reference['data'])
    for key in TRAINING_DATA_KEYS:
        if key in recipe['data']:
            data[key] = copy.deepcopy(recipe['data'][key])
    for key in DROP_DATA_KEYS:
        data.pop(key, None)
    reference_folder = Path(tcfg['reference_model_folder_path'])
    data.update({
        'dataset_name': f'{tcfg["name"]}_{model["panel"]}',
        'tokenized_data_folder_path': str(data_folder(tcfg, model['panel']) / 'dataset'),
        'raw_data_folder_path': str(Path(tcfg['silver_folder_path']) / tcfg['dataset_name']),
        'token_dict_folder_path': str(reference_folder / 'token_dictionary.pkl'),
        'pad_empty_neighbors': True,
    })
    if (reference_folder / 'norm_factors.csv').is_file():
        data['norm_factor_file_path'] = str(reference_folder / 'norm_factors.csv')
    config['data'] = data
    # Every tokenizer setting must be the one the data was tokenized with
    tokenized_with = record['tokenizer_data_config']
    exempt = set(TRAINING_DATA_KEYS) | set(DROP_DATA_KEYS) | set(PATH_DATA_KEYS)
    for key in sorted((set(data) | set(tokenized_with)) - exempt):
        if data.get(key) != tokenized_with.get(key):
            raise ValueError(f'data.{key} would be {data.get(key)!r}, but the training data was tokenized with '
                             f'{tokenized_with.get(key)!r}.')
    added = []
    for section, key, value in DEV_DEFAULTS:
        if key not in config[section]:
            config[section][key] = value
            added.append(f'{section}.{key}: {value}')
    if args.batch_size:
        data['batch_size'] = args.batch_size
    if args.epochs:
        config['optimization']['epochs'] = args.epochs
    config['seed'] = model['seed']
    epochs = int(config['optimization']['epochs'])
    if epochs % args.checkpoint_freq:
        raise ValueError(f'{epochs} epochs are not a multiple of --checkpoint_freq {args.checkpoint_freq}: the last '
                         'epoch would have no terra-ep<N> checkpoint to package.')
    state = config['state']
    state['checkpoint_freq'] = args.checkpoint_freq      # terra-ep<N> after every checkpoint_freq epochs
    state.update({'folder_path': str(run_folder(tcfg, model)), 'load_checkpoint': False, 'read_checkpoint': None})
    # train() reads state.checkpoint_freq_iter as an integer: without mid-epoch checkpoints, a number of iterations
    # no epoch reaches
    state['checkpoint_freq_iter'] = args.checkpoint_freq_iter or NO_MID_EPOCH_CHECKPOINTS
    return config, added


def training_job_config(template: dict, tcfg: dict, model: dict, config_name: str, config_folder: str, args) -> dict:
    job = copy.deepcopy(template)
    requirement = job.setdefault('job_requirement', {})
    requirement.update({'LSB_DEFAULT_USERGROUP': args.group, 'QUEUE': args.queue, 'NUM_NODES': 1,
                        'NUM_GPUS_NODE': args.gpus, 'NUM_PROCESSES_NODE': args.cores, 'MEM_NODE': args.mem_gb})
    job.setdefault('infiniband', {})['NCCL_IB_DISABLE'] = 1     # one node
    artifact = job.setdefault('artifact', {})
    logs = str(Path(tcfg['training_folder_path']) / 'launcher')
    for key in ('ARTIFACT_LOCATION', 'WAREHOUSE_PATH', 'OUTPUT_DIR', 'LOG_DIR'):
        artifact[key] = logs
    artifact['EXPERIMENT_NAME'] = args.experiment_name
    artifact['RUN_NAME'] = config_name
    job.setdefault('runner_script', {})['CONFIG_FILE'] = (
        f'${{TERRA_REPO_DIR}}/configs/model/{config_folder}/{config_name}.yaml')
    return job


def cmd_make_configs(args):
    import yaml

    tcfg = load_training_config(args.training_folder_path)
    ref_cfg = reference_config(tcfg)
    terra_dir = Path(args.terra_dir or os.environ.get('TERRA_REPO_DIR') or DEFAULT_TERRA_DIR).resolve()
    recipe_job_path = Path(args.recipe_job_config or terra_dir / DEFAULT_RECIPE_JOB_CONFIG)
    if not recipe_job_path.is_file():
        raise FileNotFoundError(f'Recipe job config {recipe_job_path} not found; give --recipe_job_config (a TERRA '
                                'job config whose CONFIG_FILE is the training recipe).')
    with open(recipe_job_path) as f:
        template = yaml.safe_load(f)
    recipe_path = Path(args.recipe_model_config or expand(template['runner_script']['CONFIG_FILE'], terra_dir))
    with open(recipe_path) as f:
        recipe = yaml.safe_load(f)
    reference = gps.read_model_config(ref_cfg)
    models = [find_model(tcfg, m) for m in args.models] if args.models else tcfg['models']
    model_dir = terra_dir / 'configs' / 'model' / args.config_folder
    job_dir = terra_dir / 'configs' / 'job' / args.config_folder
    commands = []
    for model in models:
        record_path = data_folder(tcfg, model['panel']) / 'data_record.json'
        if not record_path.is_file():
            raise FileNotFoundError(f'The training data of {model["panel"]} is not ready ({record_path}); run '
                                    'prepare_data first.')
        record = read_json(record_path)
        config_name = f'{tcfg["name"]}_{model["model_id"]}'
        config, added = training_model_config(recipe, reference, tcfg, model, record, args)
        job = training_job_config(template, tcfg, model, config_name, args.config_folder, args)
        paths = (model_dir / f'{config_name}.yaml', job_dir / f'{config_name}.yaml')
        existing = [p for p in paths if p.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(f'{existing[0]} exists; use --overwrite to replace the configs.')
        for path, content in zip(paths, (config, job)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(f'# Written by terra-reproducibility analysis/panel_robustness/panel_training.py make_configs\n'
                        f'# Recipe: {recipe_path} (job: {recipe_job_path})\n# Training data: {record_path}\n')
                yaml.safe_dump(content, f, sort_keys=False)
        run_folder(tcfg, model).mkdir(parents=True, exist_ok=True)
        log(f'{model["model_id"]}: {record["n_rows"]} training rows; wrote {paths[0]} and {paths[1]}'
            + (f'; added {", ".join(added)}' if added else '') + '.')
        commands.append(f'bash scripts/{args.submit_script} configs/job/{args.config_folder}/{config_name}.yaml')
    write_json(Path(tcfg['training_folder_path']) / 'configs.json', {
        'written': time.strftime('%Y-%m-%d %H:%M:%S'), 'terra_dir': str(terra_dir), 'recipe_model_config':
        str(recipe_path), 'recipe_job_config': str(recipe_job_path), 'config_folder': args.config_folder,
        'models': [m['model_id'] for m in models], 'gpus': args.gpus, 'batch_size_per_gpu': args.batch_size,
        'epochs': args.epochs, 'queue': args.queue})
    print(f'cd {terra_dir} && source scripts/cluster_env.sh')
    for command in commands:
        print(command)


# ----------------------------------------------------------------------------------------------------------------------
# Packaging
# ----------------------------------------------------------------------------------------------------------------------

def finished_epochs(folder: Path) -> list:
    """Epoch numbers of the end-of-epoch checkpoints (terra-ep<N>.pth.tar) of a run folder."""
    epochs = []
    for path in folder.glob('*-ep*.pth.tar'):
        match = re.fullmatch(r'.+-ep(\d+)\.pth\.tar', path.name)
        if match:
            epochs.append((int(match.group(1)), path))
    return sorted(epochs)


def strip_prefix(state: dict) -> dict:
    return {key[len('module.'):] if key.startswith('module.') else key: value for key, value in state.items()}


def cmd_package(args):
    import torch
    import yaml

    tcfg = load_training_config(args.training_folder_path)
    models = [find_model(tcfg, m) for m in args.models] if args.models else tcfg['models']
    for model in models:
        source = run_folder(tcfg, model)
        target = model_folder(tcfg, model)
        params_path = source / 'params.yaml'
        epochs = finished_epochs(source)
        if not params_path.is_file() or not epochs:
            log(f'{model["model_id"]}: no finished epoch in {source}; skipped.')
            continue
        with open(params_path) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)  # written with yaml.dump by train()
        n_epochs = int(params['optimization']['epochs'])
        if args.epoch is not None:
            chosen = [e for e in epochs if e[0] == args.epoch]
            if not chosen:
                raise FileNotFoundError(f'{model["model_id"]}: no checkpoint of epoch {args.epoch} in {source}.')
            epoch, checkpoint_path = chosen[0]
        else:
            epoch, checkpoint_path = epochs[-1]
            if epoch < n_epochs - 1 and not args.allow_unfinished:
                log(f'{model["model_id"]}: training has reached epoch {epoch} of {n_epochs} (0-based last '
                    f'{n_epochs - 1}); skipped (--allow_unfinished packages it anyway).')
                continue
        marker = target / 'panel_model.json'
        if marker.is_file() and read_json(marker).get('checkpoint') == str(checkpoint_path) and not args.force:
            log(f'{model["model_id"]}: already packaged from {checkpoint_path.name}.')
            continue
        target.mkdir(parents=True, exist_ok=True)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        torch.save({'encoder': strip_prefix(checkpoint['encoder']), 'predictor': strip_prefix(checkpoint['predictor']),
                    'target_encoder': strip_prefix(checkpoint['target_encoder']), 'epoch': checkpoint.get('epoch'),
                    'loss': checkpoint.get('loss')}, target / 'model_checkpoint.pt')
        params['data']['pad_empty_neighbors'] = True
        with open(target / 'model_config.yaml', 'w') as f:
            yaml.safe_dump(params, f, sort_keys=False)
        reference_folder = Path(tcfg['reference_model_folder_path'])
        for name in MODEL_FILES:
            if (reference_folder / name).is_file():
                shutil.copy2(reference_folder / name, target / name)
        record = read_json(data_folder(tcfg, model['panel']) / 'data_record.json')
        write_json(marker, {
            'packaged': time.strftime('%Y-%m-%d %H:%M:%S'), 'training_name': tcfg['name'],
            'model_id': model['model_id'], 'panel': model['panel'], 'seed': model['seed'], 'epoch': epoch,
            'n_epochs': n_epochs, 'checkpoint': str(checkpoint_path), 'run_folder': str(source),
            'training_rows': record['n_rows'], 'n_panel_genes': record['n_panel_genes'],
            'training_sections': sorted(record['sections']), 'reference_run_folder': tcfg['reference_run_folder'],
            'reference_model_folder': tcfg['reference_model_folder_path']})
        log(f'{model["model_id"]}: packaged epoch {epoch} into {target}.')


# ----------------------------------------------------------------------------------------------------------------------
# Status and run list
# ----------------------------------------------------------------------------------------------------------------------

def embedding_run_folder(tcfg: dict, model: dict, artifacts: Path) -> Path:
    return artifacts / f'{tcfg["name"]}_{model["model_id"]}'


def cmd_status(args):
    tcfg = load_training_config(args.training_folder_path)
    artifacts = Path(args.artifacts or Path(tcfg['reference_run_folder']).parent)
    print(f'Training {tcfg["name"]}: {len(tcfg["training_sections"])} training sections, reference '
          f'{tcfg["reference_run_folder"]}')
    for panel in tcfg['panels']:
        record_path = data_folder(tcfg, panel) / 'data_record.json'
        state = f'{read_json(record_path)["n_rows"]} rows' if record_path.is_file() else 'not tokenized'
        print(f'  data {panel}: {state}')
    for model in tcfg['models']:
        epochs = finished_epochs(run_folder(tcfg, model))
        packaged = (model_folder(tcfg, model) / 'panel_model.json').is_file()
        embedding = embedding_run_folder(tcfg, model, artifacts)
        done = (embedding / 'results' / 'summary_done.json').is_file()
        print(f'  model {model["model_id"]}: last epoch {epochs[-1][0] if epochs else "-"}, '
              f'{"packaged" if packaged else "not packaged"}, embedding run {embedding.name}: '
              f'{"done" if done else "not done"}')


def cmd_list_packaged(args):
    """Tab-separated: model id, panel, model folder, reference run, for every packaged model."""
    tcfg = load_training_config(args.training_folder_path)
    for model in tcfg['models']:
        folder = model_folder(tcfg, model)
        if (folder / 'panel_model.json').is_file():
            print('\t'.join([model['model_id'], model['panel'], str(folder), tcfg['reference_run_folder']]))


def cmd_runs(args):
    tcfg = load_training_config(args.training_folder_path)
    artifacts = Path(args.artifacts or Path(tcfg['reference_run_folder']).parent)
    items = [f'{args.reference_label}={tcfg["reference_run_folder"]}']
    for model in tcfg['models']:
        folder = embedding_run_folder(tcfg, model, artifacts)
        if (folder / 'run_config.json').is_file():
            items.append(f'{model["model_id"]}={folder}')
    print(' '.join(items))


# ----------------------------------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------------------------------

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    configure = sub.add_parser('configure', help='Write the training configuration.')
    configure.add_argument('--training_folder_path', required=True)
    configure.add_argument('--name', default=None, help='Default: the training folder name.')
    configure.add_argument('--reference_run_folder', default=DEFAULT_REFERENCE_RUN,
                           help='gene_panel_subsampling.py run whose panels, model folder (tokenizer settings, '
                                'vocabulary, gene mapping) and dataset are used; its sections are held out.')
    configure.add_argument('--panels', nargs='+', default=DEFAULT_PANELS)
    configure.add_argument('--seeds', nargs='+', type=int, default=[0])
    configure.add_argument('--full_extra_seeds', nargs='*', type=int, default=[1],
                           help='More seeds for the full panel (training-seed ceiling).')
    configure.add_argument('--training_sections', nargs='+', default=None,
                           help='Default: every section of the dataset except the reference run\'s sections.')
    configure.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    configure.add_argument('--min_panel_genes', type=int, default=10,
                           help='Training rows need at least this many detected panel genes in the center cell '
                                '(the corpus rule of 10 vocabulary genes per cell, applied to the panel).')
    configure.add_argument('--nproc', type=int, default=8, help='Tokenization processes.')
    configure.add_argument('--overwrite', action='store_true')

    prepare = sub.add_parser('prepare_data', help='Tokenize the training data of one panel.')
    prepare.add_argument('--training_folder_path', required=True)
    prepare.add_argument('--panel_index', type=int, default=int(os.environ.get('LSB_JOBINDEX', '0') or 0),
                         help='1-based panel index (default: LSB_JOBINDEX).')
    prepare.add_argument('--force', action='store_true')

    configs = sub.add_parser('make_configs', help='Write the training configs into the TERRA checkout.')
    configs.add_argument('--training_folder_path', required=True)
    configs.add_argument('--models', nargs='+', default=None, help='Default: every model.')
    configs.add_argument('--terra_dir', default=None, help=f'Default: $TERRA_REPO_DIR or {DEFAULT_TERRA_DIR}.')
    configs.add_argument('--recipe_job_config', default=None,
                         help=f'TERRA job config of the recipe (default: <terra_dir>/{DEFAULT_RECIPE_JOB_CONFIG}); '
                              'its CONFIG_FILE is the recipe model config.')
    configs.add_argument('--recipe_model_config', default=None, help='Overrides the recipe job config\'s CONFIG_FILE.')
    configs.add_argument('--config_folder', default='panel_training',
                         help='Subfolder of configs/model and configs/job.')
    configs.add_argument('--epochs', type=int, default=None, help='Default: the recipe\'s.')
    configs.add_argument('--batch_size', type=int, default=144, help='Per GPU (TERRA-96M: 144).')
    configs.add_argument('--gpus', type=int, default=4)
    configs.add_argument('--cores', type=int, default=32)
    configs.add_argument('--mem_gb', type=int, default=256)
    configs.add_argument('--queue', default='training-parallel')
    configs.add_argument('--group', default='s10396')
    configs.add_argument('--experiment_name', default='terra_panel_training')
    configs.add_argument('--checkpoint_freq', type=int, default=5,
                         help='Epochs between terra-ep<N> checkpoints (the epochs must be a multiple; each '
                              'TERRA-96M-sized checkpoint is about 0.6 GB in the run folder).')
    configs.add_argument('--checkpoint_freq_iter', type=int, default=None,
                         help='Iterations between mid-epoch checkpoints (default: none; end-of-epoch checkpoints are '
                              'always written).')
    configs.add_argument('--submit_script', default='bsub_submit_tiger.sh',
                         help='TERRA launcher in scripts/ that the printed commands use.')
    configs.add_argument('--overwrite', action='store_true')

    package = sub.add_parser('package', help='Model folders of the trained models.')
    package.add_argument('--training_folder_path', required=True)
    package.add_argument('--models', nargs='+', default=None, help='Default: every model with a finished run.')
    package.add_argument('--epoch', type=int, default=None, help='Default: the last finished epoch.')
    package.add_argument('--allow_unfinished', action='store_true')
    package.add_argument('--force', action='store_true')

    listing = sub.add_parser('list_packaged', help='Packaged models (tab-separated).')
    listing.add_argument('--training_folder_path', required=True)

    for name, help_text in (('status', 'Show what is done.'), ('runs', 'LABEL=RUN_FOLDER items for comparison.')):
        command = sub.add_parser(name, help=help_text)
        command.add_argument('--training_folder_path', required=True)
        command.add_argument('--artifacts', default=None,
                             help='Folder of the embedding runs (default: the reference run\'s parent).')
        if name == 'runs':
            command.add_argument('--reference_label', default='terra96m')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    {'configure': cmd_configure, 'prepare_data': cmd_prepare_data, 'make_configs': cmd_make_configs,
     'package': cmd_package, 'status': cmd_status, 'runs': cmd_runs,
     'list_packaged': cmd_list_packaged}[args.command](args)


if __name__ == '__main__':
    main()
