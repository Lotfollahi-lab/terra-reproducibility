#!/usr/bin/env python
"""
TERRA training configs for a given number of GPUs: one per split variant of
a corpus built by the Slurm corpus build (make_splits.py, --variants), and
one per model trained on every cell of a single dataset (--parts), which
reads that dataset's tokenized part directly and needs no corpus or split.

Each config is the layout config, a committed TERRA model config that reads
the 16 x 512 compact corpus in the model's layout (default
configs/model/hst_corpus_110m/hst_corpus_110m_knn10_seq256.yaml: 10
neighbours x 256 genes per cell, as TERRA-96M reads cells), with:
- the recipe: --recipe terra96m (default) sets the model, masking and
  optimization of the released TERRA-96M (TERRA commit cc87839,
  configs/model/hst_corpus_110m/hst_corpus_110m_gtsmall_aws.yaml) on the keys
  current TERRA reads; --recipe layout keeps the layout config's own recipe;
- the data: for a variant, the corpus, training split, norm factors, token
  dictionary and held-out batch ids from <splits_dir>/<variant>/
  split_summary.json; for a part model, the dataset's tokenized part
  (<data_directory_path>/gold/cell-graph-tokenizer/<dataset>/...), the norm
  factors and token dictionary it was tokenized with, no split and no
  held-out batches;
- the batch: --batch_size per GPU on --gpus GPUs. Learning rate, EMA, weight
  decay and the mid-epoch checkpoint spacing are scaled from TERRA-96M's
  global batch of 4,608 by the rule of the hst_corpus_110m configs
  (r = global / 4608: lr x sqrt(r), 1 - ema x r, weight decay x sqrt(r),
  checkpoint_freq_iter / r); --no_scaling keeps the recipe's values;
- a new run folder (state.folder_path null; submit_training.sh sets it per
  run).

The configs are written as <out_dir>/<name>_<variant>.yaml, with a header
that records the sources, the GPU count they are made for (submit_training.sh
checks it) and the steps per epoch. Before writing, each config goes through
train()'s start-up check of the data section against the corpus record
(terra.tokenizers.params.reconcile_tokenizer_params), its training split is
loaded through TERRA (prepare_dataset), and the model is built on the CPU
(init_model_from_config); --no_check skips the three.

Needs the TERRA environment and TERRA_DATA_DIR in the environment: the
configs give their paths as ${TERRA_DATA_DIR}/..., which TERRA's config loader
expands.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

# The released TERRA-96M recipe as values of the keys current TERRA reads.
# Keys the release did not have get the value that reproduces its behaviour:
# attention_backend 'torch' and target_enc_layer_norm true restore cc87839's
# attention kernels and target normalisation; masks are not padded to the
# batch maximum and every cell may lose all its context genes, as then.
TERRA96M = {
    'data': {'batch_size': 144},
    'meta': {'count_encoding': 'mlp', 'n_value_bins': 100,
             'attention_backend': 'torch', 'target_enc_layer_norm': True,
             'zero_fully_masked_attention_rows': False},
    'mask': {'per_block_mask_ratio': 0.8, 'pad_masks_to_batch_max': False,
             'min_context_genes': 0},
    'optimization': {'warmup': 1, 'epochs': 5, 'ema': [0.9995, 1.0],
                     'start_lr': 1.0e-06, 'lr': 6.0e-05, 'final_lr': 1.0e-06,
                     'weight_decay': 0.04, 'final_weight_decay': 0.4,
                     'ipe_scale': 1.25, 'clip_grad': 2.0},
    'state': {'checkpoint_freq_iter': 5000},
}
# TERRA-96M: 144 per GPU on 32 GPUs, with these values
REFERENCE = {'global_batch': 4608, 'lr': 6.0e-05, 'ema': 0.9995,
             'weight_decay': 0.04, 'final_weight_decay': 0.4,
             'checkpoint_freq_iter': 5000}
# Data keys the variant's split summary sets (config_data of make_splits.py)
VARIANT_DATA_KEYS = ('tokenized_data_folder_path', 'precomputed_split',
                     'norm_factor_file_path', 'token_dict_folder_path',
                     'val_batch_ids', 'test_batch_ids')
# Belong to another corpus or split
DROP_DATA_KEYS = ('precomputed_n_nonzero_tokens', 'pf_targets_file_path')


class ConfigError(Exception):
    """An input the configs must not be written from."""


def log(message: str):
    print(message, flush=True)


def scaled(global_batch: int) -> dict:
    """Recipe values for `global_batch`, scaled from TERRA-96M's."""
    r = global_batch / REFERENCE['global_batch']
    return {
        'r': r,
        'lr': float(f"{REFERENCE['lr'] * math.sqrt(r):.3g}"),
        'ema': round(1 - (1 - REFERENCE['ema']) * r, 6),
        'weight_decay': float(f"{REFERENCE['weight_decay'] * math.sqrt(r):.3g}"),
        'final_weight_decay': float(
            f"{REFERENCE['final_weight_decay'] * math.sqrt(r):.3g}"),
        'checkpoint_freq_iter': max(1, round(
            REFERENCE['checkpoint_freq_iter'] / r)),
    }


def expand(obj):
    """${VAR} expansion as TERRA's config loader does it (terra.utils.config)."""
    if isinstance(obj, dict):
        return {k: expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def git_commit(path) -> str:
    try:
        return subprocess.run(
            ['git', '-C', str(Path(path).resolve().parent), 'log', '--oneline',
             '-1'], capture_output=True, text=True).stdout.strip() or '?'
    except OSError:
        return '?'


def silver_root(corpus_path: str) -> str:
    """<data root>/silver for a corpus at <data root>/gold/<tokenizer>/..."""
    head, sep, _ = corpus_path.partition('/gold/')
    return f'{head}/silver' if sep else str(Path(corpus_path).parent)


def with_data_dir(path, data_directory_path: Path) -> str:
    """`path` as ${TERRA_DATA_DIR}/... when below the data directory."""
    path = Path(path).resolve()
    try:
        return '${TERRA_DATA_DIR}/' + str(path.relative_to(
            Path(data_directory_path).resolve()))
    except ValueError:
        return str(path)


def part_summary(model: str, dataset: str, args) -> dict:
    """The summary of a model trained on every row of one dataset's
    tokenized part, in the form of make_splits.py's split summaries."""
    from datasets import load_from_disk
    from corpus_tools import corpus_dataset_path
    for option in ('data_directory_path', 'norm_factor_csv',
                   'token_dictionary_file_path'):
        if not getattr(args, option):
            raise ConfigError(f'--parts needs --{option}')
    part = corpus_dataset_path(args.data_directory_path, dataset)
    if not (part / 'terra_tokenizer_params.json').exists():
        raise ConfigError(f'the tokenized part of {dataset} is missing or '
                          f'incomplete: {part} (run the tokenize stage)')
    data_dir = Path(args.data_directory_path)
    return {
        'variant': model,
        'datasets': [dataset],
        'n_rows': {'train': len(load_from_disk(str(part))),
                   'validation': 0, 'test': 0},
        'samples': {'validation': [], 'test': []},
        'config_data': {
            'tokenized_data_folder_path': with_data_dir(part, data_dir),
            'norm_factor_file_path': with_data_dir(args.norm_factor_csv,
                                                   data_dir),
            'token_dict_folder_path': with_data_dir(
                args.token_dictionary_file_path, data_dir),
            'val_batch_ids': [],
            'test_batch_ids': [],
        },
    }


def build_config(layout: dict, recipe: str, summary: dict, name: str,
                 args) -> tuple[dict, dict]:
    config = copy.deepcopy(layout)
    if recipe == 'terra96m':
        for section, values in TERRA96M.items():
            config.setdefault(section, {}).update(copy.deepcopy(values))
    data = config['data']
    for key in list(data):
        if key in DROP_DATA_KEYS or key.startswith('precomputed_epoch'):
            del data[key]
    config_data = summary['config_data']
    missing = [k for k in ('tokenized_data_folder_path',
                           'norm_factor_file_path', 'token_dict_folder_path')
               if k not in config_data]
    if missing:
        raise ConfigError(f"split summary of {summary['variant']} lacks "
                          f'config_data {missing}')
    for key in VARIANT_DATA_KEYS:
        if key in config_data:
            data[key] = copy.deepcopy(config_data[key])
    if 'precomputed_split' not in config_data:
        # every row of the dataset (a part model); not the layout config's split
        data.pop('precomputed_split', None)
    data['dataset_name'] = name
    data['raw_data_folder_path'] = silver_root(
        config_data['tokenized_data_folder_path'])
    if args.batch_size:
        data['batch_size'] = args.batch_size
    if args.num_workers is not None:
        data['num_workers'] = args.num_workers
    optimization, state = config['optimization'], config.setdefault('state', {})
    if args.epochs:
        optimization['epochs'] = args.epochs
    global_batch = int(data['batch_size']) * args.gpus
    values = scaled(global_batch)
    if not args.no_scaling:
        optimization['lr'] = values['lr']
        optimization['ema'] = [values['ema'], 1.0]
        optimization['weight_decay'] = values['weight_decay']
        optimization['final_weight_decay'] = values['final_weight_decay']
        state['checkpoint_freq_iter'] = values['checkpoint_freq_iter']
    state.update({'folder_path': None, 'load_checkpoint': False,
                  'read_checkpoint': None})
    values['global_batch'] = global_batch
    return config, values


def check_config(config: dict, n_train: int, token_dict_cache: dict):
    """train()'s start-up check, the training split and the model build."""
    from terra.datasets.utils import prepare_dataset
    from terra.tokenizers.params import reconcile_tokenizer_params
    from terra.utils.helper import init_model_from_config
    import pickle

    data = expand(copy.deepcopy(config['data']))
    for key in ('tokenized_data_folder_path', 'precomputed_split',
                'token_dict_folder_path', 'norm_factor_file_path'):
        if key == 'precomputed_split' and key not in data:
            continue
        if '$' in str(data[key]) or not os.path.exists(data[key]):
            raise ConfigError(f"data.{key} '{data[key]}' does not exist (is "
                              'TERRA_DATA_DIR set?)')
    try:
        reconcile_tokenizer_params(copy.deepcopy(data))
    except ValueError as error:
        raise ConfigError('the config does not fit the corpus (the check '
                          f'train() runs at start-up):\n{error}')
    dataset = prepare_dataset({'data': data, 'meta': config['meta']},
                              train_mode=True)[0]
    if len(dataset) != n_train:
        raise ConfigError(f'the training split loads with {len(dataset):,} '
                          f'rows, its summary gives {n_train:,}')
    path = data['token_dict_folder_path']
    if path not in token_dict_cache:
        with open(path, 'rb') as f:
            token_dict_cache[path] = pickle.load(f)
    encoder, predictor = init_model_from_config(
        {**config, 'data': data}, token_dict_cache[path], 'cpu')
    n_params = sum(p.numel() for p in encoder.parameters())
    n_params += sum(p.numel() for p in predictor.parameters())
    del encoder, predictor
    return len(dataset), n_params


def main(args) -> int:
    parts = {}
    for entry in (e for e in args.parts.split(',') if e):
        model, sep, dataset = entry.partition('=')
        if not sep or not model or not dataset:
            raise ConfigError(f'--parts entry {entry!r}: expected '
                              'model=dataset')
        parts[model] = dataset
    splits_dir = Path(args.splits_dir) if args.splits_dir else None
    if args.variants:
        variants = [v for v in args.variants.split(',') if v]
    elif parts:
        variants = []
    else:
        variants = sorted(p.name for p in splits_dir.iterdir()
                          if (p / 'split_summary.json').exists()) \
            if splits_dir and splits_dir.is_dir() else []
    if variants and splits_dir is None:
        raise ConfigError('--variants needs --splits_dir')
    if not variants and not parts:
        raise ConfigError(f'no split variant in {splits_dir} (run the split '
                          'stage first) and no --parts')
    both = sorted(set(variants) & set(parts))
    if both:
        raise ConfigError(f'{both} given as a variant and as a part model')
    layout_path = Path(os.path.expandvars(args.layout_config))
    if '$' in str(layout_path) or not layout_path.exists():
        raise ConfigError(f"layout config '{args.layout_config}' does not "
                          'exist (is TERRA_REPO_DIR set?)')
    layout = yaml.safe_load(layout_path.read_text())
    out_dir = Path(args.out_dir)
    log(f'Layout config: {layout_path} (TERRA {git_commit(layout_path)})')
    log(f'Recipe: {args.recipe}; {args.gpus} GPU(s)')
    if args.recipe == 'layout' and (layout.get('meta') or {}).get(
            'count_encoding') == 'value_bins' and not layout['data'].get('nz_spc'):
        log('WARNING: count_encoding value_bins with data.nz_spc false takes '
            "the code path of MD-1 (a batch without padding stops DDP) unless "
            'TERRA has the MD-1 fix; the terra96m recipe (mlp) does not.')

    planned, token_dicts = [], {}
    for variant in variants + list(parts):
        if variant in parts:
            summary = part_summary(variant, parts[variant], args)
        else:
            summary_path = splits_dir / variant / 'split_summary.json'
            if not summary_path.exists():
                raise ConfigError(f'no split summary {summary_path} (run the '
                                  'split stage first)')
            summary = json.loads(summary_path.read_text())
        name = f'{args.name}_{variant}'
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise ConfigError(f'config name {name!r}: use letters, digits, '
                              '_ . - only')
        out = out_dir / f'{name}.yaml'
        if out.exists() and not args.overwrite:
            raise ConfigError(f'{out} exists (--overwrite replaces it)')
        config, values = build_config(layout, args.recipe, summary, name, args)
        n_train = int(summary['n_rows']['train'])
        n_params = None
        if not args.no_check:
            _, n_params = check_config(config, n_train, token_dicts)
        planned.append((variant, summary, config, values, out, n_params))

    out_dir.mkdir(parents=True, exist_ok=True)
    written_by = ('terra-reproducibility analysis/slurm_corpus_build/'
                  'make_training_configs.py')
    for variant, summary, config, values, out, n_params in planned:
        opt, data, state = config['optimization'], config['data'], config['state']
        n_train = int(summary['n_rows']['train'])
        steps = math.ceil(n_train / values['global_batch'])
        samples = summary.get('samples') or {}
        header = [
            f"TERRA training config {config['data']['dataset_name']} "
            f'(written by {written_by})',
            f'recipe: {args.recipe}'
            + (' (released TERRA-96M, TERRA cc87839 '
               'hst_corpus_110m_gtsmall_aws.yaml)' if args.recipe == 'terra96m'
               else '')
            + f' on the keys of {layout_path} (TERRA {git_commit(layout_path)})',
            (f"variant {variant}: datasets {', '.join(summary['datasets'])}; "
             f'{n_train:,} training rows; held out: '
             f"{len(samples.get('validation') or [])} validation, "
             f"{len(samples.get('test') or [])} test samples")
            if 'precomputed_split' in summary['config_data'] else
            (f"model {variant}: every row of the tokenized part of "
             f"{summary['datasets'][0]} ({n_train:,} rows; no split, nothing "
             'held out)'),
            f"layout: {args.gpus} GPU(s) x {data['batch_size']} per GPU = global "
            f"{values['global_batch']}; about {steps:,} steps per epoch, "
            f"{opt['epochs']} epochs ({steps * opt['epochs']:,} steps)",
            (f"scaled from TERRA-96M (global {REFERENCE['global_batch']}): r = "
             f"{values['r']:.4f}; lr {opt['lr']}, ema {opt['ema']}, weight decay "
             f"{opt['weight_decay']} -> {opt['final_weight_decay']}, "
             f"checkpoint_freq_iter {state['checkpoint_freq_iter']}")
            if not args.no_scaling else 'recipe values, not scaled (--no_scaling)',
            'run: submit_training.sh sets state.folder_path to a new run '
            'folder; resume from the last end-of-epoch checkpoint with '
            'submit_training.sh --resume <run folder>',
            f'training_gpus: {args.gpus}',
        ]
        with open(out, 'w') as f:
            f.write(''.join(f'# {line}\n' for line in header))
            yaml.safe_dump(config, f, sort_keys=False, width=1000)
        log(f"{variant}: {n_train:,} training rows, global batch "
            f"{values['global_batch']}, ~{steps:,} steps per epoch x "
            f"{opt['epochs']} epochs, lr {opt['lr']}"
            + (f', {n_params / 1e6:.1f}M parameters' if n_params else '')
            + f' -> {out}')
    log('Checked against the corpus record, split loaded, model built: passed'
        if not args.no_check else 'Not checked (--no_check)')
    return 0


def parse_arguments(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--splits_dir', default=None,
                    help='<TERRA_DATA_DIR>/tokenizer/<corpus>/splits')
    ap.add_argument('--variants', default='',
                    help='Comma-separated variants (default without --parts: '
                         'every variant in --splits_dir).')
    ap.add_argument('--parts', default='',
                    help="Models trained on every row of one dataset's "
                         "tokenized part, 'model=dataset,model=dataset'.")
    ap.add_argument('--data_directory_path', default=None,
                    help='For --parts: the data root (TERRA_DATA_DIR).')
    ap.add_argument('--norm_factor_csv', default=None,
                    help='For --parts: the norm factors the parts were '
                         'tokenized with.')
    ap.add_argument('--token_dictionary_file_path', default=None,
                    help='For --parts.')
    ap.add_argument('--name', required=True,
                    help='Config name prefix: <name>_<variant>.yaml.')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--gpus', type=int, required=True,
                    help='GPUs of a run (nodes x GPUs per node).')
    ap.add_argument('--batch_size', type=int, default=None,
                    help="Per GPU (default: the recipe's; terra96m: 144, "
                         'sized for 80 GB GPUs at 10 x 256).')
    ap.add_argument('--epochs', type=int, default=None,
                    help="Default: the recipe's (terra96m: 5).")
    ap.add_argument('--num_workers', type=int, default=None,
                    help="DataLoader workers per GPU (default: the recipe's).")
    ap.add_argument('--recipe', choices=('terra96m', 'layout'),
                    default='terra96m')
    ap.add_argument('--layout_config',
                    default='${TERRA_REPO_DIR}/configs/model/hst_corpus_110m/'
                            'hst_corpus_110m_knn10_seq256.yaml')
    ap.add_argument('--no_scaling', action='store_true')
    ap.add_argument('--no_check', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    return ap.parse_args(argv)


if __name__ == '__main__':
    try:
        sys.exit(main(parse_arguments()))
    except ConfigError as error:
        log(f'ERROR: {error}')
        sys.exit(1)
