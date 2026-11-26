#!/usr/bin/env python
"""
Write a test copy of a TERRA training job that reads the re-tokenized
hst_corpus_110m corpus (16 neighbours x 512 genes, compact rows), e.g. to run
the TERRA-96M recipe on one GPU.

Reads a job config (the format of the TERRA checkout's
scripts/bsub_submit_tiger.sh) and the model config it names (the recipe:
model, masking, optimization, batch size), and writes both test configs into
the TERRA checkout (--terra_dir, default $TERRA_REPO_DIR), under one name
(--config_name, default <job config name>_test<gpus>gpu), as the committed
configs are kept:

  configs/model/<dataset_name>/<config_name>.yaml
      the recipe, with the data keys that describe the corpus and how the
      model reads it taken from --layout_config, a committed config that reads
      the new corpus in the recipe's layout: corpus, train split, norm
      factors, token dictionary, silver folder, layout keys (n_neighs /
      model_input_size of the corpus, n_segments, tokenized_seq_len_cell,
      truncate_neighbors), tokenizer settings and the 219 held-out batch ids.
      precomputed_n_nonzero_tokens, precomputed_epoch_* and
      pf_targets_file_path are dropped: compact rows give the lengths of each
      layout, and shifted_log reads no pf targets. A new run folder is used
      (state.folder_path null, no resume) and, by default, no mid-epoch
      checkpoint is written (--checkpoint_freq_iter).
  configs/job/<dataset_name>/<config_name>.yaml
      the job config with one node of --gpus GPUs, --cores cores and --mem_gb
      GB, RUN_NAME <run_name> (also the wandb run id with EXPERIMENT_NAME) and
      CONFIG_FILE the test model config (as ${TERRA_REPO_DIR}/configs/...,
      which the launcher resolves). Only these six values change; the rest of
      the file is kept as it is.

Existing files are not replaced (--overwrite replaces them), and the source
configs are never written to.

The recipe must read the layout of the layout config (same seq_len_cell and
seq_len_neighborhood): the default, hst_corpus_110m_knn10_seq256.yaml, reads
10 neighbours x 256 genes, as TERRA-96M does.

Before writing anything, the test model config's data section goes through the
check that train() runs at start-up (terra.tokenizers.params.
reconcile_tokenizer_params: tokenizer settings, norm-factor sha256 and layout
against the corpus record), so that a mismatch shows here and not after the
GPU job has started. Run it with the TERRA venv, after sourcing the TERRA
checkout's scripts/cluster_env.sh (the TERRA_* paths the configs use).

Example (farm):
  source /nfs/team361/sb75/terra/scripts/cluster_env.sh
  source /nfs/team361/sb75/.venvs/terra/bin/activate
  python analysis/model_training/make_test_run_configs.py \\
      --job_config /nfs/team361/sb75/terra/configs/job/hst_corpus_110m/hst_corpus_110m_terra96m.yaml \\
      --run_name terra96m_corpus16x512_test1gpu_1
  cd /nfs/team361/sb75/terra && bash scripts/bsub_submit_tiger.sh \\
      configs/job/hst_corpus_110m/hst_corpus_110m_terra96m_test1gpu.yaml
"""

import argparse
import copy
import os
import re
import sys

import yaml

# Data keys taken from the layout config: what the corpus is and how the model
# reads it (the tokenizer settings come from TERRA's own lists below)
LAYOUT_KEYS = (
    'dataset_name', 'tokenizer_type', 'model_input_size', 'n_neighs',
    'seq_len_cell', 'seq_len_neighborhood', 'n_segments',
    'tokenized_seq_len_cell', 'truncate_neighbors',
    'raw_data_folder_path', 'token_dict_folder_path',
    'tokenized_data_folder_path', 'norm_factor_file_path',
    'precomputed_split', 'val_batch_ids', 'test_batch_ids')
# Must be in the layout config
REQUIRED_LAYOUT_KEYS = (
    'tokenized_data_folder_path', 'precomputed_split', 'norm_factor_file_path',
    'seq_len_cell', 'seq_len_neighborhood', 'n_segments',
    'tokenized_seq_len_cell', 'truncate_neighbors')
# Dropped from the recipe's data section
DROP_KEYS = ('precomputed_n_nonzero_tokens', 'pf_targets_file_path')
# Job config values this script sets
JOB_KEYS = ('NUM_NODES', 'NUM_GPUS_NODE', 'NUM_PROCESSES_NODE', 'MEM_NODE',
            'RUN_NAME', 'CONFIG_FILE')


def fail(message):
    print(f'ERROR: {message}', file=sys.stderr)
    sys.exit(1)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def expand(obj):
    """${VAR} expansion as TERRA's config loader does it (terra.utils.config)."""
    if isinstance(obj, dict):
        return {k: expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def find_key(obj, key):
    """Values of `key` anywhere in a nested dict."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found += find_key(v, key)
    return found


def resolved_path(value, what):
    path = os.path.expandvars(str(value))
    if '$' in path:
        fail(f"{what} '{value}' needs variables that are not set: source the "
             "TERRA checkout's scripts/cluster_env.sh first.")
    return path


def short(value):
    if isinstance(value, list) and len(value) > 4:
        return f'[{len(value)} items: {value[0]!r}, {value[1]!r}, ...]'
    return repr(value)


def set_job_values(text, values):
    """Replace the values of the given (unique, capital) keys in the job
    config's text, keeping its layout and comments."""
    lines = text.split('\n')
    for key, value in values.items():
        pattern = re.compile(
            rf'^(\s*{key}:[ \t]*)("[^"]*"|\'[^\']*\'|[^#\s]+)(.*)$')
        hits = [i for i, line in enumerate(lines) if pattern.match(line)]
        if len(hits) != 1:
            fail(f'the job config has {len(hits)} lines with a value for {key} '
                 '(expected 1).')
        new = f'"{value}"' if isinstance(value, str) else str(value)
        lines[hits[0]] = pattern.sub(lambda m: m.group(1) + new + m.group(3),
                                     lines[hits[0]])
    return '\n'.join(lines)


def main(args):
    from terra.tokenizers.params import (NORM_METHOD_KEYS, OPTIONAL_KEYS,
                                         reconcile_tokenizer_params)

    # ---- the job config and the recipe it names
    with open(args.job_config) as f:
        job_text = f.read()
    job = yaml.safe_load(job_text)
    config_files = find_key(job, 'CONFIG_FILE')
    run_names = find_key(job, 'RUN_NAME')
    if len(config_files) != 1 or len(run_names) != 1:
        fail(f"'{args.job_config}' needs exactly one CONFIG_FILE and one "
             'RUN_NAME.')
    if args.run_name == run_names[0]:
        fail(f'--run_name {args.run_name!r} is the RUN_NAME of the job config: '
             'wandb would append the test to that run. Use a new name.')
    recipe_path = resolved_path(config_files[0], 'CONFIG_FILE')
    recipe = load_yaml(recipe_path)
    layout_path = resolved_path(args.layout_config, '--layout_config')
    layout = load_yaml(layout_path)
    r_data, l_data = recipe['data'], layout['data']

    missing = [k for k in REQUIRED_LAYOUT_KEYS if k not in l_data]
    if missing:
        fail(f"the layout config '{layout_path}' lacks {missing}.")
    # ---- the recipe must read the layout config's layout
    for key in ('seq_len_cell', 'seq_len_neighborhood'):
        if r_data.get(key) != l_data[key]:
            fail(f'the recipe reads data.{key} = {r_data.get(key)!r}, the '
                 f'layout config {l_data[key]!r}. Use a layout config with the '
                 "recipe's layout (configs/model/hst_corpus_110m/"
                 'hst_corpus_110m_knn<neighbours>_seq<genes>.yaml).')
    if 'n_segments' in r_data and r_data['n_segments'] != l_data['n_segments']:
        fail(f"the recipe's data.n_segments is {r_data['n_segments']!r}, the "
             f"layout config's {l_data['n_segments']!r}.")

    # ---- the test model config
    test = copy.deepcopy(recipe)
    data = test['data']
    dropped = [k for k in data
               if k in DROP_KEYS or k.startswith('precomputed_epoch')
               or k in OPTIONAL_KEYS]
    for k in dropped:
        del data[k]
    for k in LAYOUT_KEYS + tuple(NORM_METHOD_KEYS) + tuple(OPTIONAL_KEYS):
        if k in l_data:
            data[k] = copy.deepcopy(l_data[k])
    if args.precomputed_split:
        data['precomputed_split'] = args.precomputed_split
    state = test.setdefault('state', {})
    state.update({'folder_path': None, 'load_checkpoint': False,
                  'read_checkpoint': None,
                  'checkpoint_freq_iter': args.checkpoint_freq_iter})
    if args.log_freq is not None:
        state['log_freq'] = args.log_freq
    if args.epochs is not None:
        test['optimization']['epochs'] = args.epochs

    # ---- train()'s start-up check of the data section against the corpus
    check = expand(copy.deepcopy(data))
    for key in ('tokenized_data_folder_path', 'precomputed_split',
                'token_dict_folder_path', 'norm_factor_file_path'):
        if '$' in str(check[key]) or not os.path.exists(check[key]):
            fail(f"data.{key} '{check[key]}' does not exist (are the TERRA_* "
                 'variables set?).')
    try:
        reconcile_tokenizer_params(check)
    except ValueError as e:
        fail(f'the test config does not fit the corpus (the check train() '
             f'runs at start-up):\n{e}')

    meta = test.get('meta') or {}
    if meta.get('count_encoding') == 'value_bins' and not data.get('nz_spc'):
        print('WARNING: count_encoding value_bins with data.nz_spc false '
              'takes the code path of MD-1: at TERRA 7abf568 a batch without '
              'padding crashes or hangs DDP. Use a TERRA checkout with the '
              'MD-1 fix, or count_encoding mlp.')

    # ---- where the test configs go: the TERRA checkout's configs folders,
    # one name for the model and the job config
    terra_dir = os.path.abspath(resolved_path(args.terra_dir, '--terra_dir'))
    if not os.path.isdir(os.path.join(terra_dir, 'configs')):
        fail(f"'{terra_dir}' has no configs folder (--terra_dir is the TERRA "
             'checkout).')
    config_name = args.config_name or (
        os.path.splitext(os.path.basename(args.job_config))[0]
        + f'_test{args.gpus}gpu')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', config_name):
        fail(f'--config_name {config_name!r}: use letters, digits, _ . - only.')
    rel_model = f"configs/model/{data['dataset_name']}/{config_name}.yaml"
    rel_job = f"configs/job/{data['dataset_name']}/{config_name}.yaml"
    model_out = os.path.join(terra_dir, rel_model)
    job_out = os.path.join(terra_dir, rel_job)
    sources = {os.path.realpath(p)
               for p in (args.job_config, recipe_path, layout_path)}
    for path in (model_out, job_out):
        if os.path.realpath(path) in sources:
            fail(f"'{path}' is one of the source configs: choose another "
                 '--config_name.')
        if os.path.exists(path) and not args.overwrite:
            fail(f"'{path}' exists (--overwrite replaces it).")
    # CONFIG_FILE as the committed job configs give it (the launcher resolves
    # ${TERRA_REPO_DIR} at submission)
    env_repo = os.environ.get('TERRA_REPO_DIR')
    if env_repo and os.path.realpath(env_repo) == os.path.realpath(terra_dir):
        config_file = '${TERRA_REPO_DIR}/' + rel_model
    else:
        config_file = model_out

    # ---- the test job config
    new_values = {'NUM_NODES': 1, 'NUM_GPUS_NODE': args.gpus,
                  'NUM_PROCESSES_NODE': args.cores, 'MEM_NODE': args.mem_gb,
                  'RUN_NAME': args.run_name, 'CONFIG_FILE': config_file}
    job_test_text = set_job_values(job_text, new_values)
    job_test = yaml.safe_load(job_test_text)
    for key, value in new_values.items():
        if find_key(job_test, key) != [value]:
            fail(f'could not set {key} in the job config '
                 f'(got {find_key(job_test, key)}).')

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    os.makedirs(os.path.dirname(job_out), exist_ok=True)
    written_by = ('(written by terra-reproducibility analysis/model_training/'
                  'make_test_run_configs.py)')
    with open(model_out, 'w') as f:
        f.write(f'# Test copy of {recipe_path}\n'
                f'# with the data keys of {layout_path}\n# {written_by}\n')
        yaml.safe_dump(test, f, sort_keys=False, width=1000)
    with open(job_out, 'w') as f:
        f.write(f'# Test copy of {args.job_config}: 1 node x {args.gpus} '
                f'GPU(s), RUN_NAME {args.run_name}\n# {written_by}\n'
                + job_test_text)

    # ---- what changed
    print(f'Recipe: {recipe_path}')
    print(f'Layout: {layout_path}')
    print('Data keys changed:')
    for k in sorted(set(r_data) | set(data)):
        old, new = r_data.get(k, '<absent>'), data.get(k, '<absent>')
        if old != new:
            print(f'  {k}: {short(old)} -> {short(new)}')
    print('State: ' + ', '.join(
        f'{k} {state.get(k)!r}' for k in (
            'folder_path', 'load_checkpoint', 'read_checkpoint',
            'checkpoint_freq_iter', 'log_freq')))
    print(f"Epochs: {test['optimization']['epochs']}, batch size per GPU: "
          f"{data.get('batch_size')}")
    print('Job: ' + ', '.join(f'{k} {v}' for k, v in new_values.items()))
    print('Check against the corpus record (as train() runs it): passed')
    print(f'Wrote {model_out}')
    print(f'Wrote {job_out}')
    print(f'Submit: cd {terra_dir} && bash scripts/bsub_submit_tiger.sh '
          f'{rel_job}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--job_config', required=True,
                   help='Job config of the recipe (it names the model config).')
    p.add_argument('--layout_config',
                   default='${TERRA_REPO_DIR}/configs/model/hst_corpus_110m/'
                           'hst_corpus_110m_knn10_seq256.yaml',
                   help='Committed model config that reads the new corpus in '
                        "the recipe's layout (default: 10 neighbours x 256 "
                        'genes).')
    p.add_argument('--terra_dir', default='${TERRA_REPO_DIR}',
                   help='TERRA checkout whose configs/model and configs/job '
                        'folders get the test configs (default: '
                        '$TERRA_REPO_DIR).')
    p.add_argument('--config_name', default=None,
                   help='File name (without .yaml) of both test configs '
                        '(default: <job config name>_test<gpus>gpu).')
    p.add_argument('--run_name', required=True,
                   help='New RUN_NAME (the wandb run id with EXPERIMENT_NAME).')
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--cores', type=int, default=24)
    p.add_argument('--mem_gb', type=int, default=128)
    p.add_argument('--checkpoint_freq_iter', type=int, default=10**9,
                   help='Mid-epoch checkpoint every N iterations (default: '
                        'none; each is about 0.6 GB).')
    p.add_argument('--log_freq', type=int, default=None,
                   help="Default: the recipe's.")
    p.add_argument('--epochs', type=int, default=None,
                   help="Default: the recipe's.")
    p.add_argument('--precomputed_split', default=None,
                   help="Another split pickle of the corpus (default: the "
                        "layout config's train split).")
    p.add_argument('--overwrite', action='store_true')
    main(p.parse_args())
