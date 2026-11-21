"""
Set the corpus tokenization settings in every per-dataset cell-graph tokenizer
config (config/cell-graph-tokenizer/cell-graph-tokenizer_<dataset>.yaml) of a
cohort, including the TERRA opt-ins that create_tokenized_dataset.py forwards
to the tokenizer but set_tokenization_config_params.py does not write, and
check that create_tokenized_dataset.py will find every key it reads.

Settings written:
- experiment.model_input_size = seq_len_cell * (n_neighs + 1)
  (CellGraphTokenizer: one segment of seq_len_cell tokens for the cell and
  for each of its n_neighs nearest neighbours);
- tokenization.n_neighs, and the fixed settings of FIXED_TOKENIZATION below
  (shifted_log counts, all other normalizations off, stable rank ties ordered
  by the 'tie_break_mean' column of the norm-factor file, an all-padding
  segment for neighbours without an expressed vocabulary gene, explicitly
  stored zeros removed; parallel processing with the generator);
- tokenization.storage_format: 'compact' (default) stores each cell's row
  without padding (TERRA terra.datasets.compact; the model reads the same
  inputs as from padded rows, but only a TERRA with that module can tokenize
  and read such data). 'padded' removes the key, so any TERRA tokenizes the
  configs and writes today's padded rows;
- tokenization.stream_per_file: true (default; --no-stream_per_file removes
  it): the dataset is assembled one input file at a time (same rows, lower
  peak memory);
- paths.norm_factor_file_path (read for rank_tie_break_norm_factor);
  paths.pf_targets_file_path is cleared (only read for pflog1ppf);
- optionally paths.data_directory_path and paths.token_dictionary_file_path.
Keys the cell-graph tokenizer rejects (aggregate_own_neighbors) or that
would record non-default, irrelevant settings are removed.

Run from the terra-reproducibility root (the per-dataset tokenization jobs
read the configs from there). hst_corpus_110m, 16 neighbours x 512 genes
(model_input_size 8704), compact rows:
    python analysis/tokenization/set_tokenizer_options.py \
        --names analysis/data_preparation/hst_corpus_110m/hst_corpus_110m_dataset_names.txt \
        --n_neighs 16 --seq_len_cell 512 --storage_format compact \
        --norm_factor_file_path /nfs/team361/sb75/terra-reproducibility/artifacts/data/hst_corpus_110m/hst_corpus_110m_train_norm_factors.csv \
        --data_directory_path /lustre/scratch126/cellgen/lotfollahi/DATASETS/
Add --check_only to validate the configs without writing them (same
arguments), and --require_files on the farm to also require the norm-factor
file (with a finite tie_break_mean for every vocabulary gene), the token
dictionary and every silver folder. The script prints the settings of the
first config, the output names the tokenization, concatenation, split and
n_nonzero steps derive from them, and the number of configs written.
"""
import argparse
import ast
import pickle
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CREATE_SCRIPT = (REPO_ROOT / 'analysis' / 'tokenization'
                 / 'create_tokenized_dataset.py')
TOKENIZER = 'cell-graph-tokenizer'
# Part of the output names (create_tokenized_dataset.py, concatenate_datasets.py)
NUM_SHARDS = 32
TIE_BREAK_COLUMN = 'tie_break_mean'
STORAGE_FORMATS = ('padded', 'compact')

FIXED_TOKENIZATION = {
    'rank_cell_norm_method': None,
    'rank_gene_norm_method': None,
    'rank_count_norm_method': None,
    'count_cell_norm_method': None,
    'count_gene_norm_method': None,
    'count_count_norm_method': 'shifted_log',
    'include_zero_expr_genes': False,
    'radius': None,
    'delaunay': False,
    'num_shards': NUM_SHARDS,
    'processing_mode': 'parallel',
    'use_generator': True,
    'keep_in_memory': False,
    # TERRA opt-ins: forwarded by create_tokenized_dataset.py only when they
    # are keys of the 'tokenization' section, recorded in the dataset's
    # terra_tokenizer_params.json, reconciled by train() and read by inference
    'stable_rank_ties': True,
    'rank_tie_break_norm_factor': TIE_BREAK_COLUMN,
    'pad_empty_neighbors': True,
    'eliminate_stored_zeros': True,
}
# Removed from the 'tokenization' section:
# - aggregate_own_neighbors, split_cell_neigh_equally: cell-neighborhood
#   tokenizer only. create_tokenized_dataset.py forwards
#   aggregate_own_neighbors, and CellGraphTokenizer passes it on to
#   CellBaseTokenizer.__init__, which raises a TypeError;
# - the other forwarded opt-ins: they do not change shifted_log tokens without
#   gene or cell normalization; left out, the tokenizer defaults are recorded
#   for every dataset alike (concatenate_datasets.py refuses differing records);
# - the forwarded column options: the defaults (all five special-token
#   columns, no neighbour cell ids) are what the model configs expect
REMOVED_TOKENIZATION = (
    'aggregate_own_neighbors', 'split_cell_neigh_equally',
    'zero_expr_from_raw_counts', 'gcrd_basis_target_read_depth',
    'gcrd_target_read_depth_per_gene', 'seurat_loess_span',
    'include_special_tokens', 'special_tokens', 'add_neigh_cell_ids',
)
# Opt-ins of FIXED_TOKENIZATION that create_tokenized_dataset.py must forward
OPTINS = ('stable_rank_ties', 'rank_tie_break_norm_factor',
          'pad_empty_neighbors', 'eliminate_stored_zeros')


def parse_names(names_file_path: str) -> list[str]:
    """Dataset names: first comma-separated field of each non-empty line, as
    concatenate_datasets.py reads them (pd.read_csv(index_col=0))."""
    with open(names_file_path) as f:
        names = [line.split(',')[0].strip() for line in f if line.strip()]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f'Duplicated dataset names: {duplicates}')
    return names


def _subscript_chain(node: ast.AST) -> list | None:
    """['tokenization', 'n_neighs'] for config["tokenization"]["n_neighs"]."""
    keys = []
    while isinstance(node, ast.Subscript):
        if not (isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            return None
        keys.append(node.slice.value)
        node = node.value
    if isinstance(node, ast.Name) and node.id == 'config':
        return keys[::-1]
    return None


def keys_read_by_create_script(script_path: Path) -> tuple[set, set, set]:
    """
    The config keys create_tokenized_dataset.py reads: (section, key) pairs
    read with config[...][...] (required), read with .get() (optional), and
    the 'tokenization' keys it forwards to the tokenizer when present.
    """
    tree = ast.parse(script_path.read_text())
    required, optional, forwarded = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            chain = _subscript_chain(node)
            if chain and len(chain) == 2:
                required.add(tuple(chain))
        elif (isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)
              and node.func.attr == 'get' and node.args
              and isinstance(node.args[0], ast.Constant)):
            chain = _subscript_chain(node.func.value)
            if chain and len(chain) == 1:
                optional.add((chain[0], node.args[0].value))
        elif (isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name)
                      and t.id == 'optional_tokenizer_kwargs'
                      for t in node.targets)):
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Tuple):
                    forwarded |= {e.value for e in sub.elts
                                  if isinstance(e, ast.Constant)}
    return required, optional, forwarded


def tokenizer_parameters() -> set | None:
    """Keyword arguments CellGraphTokenizer accepts, or None without TERRA."""
    try:
        import inspect
        from terra.tokenizers.cell_tokenizers import (CellBaseTokenizer,
                                                      CellGraphTokenizer)
    except ImportError:
        return None
    params = set()
    for cls in (CellGraphTokenizer, CellBaseTokenizer):
        params |= {name for name, p in inspect.signature(
            cls.__init__).parameters.items()
            if p.kind is not inspect.Parameter.VAR_KEYWORD}
    return params - {'self'}


def output_file_prefix(config: dict) -> str:
    """The name create_tokenized_dataset.py gives the tokenized dataset
    (<prefix>.dataset), computed as it does."""
    e, t = config['experiment'], config['tokenization']
    graph = ''
    if t['n_neighs']:
        graph += f"knn_{t['n_neighs']}_"
    if t['radius']:
        graph += f"radius_{t['radius']}_"
    if t['delaunay']:
        graph += 'delaunay_'
    return (f"{e['experiment_name']}_{t['num_shards']}_"
            f"{t['rank_cell_norm_method']}_{t['rank_gene_norm_method']}_"
            f"{t['rank_count_norm_method']}_{t['count_cell_norm_method']}_"
            f"{t['count_gene_norm_method']}_{t['count_count_norm_method']}_"
            f"{graph}{e['model_input_size']}"
            f"{'_incl_zero_expr' if t['include_zero_expr_genes'] else ''}")


def check_norm_factor_file(norm_factor_file_path: str,
                           token_dictionary_file_path: str | None) -> str:
    """Check the tie-break column; return a one-line result."""
    import numpy as np
    import pandas as pd
    df = pd.read_csv(norm_factor_file_path)
    missing = {'gene_id', TIE_BREAK_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"'{norm_factor_file_path}' has no column(s) "
                         f'{sorted(missing)}.')
    # First row of a gene, as the tokenizer reads it
    values = df.drop_duplicates('gene_id').set_index('gene_id')[
        TIE_BREAK_COLUMN].astype(np.float64)
    result = (f'{len(values)} genes, {int(np.isfinite(values).sum())} with a '
              f'finite {TIE_BREAK_COLUMN}')
    if token_dictionary_file_path and Path(token_dictionary_file_path).exists():
        with open(token_dictionary_file_path, 'rb') as f:
            genes = [k for k in pickle.load(f) if 'ENS' in k]
        found = values.reindex(genes)
        bad = found.index[~np.isfinite(found.to_numpy())].tolist()
        if bad:
            raise ValueError(
                f'{len(bad)} of {len(genes)} vocabulary genes have no finite '
                f"'{TIE_BREAK_COLUMN}' in '{norm_factor_file_path}' (e.g. "
                f'{bad[:5]}); tokenizing a cell with such a gene raises.')
        result += f'; all {len(genes)} vocabulary genes covered'
    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--names', required=True,
                    help='File with the dataset names of the cohort.')
    ap.add_argument('--n_neighs', type=int, required=True,
                    help='Number of spatial neighbours (kNN) per cell.')
    ap.add_argument('--seq_len_cell', type=int, required=True,
                    help='Gene tokens per cell (segment length).')
    ap.add_argument('--norm_factor_file_path', required=True,
                    help=f"Norm-factor csv with a '{TIE_BREAK_COLUMN}' "
                         'column for every vocabulary gene.')
    ap.add_argument('--data_directory_path', default=None,
                    help='Data root holding silver/ and gold/ (kept if '
                         'not given).')
    ap.add_argument('--token_dictionary_file_path', default=None,
                    help='Token dictionary (kept if not given).')
    ap.add_argument('--config_dir',
                    default=str(REPO_ROOT / 'config' / TOKENIZER))
    ap.add_argument('--cohort_name', default='hst_corpus_110m',
                    help='Only used to print the corpus names.')
    ap.add_argument('--storage_format', choices=STORAGE_FORMATS,
                    default='compact',
                    help="'compact' (default): rows without padding (needs "
                         "a TERRA with terra.datasets.compact); 'padded': "
                         "today's rows (the key is removed).")
    ap.add_argument('--stream_per_file',
                    action=argparse.BooleanOptionalAction, default=True,
                    help='Set tokenization.stream_per_file: true (default; '
                         'lower peak memory, same rows), or remove it.')
    ap.add_argument('--require_files', action='store_true',
                    help='Fail if the norm-factor file, token dictionary or '
                         'a silver folder is missing (default: warn).')
    ap.add_argument('--check_only', action='store_true',
                    help='Validate the configs as they are; write nothing.')
    args = ap.parse_args()

    if args.n_neighs < 1 or args.seq_len_cell < 1:
        raise ValueError('n_neighs and seq_len_cell must be positive.')
    model_input_size = args.seq_len_cell * (args.n_neighs + 1)
    # CellGraphTokenizer: seq_len_cell = int(model_input_size / (n_neighs + 1))
    assert int(model_input_size / (args.n_neighs + 1)) == args.seq_len_cell

    required, optional, forwarded = keys_read_by_create_script(CREATE_SCRIPT)
    compact = args.storage_format == 'compact'
    # Tokenizer keyword arguments the configs set beyond the fixed ones
    set_optins = set(OPTINS) | ({'storage_format'} if compact else set())
    if (args.stream_per_file
            and ('tokenization', 'stream_per_file') not in optional):
        raise ValueError('create_tokenized_dataset.py does not read '
                         'tokenization.stream_per_file.')
    not_forwarded = set_optins - forwarded
    if not_forwarded:
        # They would be written to the configs but silently ignored
        raise ValueError(
            f'{CREATE_SCRIPT} does not forward {sorted(not_forwarded)} to the '
            'tokenizer; update it before tokenizing.')
    accepted = tokenizer_parameters()
    if accepted is None:
        print('WARNING: TERRA is not importable here; not checked that its '
              f'CellGraphTokenizer accepts {sorted(set_optins)}.')
    else:
        rejected = set_optins - accepted
        if rejected:
            raise ValueError(
                f'The installed TERRA\'s CellGraphTokenizer does not accept '
                f'{sorted(rejected)}; tokenizing these configs needs a TERRA '
                'with them (storage_format: feat/compact-tokenized-data).')

    names = parse_names(args.names)
    config_dir = Path(args.config_dir)
    problems, warnings, prefixes = [], [], []
    files_checked = False
    for i, name in enumerate(names):
        path = config_dir / f'{TOKENIZER}_{name}.yaml'
        if not path.exists():
            problems.append(f'{name}: no config {path}')
            continue
        with open(path) as f:
            config = yaml.safe_load(f)
        if not args.check_only:
            assert config['experiment']['tokenizer'] == TOKENIZER, path
            assert config['experiment']['experiment_name'] == name, path
            config['experiment']['model_input_size'] = model_input_size
            tok = config['tokenization']
            tok.update(FIXED_TOKENIZATION)
            tok['n_neighs'] = args.n_neighs
            if compact:
                tok['storage_format'] = 'compact'
            else:
                tok.pop('storage_format', None)
            if args.stream_per_file:
                tok['stream_per_file'] = True
            else:
                tok.pop('stream_per_file', None)
            for key in REMOVED_TOKENIZATION:
                tok.pop(key, None)
            paths = config['paths']
            paths['norm_factor_file_path'] = args.norm_factor_file_path
            paths['pf_targets_file_path'] = None
            if args.data_directory_path:
                paths['data_directory_path'] = args.data_directory_path
            if args.token_dictionary_file_path:
                paths['token_dictionary_file_path'] = (
                    args.token_dictionary_file_path)
            with open(path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            with open(path) as f:
                config = yaml.safe_load(f)

        # What create_tokenized_dataset.py will read
        e, tok, paths = (config['experiment'], config['tokenization'],
                         config['paths'])
        for section, key in sorted(required):
            if key not in (config.get(section) or {}):
                problems.append(f'{name}: {section}.{key} missing (read by '
                                'create_tokenized_dataset.py)')
        expected = dict(FIXED_TOKENIZATION, n_neighs=args.n_neighs)
        for key, value in expected.items():
            if tok.get(key, '<absent>') != value:
                problems.append(f'{name}: tokenization.{key} = '
                                f'{tok.get(key, "<absent>")!r}, not {value!r}')
        for key in REMOVED_TOKENIZATION:
            if key in tok:
                problems.append(f'{name}: tokenization.{key} must be removed')
        if tok.get('storage_format', 'padded') != args.storage_format:
            problems.append(f'{name}: tokenization.storage_format = '
                            f'{tok.get("storage_format", "<absent>")!r}, not '
                            f'{args.storage_format!r}')
        if not compact and 'storage_format' in tok:
            problems.append(f'{name}: tokenization.storage_format must be '
                            'removed for padded rows (a TERRA without the '
                            'compact format rejects the key)')
        if bool(tok.get('stream_per_file', False)) != args.stream_per_file:
            problems.append(f'{name}: tokenization.stream_per_file = '
                            f'{tok.get("stream_per_file", "<absent>")!r}')
        if (args.data_directory_path
                and paths.get('data_directory_path')
                != args.data_directory_path):
            problems.append(f'{name}: data_directory_path '
                            f'{paths.get("data_directory_path")!r}')
        if (args.token_dictionary_file_path
                and paths.get('token_dictionary_file_path')
                != args.token_dictionary_file_path):
            problems.append(f'{name}: token_dictionary_file_path '
                            f'{paths.get("token_dictionary_file_path")!r}')
        if e.get('experiment_name') != name:
            problems.append(f'{name}: experiment_name '
                            f'{e.get("experiment_name")!r}')
        if e.get('tokenizer') != TOKENIZER:
            problems.append(f'{name}: tokenizer {e.get("tokenizer")!r}')
        if e.get('model_input_size') != model_input_size:
            problems.append(f'{name}: model_input_size '
                            f'{e.get("model_input_size")} != '
                            f'{model_input_size}')
        if paths.get('norm_factor_file_path') != args.norm_factor_file_path:
            problems.append(f'{name}: norm_factor_file_path '
                            f'{paths.get("norm_factor_file_path")!r}')
        if paths.get('pf_targets_file_path') is not None:
            problems.append(f'{name}: pf_targets_file_path is set')
        if config['wandb'].get('save_artifact'):
            # The artifact branch of create_tokenized_dataset.py adds the
            # whole silver and gold folders to a wandb artifact
            problems.append(f'{name}: wandb.save_artifact must be false')
        prefixes.append(output_file_prefix(config))

        # Files the tokenization reads or overwrites, where reachable
        data_dir = Path(paths['data_directory_path'])
        if data_dir.exists():
            silver = data_dir / 'silver' / name
            if not any(silver.glob('**/*.h5ad')):
                (problems if args.require_files else warnings).append(
                    f'{name}: no .h5ad file under {silver}')
            part = (data_dir / 'gold' / TOKENIZER / name
                    / f'{prefixes[-1]}.dataset')
            if part.exists():
                warnings.append(f'{name}: {part} exists and will be '
                                'overwritten')
        if not files_checked:
            files_checked = True
            nf = paths['norm_factor_file_path']
            td = paths['token_dictionary_file_path']
            if Path(nf).exists():
                print('Norm-factor file:', check_norm_factor_file(nf, td))
            else:
                (problems if args.require_files else warnings).append(
                    f'norm-factor file {nf} not found (needed before '
                    'tokenizing)')
            if not Path(td).exists():
                (problems if args.require_files else warnings).append(
                    f'token dictionary {td} not found')
        if i == 0:
            print(f'--- {path}')
            print(yaml.safe_dump(config, default_flow_style=False))

    suffixes = {p[len(n):] for p, n in zip(prefixes, names)}
    print(f'model_input_size {model_input_size} = {args.seq_len_cell} genes x '
          f'({args.n_neighs} neighbours + 1); storage format '
          f'{args.storage_format}; stream_per_file {args.stream_per_file}')
    if len(suffixes) == 1:
        suffix = suffixes.pop()
        corpus = f'{args.cohort_name}{suffix}'
        print('Per-dataset parts : <data_directory_path>/gold/'
              f'{TOKENIZER}/<name>/<name>{suffix}.dataset')
        print(f'Corpus            : <data_directory_path>/gold/{TOKENIZER}/'
              f'{args.cohort_name}/{corpus}.dataset')
        print(f'Split files       : <tokenizer_folder>/{corpus}'
              '_precomputed_split_{train,validation,test,full}.pkl')
        print(f'n_nonzero files   : <tokenizer_folder>/{corpus}'
              '_n_nonzero_tokens_{train,validation,test,full}.pkl')
    else:
        problems.append(f'the configs give different name suffixes: '
                        f'{sorted(suffixes)}')
    for w in warnings[:20]:
        print('WARNING:', w)
    if len(warnings) > 20:
        print(f'WARNING: ... {len(warnings) - 20} more')
    if problems:
        for p in problems[:50]:
            print('PROBLEM:', p)
        raise SystemExit(f'{len(problems)} problem(s) in {len(names)} '
                         'configs.')
    print(f"{len(names)} configs "
          f"{'checked' if args.check_only else 'written and checked'}.")


if __name__ == '__main__':
    main()
