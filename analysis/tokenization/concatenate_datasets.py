"""
Concatenate the per-dataset tokenized datasets of a cohort into one corpus.

The parts are read from
<data_directory_path>/gold/<tokenizer>/<dataset>/<dataset>_<settings>.dataset,
where <settings> is '<num_shards>_<6 norm methods>_knn_<n_neighs>_<seq_len>'
(the output_file_prefix of create_tokenized_dataset.py, seq_len being the
model_input_size). The corpus is written to
<data_directory_path>/gold/<tokenizer>/<cohort>/<cohort>_<settings>.dataset
together with the parts' tokenizer settings record
(terra_tokenizer_params.json).

Before the save, the script refuses parts without a tokenizer settings record
(a tokenization job that died between writing the rows and the record, a
part copied by hand, or a part tokenized before records were written; pass
--allow_parts_without_record to concatenate such parts into a corpus without
a record), parts with different records, parts of different storage formats
(padded rows, or compact rows without padding, detected from the
'segment_lengths' column), a part whose record contradicts its columns, a
format other than --expect_storage_format, and missing or duplicated cell
ids.

The corpus is written to a sibling folder <corpus>.partial-<id> (<id>: the
LSF job id, or the process id outside LSF), checked (it loads with every
row), given the record, and only then renamed to <corpus>. A killed run
therefore never leaves a corpus folder that mixes two builds; it leaves its
.partial-<id> folder, which nothing reads (delete it when no concatenation
job is running). An existing corpus is refused unless --overwrite: the new
corpus is then written next to it (disk space for both is needed until the
swap), renamed into place, and the old one deleted. Processes that already
have the old corpus open keep reading it (Lustre keeps deleted files that
are open); a process that opens the corpus by path after the swap reads the
new one.

Example (hst_corpus_110m, 16 neighbours x 512 genes, compact rows):
    python concatenate_datasets.py --cohort_name hst_corpus_110m \
        --cohort_dataset_names_file_path <names.txt> \
        --n_neighs 16 --seq_len 8704 --expect_storage_format compact
"""
import argparse
import json
import os
import shutil
import tracemalloc
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd

STORAGE_FORMATS = ('padded', 'compact')


def storage_format_of(ds) -> str:
    """'compact' for compact cell-graph rows (no padding stored; TERRA
    terra.datasets.compact.is_compact), 'padded' otherwise."""
    return 'compact' if 'segment_lengths' in ds.features else 'padded'


def find_part(gold_data_path: Path, dataset_name: str, dir_substring: str,
              seq_len: int) -> Path:
    """
    The tokenized part of `dataset_name`: the folder create_tokenized_dataset.py
    writes (<gold>/<dataset>/<dir_substring>.dataset), or else the one folder
    under <gold> named <dir_substring>.dataset or
    <dir_substring>_<seq_len>.dataset (older layouts). Names are matched
    exactly, so renamed copies (e.g. '..._old') are never picked up.
    """
    expected = gold_data_path / dataset_name / f"{dir_substring}.dataset"
    if expected.is_dir():
        return expected
    names = {f"{dir_substring}.dataset", f"{dir_substring}_{seq_len}.dataset"}
    matches = sorted(p for p in gold_data_path.rglob("*.dataset")
                     if p.is_dir() and p.name in names)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one tokenized dataset for '{dataset_name}' at "
            f"'{expected}' (or one folder named {sorted(names)} under "
            f"'{gold_data_path}'), found {len(matches)}: "
            f"{[str(m) for m in matches]}")
    return matches[0]


def corpus_exists(path) -> bool:
    """True if `path` exists (or is a link) and is not an empty folder."""
    p = Path(path)
    if not (p.exists() or p.is_symlink()):
        return False
    return not (p.is_dir() and not p.is_symlink() and not any(p.iterdir()))


def main(args):
    gold_data_path = Path(args.data_directory_path) / 'gold' / args.tokenizer

    hparam_prefix = '_'.join([str(s) for s in [
        args.num_shards,
        args.rank_cell_norm_method,
        args.rank_gene_norm_method,
        args.rank_count_norm_method,
        args.count_cell_norm_method,
        args.count_gene_norm_method,
        args.count_count_norm_method,
        "knn",
        args.n_neighs,
        args.seq_len]])
    print(hparam_prefix)

    cohort_dir = gold_data_path / args.cohort_name
    output_path = str(
        (Path(cohort_dir) / f"{args.cohort_name}_{hparam_prefix}").with_suffix(
            '.dataset'))
    # Refuse an existing corpus before the (long) loading and checks
    if corpus_exists(output_path) and not args.overwrite:
        raise FileExistsError(
            f"'{output_path}' already holds a corpus. Pass --overwrite to "
            "replace it (the new corpus is written next to it and swapped in "
            "when complete; disk space for both is needed until then), or "
            "delete it first (rm -rf) when no job reads it.")
    # The corpus is saved here first and renamed into place when complete. A
    # requeued job keeps its job id and restarts its own partial folder.
    job_id = os.environ.get('LSB_JOBID') or str(os.getpid())
    partial_path = f'{output_path}.partial-{job_id}'
    leftovers = sorted(str(p) for kind in ('partial', 'replaced')
                       for p in Path(cohort_dir).glob(
                           f'{Path(output_path).name}.{kind}-*')
                       if str(p) != partial_path)
    if leftovers:
        print('WARNING: folders of other or earlier concatenation runs (a '
              'killed run leaves its .partial-<id>; nothing reads them): '
              + ', '.join(leftovers) + '. Delete them when no concatenation '
              'job is running, to free the disk space.')

    ds_list = []
    tokenizer_params = []
    storage_formats = []

    # Read datasets
    datasets_df = pd.read_csv(
        args.cohort_dataset_names_file_path,
        index_col=0,
        header=None)
    print(datasets_df)

    for dataset_name, _ in datasets_df.iterrows():
        print('Processing: ', dataset_name)
        
        dir_substring = f"{dataset_name}_{hparam_prefix}"
        gold_dataset_dir = find_part(gold_data_path, dataset_name,
                                     dir_substring, args.seq_len)

        ds = datasets.load_from_disk(str(gold_dataset_dir))
        params_path = gold_dataset_dir / 'terra_tokenizer_params.json'
        params = (json.loads(params_path.read_text()) if params_path.exists()
                  else None)
        tokenizer_params.append(params)
        storage_format = storage_format_of(ds)
        storage_formats.append(storage_format)
        print(f"{gold_dataset_dir}: {len(ds)} rows, {storage_format}")
        if (params is not None
                and params.get('storage_format', 'padded') != storage_format):
            raise ValueError(
                f"'{gold_dataset_dir}': the record says storage_format "
                f"{params.get('storage_format', 'padded')!r}, but the "
                f"columns are {storage_format}.")

        ds_list.append(ds)

    # Every part must carry its settings record (TERRA writes it right after
    # the rows, cell_tokenizers.py: save_to_disk, then write_tokenizer_params):
    # a part without one may be incomplete, and the corpus could not get a
    # record that train() checks the model config against
    missing_record = [str(name) for name, p in zip(datasets_df.index,
                                                    tokenizer_params)
                      if p is None]
    if missing_record:
        message = (
            f'{len(missing_record)} of {len(tokenizer_params)} parts have no '
            f'terra_tokenizer_params.json: {missing_record}')
        if not args.allow_parts_without_record:
            raise ValueError(
                message + '. Their tokenization may be incomplete (the job '
                'died after writing the rows) or old: rm -rf these parts and '
                're-tokenize them (or pass --allow_parts_without_record to '
                'build a corpus without a record).')
        print(f'WARNING: {message}; the corpus gets no record '
              '(--allow_parts_without_record).')

    # One storage format for all parts (and the expected one), checked
    # before concatenating: rows of both formats in one corpus would be read
    # with the wrong layout
    found_formats = sorted(set(storage_formats))
    print(f"Storage format(s) of the parts: {found_formats}")
    if len(found_formats) > 1:
        raise ValueError(
            'The parts have different storage formats: ' + ', '.join(
                f'{name}: {fmt}' for name, fmt in zip(
                    datasets_df.index, storage_formats)))
    if (args.expect_storage_format is not None
            and found_formats != [args.expect_storage_format]):
        raise ValueError(
            f"The parts are {found_formats[0]}, not "
            f"{args.expect_storage_format} (--expect_storage_format).")

    os.makedirs(cohort_dir, exist_ok=True)

    # Save dataset
    print('Concatenating datasets...')
    tracemalloc.start()
    concatenated_dataset = datasets.concatenate_datasets(ds_list)

    # Checks before the (long) save. The concatenated dataset keeps the
    # tokenizer settings record (which train() checks the model config
    # against) only if all parts were tokenized alike; file paths and the
    # TERRA version may differ (the files' sha256 are compared instead).
    # Parts without a record were refused above unless
    # --allow_parts_without_record; the records present must still agree
    not_compared = ('terra_version', 'pf_targets_file_path',
                    'norm_factor_file_path')
    settings = [{k: v for k, v in p.items() if k not in not_compared}
                for p in tokenizer_params if p is not None]
    if any(p != settings[0] for p in settings[1:]):
        raise ValueError(
            'The datasets were tokenized with different settings '
            '(terra_tokenizer_params.json differs); re-tokenize them '
            'with one configuration before concatenating.')
    # Cell ids must be present and unique: duplicated rows were once removed
    # by a hard-coded row range (fix_dataset_duplicates.py). Counted as
    # large_string: Arrow's distinct count copies the distinct values into one
    # buffer, which holds at most 2 GiB with the 32-bit offsets of 'string'
    # (the 112.6M ids of hst_corpus_110m, about 17 characters each, come
    # within 10% of that). The cast keeps the data buffers and only adds
    # 64-bit offsets.
    cell_id_column = concatenated_dataset.data.column('cell_id')
    if cell_id_column.null_count:
        raise ValueError(f'{cell_id_column.null_count} cells have no cell id.')
    n_distinct = pc.count_distinct(
        cell_id_column.cast(pa.large_string())).as_py()
    n_duplicates = len(concatenated_dataset) - n_distinct
    if n_duplicates:
        raise ValueError(f'{n_duplicates} duplicated cell ids.')

    # Written to a sibling folder and renamed into place once complete, so a
    # killed save never leaves a corpus folder that mixes two builds (the
    # shard names depend only on the number of shards) and never rewrites
    # files that running jobs have memory-mapped
    if os.path.lexists(partial_path):
        shutil.rmtree(partial_path)

    # Shards keep the row order; num_proc only writes them in parallel
    concatenated_dataset.save_to_disk(
        partial_path,
        num_proc=args.num_proc
        #num_shards=args.num_shards
        )
    n_saved = len(datasets.load_from_disk(partial_path))
    if n_saved != len(concatenated_dataset):
        raise RuntimeError(
            f"'{partial_path}' holds {n_saved} rows instead of "
            f'{len(concatenated_dataset)}; the corpus was not replaced.')
    if not missing_record:
        with open(Path(partial_path) / 'terra_tokenizer_params.json',
                  'w') as f:
            json.dump(tokenizer_params[0], f, indent=2, sort_keys=True)

    # Swap the complete corpus into place
    replaced_path = None
    if corpus_exists(output_path):
        if not args.overwrite:
            raise FileExistsError(
                f"'{output_path}' was created while this job ran; the new "
                f"corpus stays in '{partial_path}'.")
        # Unique per attempt (a requeued job keeps its job id)
        target = f'{output_path}.replaced-{job_id}-{os.getpid()}'
        try:
            os.rename(output_path, target)
        except OSError:
            print(f"ERROR: the new corpus is complete in '{partial_path}', "
                  f"but the previous corpus '{output_path}' could not be "
                  'moved aside (it is unchanged).')
            raise
        replaced_path = target
    elif os.path.isdir(output_path):
        os.rmdir(output_path)  # an empty folder
    try:
        os.rename(partial_path, output_path)
    except OSError:
        if replaced_path is not None:
            os.rename(replaced_path, output_path)  # the old corpus back
        print(f"ERROR: the new corpus is complete in '{partial_path}' but "
              f"could not be renamed to '{output_path}'"
              + (' (the previous corpus is back in place)'
                 if replaced_path is not None else '') + '.')
        raise
    print(f"Corpus written: '{output_path}' ({n_saved} rows)")
    if not missing_record:
        print('Tokenizer settings record written: '
              f"{Path(output_path) / 'terra_tokenizer_params.json'}")
    if replaced_path is not None:
        # Processes that already have the old corpus open keep reading it
        try:
            if os.path.islink(replaced_path):
                link_target = os.readlink(replaced_path)
                os.unlink(replaced_path)
                print('The previous corpus was a link: the link was removed, '
                      f"its target '{link_target}' was kept (delete it by "
                      'hand when no job reads it).')
            else:
                shutil.rmtree(replaced_path)
                print('The previous corpus was replaced and deleted.')
        except OSError as e:
            print(f"WARNING: the previous corpus is in '{replaced_path}' and "
                  f'could not be deleted ({e}); delete it by hand.')
    current, peak = tracemalloc.get_traced_memory()
    print(f'Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_directory_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS', # '/nfs/team361/sb75/DATASETS'
                        help='Data directory path.')
    parser.add_argument('--cohort_name',
                        type=str,
                        default='hst_corpus_110m',
                        help='Name of the cohort.')
    parser.add_argument('--cohort_dataset_names_file_path',
                        type=str,
                        default='/nfs/team361/sb75/terra-reproducibility/analysis/data_preparation/hst_corpus_110m/hst_corpus_110m_dataset_names.txt', #'/nfs/team361/sb75/terra-reproducibility/analysis/data_preparation/hst_corpus_110m/hst_corpus_110m_dataset_names_additional.txt'
                        help='Path to the file containing all datasets in the cohort.')
    parser.add_argument('--tokenizer',
                        type=str,
                        default='cell-graph-tokenizer',
                        help='Tokenizer.')
    parser.add_argument('--num_shards',
                        type=int,
                        default='32',
                        help='Number of shards.')
    parser.add_argument('--rank_cell_norm_method',
                        type=str,
                        default=None, # None, 'gene_corrected_read_depth'
                        help='Rank cell norm method.')
    parser.add_argument('--rank_gene_norm_method',
                        type=str,
                        default=None, # None, 'nonzero_mean'
                        help='Rank gene norm method.')
    parser.add_argument('--rank_count_norm_method',
                        type=str,
                        default=None, # None
                        help='Rank count norm method.')
    parser.add_argument('--count_cell_norm_method',
                        type=str,
                        default=None, # None, 'gene_corrected_read_depth'
                        help='Count cell norm method.')
    parser.add_argument('--count_gene_norm_method',
                        type=str,
                        default=None, # None, 'nonzero_mean'
                        help='Count gene norm method.')
    parser.add_argument('--count_count_norm_method',
                        type=str,
                        default='shifted_log', # 'shifted_log', None
                        help='Count count norm method.')
    parser.add_argument('--n_neighs',
                        type=int,
                        default=6, # 10, 6
                        help='Number of neighbors.')
    parser.add_argument('--seq_len',
                        type=int,
                        default=7168, # 2816, 7168
                        help='Sequence length (model_input_size of the parts, '
                             'e.g. 8704 for 16 neighbours x 512 genes).')
    parser.add_argument('--expect_storage_format',
                        choices=STORAGE_FORMATS,
                        default=None,
                        help="Refuse parts of another storage format "
                             "('compact': no padding stored; 'padded'). "
                             'Default: any, as long as all parts agree.')
    parser.add_argument('--allow_parts_without_record',
                        action='store_true',
                        help='Concatenate parts without '
                             'terra_tokenizer_params.json (tokenized before '
                             'records were written) into a corpus without a '
                             'record. Default: refuse them, as such a part '
                             'may be an incomplete tokenization.')
    parser.add_argument('--num_proc',
                        type=int,
                        default=None,
                        help='Processes writing the corpus shards '
                             '(default: one).')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Replace an existing corpus: the new one is '
                             'written next to it, swapped in when complete, '
                             'and the old one deleted (disk space for both '
                             'is needed until the swap). Default: refuse an '
                             'existing corpus.')
    args = parser.parse_args()
    
    main(args)
