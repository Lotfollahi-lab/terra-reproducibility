"""
Time tokenize_adata of one TERRA source tree on a harmonized AnnData and save
the tokenized dataset (run), or compare two saved datasets column by column
(compare).

usage: PYTHONPATH=<tree>/src python bench_tokenize.py run --harmonized FILE
           --bundle DIR --out DIR [--nproc 4] [--reps 2]
       python bench_tokenize.py compare DIR_A DIR_B

The tree is chosen with PYTHONPATH=<tree>/src; with TERRA_TREE=<tree>/src set
as well, the script stops if `terra` was imported from anywhere else. With the
tokenization prototype (variant 'tok'), EFFREAD_FORMAT_IN_GEN=1 also formats
the examples inside the dataset generator, and EFFREAD_FINGERPRINT=1 in
addition gives the generated dataset a random fingerprint instead of hashing
the generator. Every repetition uses a fresh cache directory, so no repetition
reads another one's cache.
"""
import argparse
import json
import os
import shutil
import tempfile
import time


def run(args):
    import anndata as ad

    from bench_embed import check_terra_tree
    from terra import tokenize_adata

    terra_path = check_terra_tree()
    adata = ad.read_h5ad(args.harmonized)
    os.makedirs(args.out, exist_ok=True)
    times = []
    for rep in range(args.reps):
        cache = tempfile.mkdtemp(prefix='cache_', dir=args.out)
        start = time.perf_counter()
        # As harmonize_tokenize_embed_pipeline (one sample)
        dataset = tokenize_adata(
            adata.copy(), model_folder_path=args.bundle,
            cache_directory_path=cache, nproc=args.nproc)
        times.append(time.perf_counter() - start)
        print(f'rep {rep}: {times[-1]:.1f} s for {dataset.num_rows} cells',
              flush=True)
        if rep == args.reps - 1:
            target = os.path.join(args.out, 'dataset')
            shutil.rmtree(target, ignore_errors=True)
            dataset.save_to_disk(target)
        del dataset
        shutil.rmtree(cache, ignore_errors=True)
    with open(os.path.join(args.out, 'timing.json'), 'w') as f:
        json.dump({'terra': terra_path, 'n_cells': adata.n_obs,
                   'nproc': args.nproc,
                   'format_in_generator':
                       os.environ.get('EFFREAD_FORMAT_IN_GEN') == '1',
                   'random_fingerprint':
                       os.environ.get('EFFREAD_FINGERPRINT') == '1',
                   'seconds': times}, f, indent=2)


def compare_datasets(dir_a, dir_b):
    """Compare two saved datasets: rows, column names and order, features,
    and every Arrow column (exact)."""
    from datasets import load_from_disk

    a, b = load_from_disk(dir_a), load_from_disk(dir_b)
    table_a, table_b = a.data.table, b.data.table
    differing = [name for name in a.column_names
                 if name not in b.column_names
                 or not table_a.column(name).equals(table_b.column(name))]
    result = {
        'rows': [a.num_rows, b.num_rows],
        'same_columns_in_same_order': a.column_names == b.column_names,
        'same_features': a.features == b.features,
        'differing_columns': differing,
    }
    result['identical'] = (a.num_rows == b.num_rows
                           and result['same_columns_in_same_order']
                           and result['same_features'] and not differing)
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--harmonized', required=True)
    run_parser.add_argument('--bundle', required=True)
    run_parser.add_argument('--out', required=True)
    run_parser.add_argument('--nproc', type=int, default=4)
    run_parser.add_argument('--reps', type=int, default=2)
    compare_parser = sub.add_parser('compare')
    compare_parser.add_argument('dir_a')
    compare_parser.add_argument('dir_b')
    args = parser.parse_args()
    if args.command == 'run':
        run(args)
    else:
        print(json.dumps(compare_datasets(args.dir_a, args.dir_b), indent=2))


if __name__ == '__main__':
    main()
