"""
Harmonize a raw-count spatial AnnData for a model bundle and tokenize it once,
with the base TERRA tree, as the common input of the embedding benchmarks.

usage: PYTHONPATH=<base tree>/src python prepare_data.py --h5ad FILE
           --bundle DIR --out DIR [--n-cells 20000]
           [--sample-key KEY --sample VALUE] [--nproc 4]

The AnnData needs raw counts in X (or in layers['counts']), gene symbols or
Ensembl IDs as var_names, and coordinates in obsm['spatial']. With
--sample-key/--sample one section is selected; without them the file should
hold one section. Cells without finite coordinates are dropped. With
--n-cells the cells closest to the median position are kept (a contiguous
patch, so that the neighborhoods stay realistic).

Writes DIR/harmonized.h5ad, DIR/dataset (tokenized Hugging Face dataset) and,
last, DIR/prepare.json, which records the inputs. If DIR/prepare.json records
the same inputs, nothing is done; otherwise DIR is prepared anew, so that a
changed input or an interrupted preparation is never reused.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile


def requested_inputs(args):
    """Everything the prepared data depends on."""
    stat = os.stat(args.h5ad)
    commit = None
    if os.environ.get('TERRA_TREE'):
        try:
            commit = subprocess.run(
                ['git', '-C', os.environ['TERRA_TREE'], 'rev-parse', 'HEAD'],
                capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return {'h5ad': os.path.realpath(args.h5ad),
            'h5ad_size': stat.st_size,
            'h5ad_mtime': int(stat.st_mtime),
            'sample_key': args.sample_key,
            'sample': args.sample,
            'n_cells': args.n_cells,
            'bundle': os.path.realpath(args.bundle),
            'terra_commit': commit}


def bundle_file(bundle, *names):
    for name in names:
        path = os.path.join(bundle, name)
        if os.path.isfile(path):
            return path
    return None


def select_cells(adata, args, n_neighs):
    """The section and the central patch of --n-cells cells."""
    import numpy as np
    import pandas as pd

    if args.sample_key:
        column = adata.obs[args.sample_key]
        keep = (column.astype(str) == str(args.sample)).to_numpy()
        if not keep.any():
            # e.g. a float column (1.0, 2.0) and SAMPLE=1
            try:
                keep = (pd.to_numeric(column, errors='coerce')
                        == float(args.sample)).to_numpy()
            except ValueError:
                pass
        if not keep.any():
            values = column.astype(str).unique().tolist()
            raise SystemExit(
                f"No cells with obs['{args.sample_key}'] == {args.sample!r}. "
                f"Values: {', '.join(values[:20])}"
                f"{', ...' if len(values) > 20 else ''}")
        adata = adata[keep].copy()
    coords = np.asarray(adata.obsm['spatial'], dtype=float)
    finite = np.isfinite(coords).all(axis=1)
    if not finite.all():
        print(f'Dropping {(~finite).sum()} cells without finite coordinates')
        adata, coords = adata[finite].copy(), coords[finite]
    if args.n_cells and adata.n_obs > args.n_cells:
        distance = np.linalg.norm(coords - np.median(coords, axis=0), axis=1)
        keep = np.sort(np.argsort(distance, kind='stable')[:args.n_cells])
        adata = adata[keep].copy()
    if adata.n_obs <= n_neighs:
        raise SystemExit(f'{adata.n_obs} cells selected, but the model needs '
                         f'more than {n_neighs} (its number of neighbors).')
    adata.obs_names_make_unique()
    return adata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5ad', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--n-cells', type=int, default=20000)
    parser.add_argument('--sample-key')
    parser.add_argument('--sample')
    parser.add_argument('--nproc', type=int, default=4)
    args = parser.parse_args()

    inputs = requested_inputs(args)
    record = os.path.join(args.out, 'prepare.json')
    if os.path.isfile(record):
        with open(record) as f:
            done = json.load(f)
        if done.get('inputs') == inputs:
            print(f'{args.out} is up to date: {done["n_cells"]} cells')
            return
    # Prepare anew: nothing of an older or interrupted preparation is kept
    os.makedirs(args.out, exist_ok=True)
    for name in ('prepare.json', 'dataset', 'dataset.tmp', 'harmonized.h5ad'):
        path = os.path.join(args.out, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)

    import anndata as ad
    import yaml

    from terra import harmonize_adata, tokenize_adata

    with open(os.path.join(args.bundle, 'model_config.yaml')) as f:
        n_neighs = yaml.safe_load(f)['data']['n_neighs']
    adata = select_cells(ad.read_h5ad(args.h5ad), args, n_neighs)
    print(f'{adata.n_obs} cells x {adata.n_vars} genes')

    # As harmonize_tokenize_embed_pipeline: the bundle's gene references,
    # no cell or gene filter
    harmonized = harmonize_adata(
        adata,
        gene_mapping_dict_file_path=bundle_file(
            args.bundle, 'ensembl_dictionary.pkl', 'ensembl_dic.pkl'),
        gene_occurrence_count_file_path=bundle_file(
            args.bundle, 'gene_count_dictionary.pkl'),
        min_genes_per_cell=0,
        min_cells_per_gene=0)
    harmonized.write_h5ad(os.path.join(args.out, 'harmonized.h5ad'))

    # The tokenization cache goes to local disk (TMPDIR)
    cache = tempfile.mkdtemp(prefix='terra_prepare_cache_')
    try:
        dataset = tokenize_adata(
            harmonized.copy(), model_folder_path=args.bundle,
            cache_directory_path=cache, nproc=args.nproc)
        dataset.save_to_disk(os.path.join(args.out, 'dataset.tmp'))
        os.rename(os.path.join(args.out, 'dataset.tmp'),
                  os.path.join(args.out, 'dataset'))
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    with open(record, 'w') as f:
        json.dump({'inputs': inputs, 'n_cells': harmonized.n_obs,
                   'n_genes_input': adata.n_vars,
                   'n_genes_harmonized': harmonized.n_vars}, f, indent=2)
    print(f'tokenized: {dataset.num_rows} rows -> {args.out}/dataset')


if __name__ == '__main__':
    main()
