"""
Harmonize a raw-count spatial AnnData for a model bundle and tokenize it once,
with the base TERRA tree, as the common input of the embedding benchmarks.

usage: PYTHONPATH=<base tree>/src python prepare_data.py --h5ad FILE
           --bundle DIR --out DIR [--n-cells 20000]
           [--sample-key KEY --sample VALUE] [--nproc 4]

The AnnData needs raw counts in X (or in layers['counts']), gene symbols or
Ensembl IDs as var_names, and coordinates in obsm['spatial']. With
--sample-key/--sample one section is selected; without them the file should
hold one section. With --n-cells the cells closest to the median position are
kept (a contiguous patch, so that the neighborhoods stay realistic).

Writes DIR/harmonized.h5ad and DIR/dataset (tokenized Hugging Face dataset).
"""
import argparse
import os
import shutil

import anndata as ad
import numpy as np

from terra import harmonize_adata, tokenize_adata


def bundle_file(bundle, *names):
    for name in names:
        path = os.path.join(bundle, name)
        if os.path.isfile(path):
            return path
    return None


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

    adata = ad.read_h5ad(args.h5ad)
    if args.sample_key:
        adata = adata[adata.obs[args.sample_key].astype(str)
                      == str(args.sample)].copy()
    if args.n_cells and adata.n_obs > args.n_cells:
        coords = np.asarray(adata.obsm['spatial'], dtype=float)
        distance = np.linalg.norm(coords - np.median(coords, axis=0), axis=1)
        keep = np.sort(np.argsort(distance, kind='stable')[:args.n_cells])
        adata = adata[keep].copy()
    adata.obs_names_make_unique()
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
    os.makedirs(args.out, exist_ok=True)
    harmonized.write_h5ad(os.path.join(args.out, 'harmonized.h5ad'))

    cache = os.path.join(args.out, 'cache')
    dataset = tokenize_adata(
        harmonized.copy(), model_folder_path=args.bundle,
        cache_directory_path=cache, nproc=args.nproc)
    dataset.save_to_disk(os.path.join(args.out, 'dataset'))
    shutil.rmtree(cache, ignore_errors=True)
    print(f'tokenized: {dataset.num_rows} rows -> {args.out}/dataset')


if __name__ == '__main__':
    main()
