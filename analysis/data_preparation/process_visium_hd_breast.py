"""
This script converts the 8-um bins of the 10x Visium HD human breast cancer sections (samples P26, P29, P32) into
silver-tier AnnData files, one per sample, for niche benchmarking at bin level.

- Bins: in-tissue 8-um bins with at least --min_counts counts. Sample P27 is left out (low quality: median 33 counts per
  16-um bin, against 520-800 for the other samples).
- Niche: the pathologist annotation of the 8-um bin (`<sample>_HD_pathology_annotations.csv`); bins without one keep no
  label (NaN; dropped by the benchmark loader).
- Counts: X (raw integer counts) as is; the input's log-normalized layer is not kept.
- Coordinates: obsm['spatial'] is in full-resolution image pixels; it is converted to microns with the section's
  microns_per_pixel scale factor.

Output: `<output_directory_path>/<dataset_name>/adata_batch<k>.h5ad` (samples in --samples order) and `samples.csv`.

Usage:
```bash
bsub < process_visium_hd_breast.lsf
```
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def main(args):
    output_path = Path(args.output_directory_path) / args.dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    summary = []
    for k, sample in enumerate(args.samples):
        adata = ad.read_h5ad(Path(args.anndata_folder_path) / f'{sample}_HD.h5ad')
        print(sample, adata, flush=True)
        X = sp.csr_matrix(adata.X)
        if not np.all(X.data == np.round(X.data)):
            raise ValueError(f'{sample}: X does not hold integer counts.')
        total_counts = np.asarray(X.sum(1)).ravel()
        keep = (adata.obs['in_tissue'].values == 1) & (total_counts >= args.min_counts)

        annotation = pd.read_csv(Path(args.annotation_folder_path) / f'{sample}_HD_pathology_annotations.csv',
                                 index_col=0)['annotation']
        niche = annotation.reindex(adata.obs_names)

        library = next(iter(adata.uns['spatial']))
        microns_per_pixel = float(adata.uns['spatial'][library]['scalefactors']['microns_per_pixel'])

        idx = np.where(keep)[0]
        obs = pd.DataFrame({
            'cell_id': [f'{args.dataset_id}_batch{k}_{i}' for i in range(len(idx))],
            'barcode': adata.obs_names[idx].values,
            'sample': sample,
            'array_row': adata.obs['array_row'].values[idx],
            'array_col': adata.obs['array_col'].values[idx],
            'total_counts': total_counts[idx],
            'niche': pd.Categorical(niche.values[idx])})
        obs.index = obs['cell_id'].values
        var = pd.DataFrame({'gene_name': adata.var_names.values,
                            'ensembl_id': adata.var['gene_ids'].astype(str).values}, index=adata.var_names.values)
        var.index.name = None
        if not var.index.is_unique:
            raise ValueError(f'{sample}: gene names are not unique.')

        out = ad.AnnData(X=X[idx].astype(np.int64), obs=obs, var=var)
        out.layers['counts'] = out.X.copy()
        out.obsm['spatial'] = np.asarray(adata.obsm['spatial'], dtype=float)[idx] * microns_per_pixel
        out.uns.update(dataset_id=args.dataset_id, assay='Visium HD', bin_size_um=8, species='human',
                       tissue='breast', sample=sample, microns_per_pixel=microns_per_pixel,
                       min_counts=args.min_counts)
        out.write_h5ad(output_path / f'adata_batch{k}.h5ad', compression='gzip')
        summary.append(dict(batch=k, sample=sample, n_bins=adata.n_obs, n_kept=len(idx),
                            n_labelled=int(obs['niche'].notna().sum())))
        print(summary[-1], obs['niche'].value_counts(dropna=False).to_dict(), flush=True)

    pd.DataFrame(summary).to_csv(output_path / 'samples.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Visium HD breast 8-um bins to silver-tier AnnData files.')
    parser.add_argument('--anndata_folder_path', type=str,
                        default='/nfs/team361/me13/playground/anndata_objects/visium_hd/square_008um')
    parser.add_argument('--annotation_folder_path', type=str,
                        default='/nfs/team361/me13/playground/pathology_annotations/visium_hd')
    parser.add_argument('--samples', nargs='+', default=['P26', 'P29', 'P32'])
    parser.add_argument('--min_counts', type=int, default=50)
    parser.add_argument('--output_directory_path', type=str, default='/nfs/team361/me13/DATASETS/silver')
    parser.add_argument('--dataset_name', type=str, default='vhb3003-3b_1p')
    parser.add_argument('--dataset_id', type=str, default='3003')
    main(parser.parse_args())
