"""
This script converts the segmented 10x Visium HD human lung adenocarcinoma section (FFPE, "Visium_HD_Human_Lung_Cancer
_HD_Only", spaceranger 3.0.0; one cell per segmented object, counts summed over its 2-um bins) into a silver-tier AnnData
file for cell-level niche benchmarking.

- Counts: the input X holds non-integer per-cell counts (bin counts after destriping, summed per cell). They are
  rounded to the nearest integer so that count-based methods (scVI in CellCharter, NicheCompass) get counts; the
  unrounded values are kept in layers['unrounded'].
- Niche: obs['region_annotation'] (pathologist regions); 'Unassigned' cells keep no label (NaN; dropped by the
  benchmark loader).
- Coordinates: obsm['spatial'] is in full-resolution image pixels; it is converted to microns with the section's
  microns_per_pixel scale factor.

Output: `<output_directory_path>/<dataset_name>/adata_batch0.h5ad`.

Usage:
```bash
bsub < process_visium_hd_luad.lsf
```
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

NO_LABEL = 'Unassigned'


def main(args):
    output_path = Path(args.output_directory_path) / args.dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.input_file_path)
    print(adata, flush=True)

    library = next(iter(adata.uns['spatial']))
    microns_per_pixel = float(adata.uns['spatial'][library]['scalefactors']['microns_per_pixel'])

    X = sp.csr_matrix(adata.X)
    counts = X.copy()
    counts.data = np.rint(counts.data)
    counts.eliminate_zeros()
    counts = counts.astype(np.int64)
    print(f'Rounded counts: {X.nnz} -> {counts.nnz} non-zero entries; total counts {X.sum():.0f} -> '
          f'{counts.sum()}.', flush=True)

    if not adata.var_names.is_unique:
        raise ValueError('Gene names are not unique.')
    var = pd.DataFrame({'gene_name': adata.var_names.values,
                        'ensembl_id': adata.var['gene_ids'].astype(str).values}, index=adata.var_names.values)
    var.index.name = None

    niche = adata.obs['region_annotation'].astype(str).replace({NO_LABEL: np.nan})
    obs = pd.DataFrame({
        'cell_id': [f'{args.dataset_id}_batch0_{i}' for i in range(adata.n_obs)],
        'object_id': adata.obs['object_id'].values,
        'bin_count': adata.obs['bin_count'].values,
        'sample': library,
        'niche': pd.Categorical(niche)})
    obs.index = obs['cell_id'].values

    out = ad.AnnData(X=counts, obs=obs, var=var)
    out.layers['counts'] = out.X.copy()
    out.layers['unrounded'] = X
    out.obsm['spatial'] = np.asarray(adata.obsm['spatial'], dtype=float) * microns_per_pixel
    out.uns.update(dataset_id=args.dataset_id, assay='Visium HD', species='human', tissue='lung',
                   source=str(args.input_file_path), sample=library, microns_per_pixel=microns_per_pixel)
    out.write_h5ad(output_path / 'adata_batch0.h5ad', compression='gzip')
    print(out.obs['niche'].value_counts(dropna=False).to_string(), flush=True)
    print(f'Wrote {output_path / "adata_batch0.h5ad"}: {out.n_obs} cells, {out.n_vars} genes.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert the Visium HD LUAD section to a silver-tier AnnData file.')
    parser.add_argument('--input_file_path', type=str, default='/nfs/team361/me13/playground/preprocessed_LUAD.h5ad')
    parser.add_argument('--output_directory_path', type=str, default='/nfs/team361/me13/DATASETS/silver')
    parser.add_argument('--dataset_name', type=str, default='vhl3002-1b_1p')
    parser.add_argument('--dataset_id', type=str, default='3002')
    main(parser.parse_args())
