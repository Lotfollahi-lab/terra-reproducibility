"""
This script converts CAMEO-Lung (theislab/CAMEO-Lung, downloaded by download_cameo_data.py) into silver-tier
AnnData files, one per sample (section), for cell-level niche benchmarking.

CAMEO-Lung stores 224 x 224 px H&E patches ("niches") of the LungPF Xenium cohort (Vannan et al., Nat. Genet. 2025;
GEO GSE250346), each with the cells inside it (raw counts, 343 genes), the cells' centroids and one pathologist niche
label. Steps:

1. Gene names. CAMEO stores no gene names. They are recovered from the GEO transcripts of one sample: cells are matched
   between CAMEO and GEO by their sorted count vector (invariant to gene order), and CAMEO column j is gene g if the
   counts agree in every matched cell (whole-cell counts, qv >= 20). The match must be exact and one-to-one. The
   recovered names are the corrected ones of the published data; CAMEO's column order is the alphabetical order of
   the pre-publication names, of which 35 were swapped (GSE250346_RDS_README.txt).
2. Cells. Cells on patch borders appear in several patches; they are deduplicated by (centroid, counts). A cell gets
   its patches' label; cells whose patches disagree, or are 'NOANNOT', keep no label (NaN; dropped by the benchmark
   loader but available as spatial context).
3. Samples. CAMEO's 'VUILD110' is an exact copy of 'THD0008' (same cells, coordinates and labels; its centroids match
   the GEO pathology annotations of THD0008, not of VUILD110), so it is left out: 22 samples.

Centroids are in microns (CAMEO stores them truncated to integers; + 0.5 is added). Output:
`<output_directory_path>/<dataset_name>/adata_batch<k>.h5ad` (samples in alphabetical order) and `samples.csv`.

Usage:
```bash
bsub < process_cameo_lung.lsf
```
"""

import argparse
import collections
import glob
import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp

DUPLICATE_SAMPLES = {'VUILD110': 'THD0008'}  # CAMEO sample -> sample it is a copy of
NO_LABEL = 'NOANNOT'
# Panel names that are outdated in the gene-name -> Ensembl ID dictionary, and Ensembl IDs of panel genes that are
# not in it at all (not in TERRA's vocabulary).
RENAMED_GENES = {'C20orf85': 'CIMIP1', 'HIST1H1C': 'H1-2'}
EXTRA_ENSEMBL_IDS = {'MEG3': 'ENSG00000214548', 'TRAC': 'ENSG00000277734'}
CONTROL_PREFIXES = ('NegControl', 'UnassignedCodeword', 'BLANK', 'Unassigned')


def read_samples(cameo_path, samples=None):
    """Valid cells of every patch, per sample: counts, centroids, label and patch index."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    label_names = None
    for file in sorted(Path(cameo_path).glob('data-*.arrow')):
        with pa.memory_map(str(file)) as source:
            table = pa.ipc.open_stream(source).read_all()
        if label_names is None:
            label_names = json.loads(table.schema.metadata[b'huggingface'])['info']['features']['annotation']['names']
        if samples is not None:
            table = table.filter(pc.is_in(table.column('name'), pa.array(samples)))
        table = table.select(['name', 'annotation', 'mask', 'cell_coords', 'gexp'])
        for name, label, mask, coords, counts in zip(*(table.column(c).to_pylist() for c in table.column_names)):
            mask = np.array(mask)
            d = out[name]
            d['X'].append(np.array(counts, dtype=np.int32)[mask])
            d['coords'].append(np.array(coords, dtype=np.int32)[mask])
            d['label'].append(np.full(mask.sum(), label))
            d['patch'].append(np.full(mask.sum(), len(d['X']) - 1))
        print(f'{file.name}: {table.num_rows} patches', flush=True)
    return {name: {k: np.concatenate(v) for k, v in d.items()} for name, d in out.items()}, label_names


def signatures(X):
    return [tuple(np.sort(row)[::-1][:40]) + (int(row.sum()),) for row in X]


def recover_gene_names(cells, transcripts_path):
    """CAMEO column -> gene name, from one sample's GEO transcripts (exact, one-to-one match required)."""
    tx = pd.read_csv(transcripts_path, usecols=['cell_id', 'feature_name', 'qv'])
    tx = tx[(tx.cell_id != 'UNASSIGNED') & (tx.qv >= 20) & ~tx.feature_name.str.startswith(CONTROL_PREFIXES)]
    geo = tx.groupby(['cell_id', 'feature_name']).size().unstack(fill_value=0)
    genes = geo.columns.values
    G = geo.values.astype(np.int32)

    _, keep = np.unique(np.hstack([cells['coords'], cells['X']]), axis=0, return_index=True)
    X = cells['X'][keep]
    sig_c, sig_g = signatures(X), signatures(G)
    count_c, count_g = collections.Counter(sig_c), collections.Counter(sig_g)
    index_g = {s: i for i, s in enumerate(sig_g) if count_g[s] == 1}
    pairs = [(i, index_g[s]) for i, s in enumerate(sig_c)
             if count_c[s] == 1 and s in index_g and X[i].sum() >= 30]
    ci, gi = map(np.array, zip(*pairs))
    agreement = (X[ci][:, :, None] == G[gi][:, None, :]).mean(0)  # CAMEO column x GEO gene
    best = agreement.argmax(1)
    if not (np.all(agreement.max(1) == 1) and len(set(best)) == len(best) == len(genes)):
        raise ValueError('CAMEO columns do not match the GEO genes exactly and one-to-one.')
    print(f'Gene names recovered from {len(pairs)} matched cells (of {len(X)}).', flush=True)
    return genes[best]


def main(args):
    output_path = Path(args.output_directory_path) / args.dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    data, label_names = read_samples(args.cameo_path)
    print(f'CAMEO samples: {sorted(data)}', flush=True)

    reference = next(iter(glob.glob(f'{args.geo_path}/GSM*_{args.reference_sample}_*transcripts.csv.gz')))
    gene_names = recover_gene_names(data[args.reference_sample], reference)
    name_to_ensembl = pickle.load(open(args.gene_mapping_dict_file_path, 'rb'))
    gene_names = np.array([RENAMED_GENES.get(g, g) for g in gene_names])
    ensembl_ids = np.array([name_to_ensembl.get(g, EXTRA_ENSEMBL_IDS.get(g)) for g in gene_names])
    if any(e is None for e in ensembl_ids):
        raise ValueError(f'No Ensembl ID for {gene_names[[e is None for e in ensembl_ids]]}.')
    var = pd.DataFrame({'gene_name': gene_names, 'ensembl_id': ensembl_ids}, index=gene_names)
    var.index.name = None

    for copy, original in DUPLICATE_SAMPLES.items():
        same = all(np.array_equal(data[copy][k], data[original][k]) for k in ['X', 'coords', 'label'])
        if not same:
            raise ValueError(f'{copy} is no longer a copy of {original}; check DUPLICATE_SAMPLES.')
        del data[copy]

    summary = []
    for k, sample in enumerate(sorted(data)):
        d = data[sample]
        key = np.hstack([d['coords'], d['X']])
        uniq, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.ravel()
        # One label per unique cell if all its patches agree, else none
        n_labels = np.zeros(len(uniq), dtype=int)
        lab = np.full(len(uniq), -1)
        for cell, label in set(zip(inverse.tolist(), d['label'].tolist())):
            n_labels[cell] += 1
            lab[cell] = label
        niche = np.array([label_names[l] for l in lab], dtype=object)
        niche[(n_labels > 1) | (niche == NO_LABEL)] = np.nan
        n_patches = np.bincount(inverse, minlength=len(uniq))

        obs = pd.DataFrame({
            'cell_id': [f'{args.dataset_id}_batch{k}_{i}' for i in range(len(uniq))],
            'sample': sample,
            'niche': pd.Categorical(niche),
            'n_patches': n_patches,
            'conflicting_patch_labels': n_labels > 1})
        obs.index = obs['cell_id'].values
        adata = ad.AnnData(X=sp.csr_matrix(d['X'][first].astype(np.int64)), obs=obs, var=var.copy())
        adata.layers['counts'] = adata.X.copy()
        adata.obsm['spatial'] = d['coords'][first].astype(float) + 0.5
        adata.uns.update(dataset_id=args.dataset_id, assay='Xenium', species='human', tissue='lung',
                         source='theislab/CAMEO-Lung', sample=sample)
        adata.write_h5ad(output_path / f'adata_batch{k}.h5ad', compression='gzip')
        labelled = obs['niche'].notna()
        summary.append(dict(batch=k, sample=sample, n_patch_rows=len(d['X']), n_cells=len(uniq),
                            n_labelled=int(labelled.sum()), n_conflicting=int((n_labels > 1).sum()),
                            n_noannot=int(((n_labels == 1) & (lab == label_names.index(NO_LABEL))).sum())))
        print(summary[-1], flush=True)

    summary = pd.DataFrame(summary)
    summary.to_csv(output_path / 'samples.csv', index=False)
    print(summary.sum(numeric_only=True).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert CAMEO-Lung to silver-tier AnnData files.')
    parser.add_argument('--cameo_path', type=str, default='/nfs/team361/sb75/DATASETS/bronze/cameo/CAMEO-Lung',
                        help='CAMEO-Lung folder (datasets.save_to_disk format).')
    parser.add_argument('--geo_path', type=str, default='/nfs/team361/me13/DATASETS/bronze/GSE250346',
                        help='Folder with the GEO transcripts of --reference_sample.')
    parser.add_argument('--reference_sample', type=str, default='VUHD095',
                        help='Sample whose GEO transcripts are used to recover the gene names.')
    parser.add_argument('--gene_mapping_dict_file_path', type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS/genes/'
                                'homo_sapiens_gene_name_to_ensembl_id_dict.pkl')
    parser.add_argument('--output_directory_path', type=str, default='/nfs/team361/me13/DATASETS/silver')
    parser.add_argument('--dataset_name', type=str, default='xhl3001-22b_1p')
    parser.add_argument('--dataset_id', type=str, default='3001')
    main(parser.parse_args())
