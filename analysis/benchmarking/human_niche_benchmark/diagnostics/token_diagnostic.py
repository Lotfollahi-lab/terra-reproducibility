"""Which genes fill TERRA's per-cell token budget?

Reproduces the TERRA tokenizer's gene ranking for model 01112025_081731_614
(``harmonize_adata`` with the model bundle's gene files, genes restricted to the
token vocabulary, ranked by raw count with ``rank_gene_tokens``; no cell- or
gene-level rank normalization in this model's config) and the per-cell budget
of ``model_input_size / (n_neighs + 1)`` = 2816 / 11 = 256 tokens for a cell
with 10 neighbours. For every gene it reports how often it occupies the top
ranks and how niche- (and section-) specific its expression is, to test whether
ubiquitous, high-count genes dominate the tokens.

Outputs (in ``<results>/token_diagnostics/``):
- ``{dataset}_gene_token_stats.csv``  one row per tokenized gene
- ``{dataset}_token_summary.json``    dataset-level summary

Run in the terra environment (token_diagnostic.lsf).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from terra.inference.harmonize import harmonize_adata
from terra.tokenizers.tokenize import rank_gene_tokens

sys.path.append(str(Path(__file__).resolve().parents[1]))
from bench_utils import (SECTION_KEY, TERRA_MODEL_FOLDER_PATH,  # noqa: E402
                         load_benchmark_dataset, method_output_dir)

TOP_K = 10   # "top of the sequence"
# Informal list of broadly expressed genes, only used to annotate the tables
UBIQUITOUS_GENES = {'ACTB', 'ACTG1', 'GAPDH', 'B2M', 'MALAT1', 'NEAT1', 'EEF1A1',
                    'EEF2', 'TPT1', 'FTL', 'FTH1', 'TMSB4X', 'TMSB10', 'UBC',
                    'UBB', 'PPIA', 'HSP90AB1', 'HSPA8', 'PTMA', 'MYL6', 'TPI1',
                    'PGK1', 'LDHA', 'ENO1', 'CD74', 'HLA-A', 'HLA-B', 'HLA-C'}


def is_ubiquitous_name(name):
    return name in UBIQUITOUS_GENES or name.startswith(('RPL', 'RPS', 'MT-'))


def eta_squared(X, groups):
    """Per gene: fraction of the expression variance explained by ``groups``."""
    codes, _ = pd.factorize(groups)
    G = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))))
    n_g = np.asarray(G.sum(1)).ravel()[:, None]
    sums = np.asarray((G @ X).todense())
    total_mean = sums.sum(0) / len(codes)
    between = (n_g * (sums / n_g - total_mean) ** 2).sum(0)
    total = np.asarray(X.multiply(X).sum(0)).ravel() - len(codes) * total_mean ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(total > 0, between / total, 0.0), sums / n_g


def main(args):
    out_dir = method_output_dir('token_diagnostics')
    model = Path(TERRA_MODEL_FOLDER_PATH)
    config = yaml.safe_load(open(model / 'model_config.yaml'))['data']
    budget = int(config['model_input_size'] / (config['n_neighs'] + 1))
    token_dict = pickle.load(open(model / 'token_dictionary.pkl', 'rb'))

    adata = load_benchmark_dataset(args.dataset)
    n_panel = adata.n_vars
    adata = harmonize_adata(
        adata,
        gene_mapping_dict_file_path=str(model / 'ensembl_dictionary.pkl'),
        gene_occurrence_count_file_path=str(model / 'gene_count_dictionary.pkl'),
        min_cells_per_gene=0, min_genes_per_cell=0)
    n_harmonized = adata.n_vars
    in_vocab = np.array(['ENS' in e and e in token_dict
                         for e in adata.var['ensembl_id']])
    adata = adata[:, in_vocab].copy()
    X = sp.csr_matrix(adata.X, dtype=np.float32)
    X.sort_indices()
    genes = adata.var_names.to_numpy()
    n_cells, n_genes = X.shape
    print(f'{args.dataset}: panel {n_panel}, harmonized {n_harmonized}, '
          f'tokenized {n_genes} genes, {n_cells} cells, budget {budget}')

    # Rank genes per cell exactly as the tokenizer does (sparse path)
    in_top = np.zeros(n_genes)
    in_budget = np.zeros(n_genes)
    rank_sum = np.zeros(n_genes)
    nnz = np.diff(X.indptr)
    truncated = nnz > budget
    tie_at_cut = np.zeros(n_cells, dtype=bool)
    count1_in_budget = np.zeros(n_cells)
    for i in range(n_cells):
        lo, hi = X.indptr[i], X.indptr[i + 1]
        data, idx = X.data[lo:hi], X.indices[lo:hi]
        order = rank_gene_tokens(data, np.arange(len(data)))
        ranked = idx[order]
        in_top[ranked[:TOP_K]] += 1
        in_budget[ranked[:budget]] += 1
        rank_sum[ranked] += np.arange(len(ranked))
        if truncated[i]:
            vals = data[order]
            tie_at_cut[i] = vals[budget - 1] == vals[budget]
            count1_in_budget[i] = np.mean(vals[:budget] == 1)

    detection = np.bincount(X.indices, minlength=n_genes) / n_cells
    mean_count = np.asarray(X.mean(0)).ravel()

    # Specificity on log1p(CP10k) of the tokenized genes
    depth = np.asarray(X.sum(1)).ravel()
    L = sp.diags(1e4 / np.maximum(depth, 1)) @ X
    L.data = np.log1p(L.data)
    eta_niche, niche_means = eta_squared(L, adata.obs['niche_gt'].astype(str).values)
    eta_section, _ = eta_squared(L, adata.obs[SECTION_KEY].astype(str).values)
    niches = pd.factorize(adata.obs['niche_gt'].astype(str).values)[1]
    overall = np.asarray(L.mean(0)).ravel()

    stats = pd.DataFrame(dict(
        gene=genes, ensembl_id=adata.var['ensembl_id'].values,
        detection_rate=detection, mean_count=mean_count,
        frac_cells_top10=in_top / n_cells, frac_cells_in_budget=in_budget / n_cells,
        mean_rank_when_expressed=np.divide(rank_sum, detection * n_cells,
                                           out=np.full(n_genes, np.nan),
                                           where=detection > 0),
        eta2_niche=eta_niche, eta2_section=eta_section,
        top_niche=niches[niche_means.argmax(0)],
        top_niche_fold=np.divide(niche_means.max(0), overall,
                                 out=np.full(n_genes, np.nan), where=overall > 0),
        ubiquitous_name=[is_ubiquitous_name(g) for g in genes],
    )).sort_values('frac_cells_top10', ascending=False)
    stats.to_csv(out_dir / f'{args.dataset}_gene_token_stats.csv', index=False)

    def token_weighted(col, weights):
        return float(np.average(stats[col], weights=weights)) if weights.sum() else None

    expressed = stats['detection_rate'] * n_cells
    summary = dict(
        dataset=args.dataset, n_cells=n_cells, n_panel_genes=n_panel,
        n_harmonized_genes=n_harmonized, n_tokenized_genes=n_genes,
        budget_per_cell=budget,
        nnz_per_cell_quantiles={q: float(np.quantile(nnz, q))
                                for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        frac_cells_truncated=float(truncated.mean()),
        frac_truncated_cells_tie_at_cut=(float(tie_at_cut[truncated].mean())
                                         if truncated.any() else None),
        mean_frac_count1_in_budget_truncated=(float(count1_in_budget[truncated].mean())
                                              if truncated.any() else None),
        # Average specificity of the genes that occupy tokens, vs. all expressed
        # genes (weights = token occurrences)
        eta2_niche_top10=token_weighted('eta2_niche', stats['frac_cells_top10']),
        eta2_niche_budget=token_weighted('eta2_niche', stats['frac_cells_in_budget']),
        eta2_niche_expressed=token_weighted('eta2_niche', expressed),
        eta2_niche_panel_median=float(stats['eta2_niche'].median()),
        eta2_section_top10=token_weighted('eta2_section', stats['frac_cells_top10']),
        eta2_section_budget=token_weighted('eta2_section', stats['frac_cells_in_budget']),
        eta2_section_expressed=token_weighted('eta2_section', expressed),
        frac_top10_tokens_ubiquitous_name=float(
            stats.loc[stats['ubiquitous_name'], 'frac_cells_top10'].sum()
            / stats['frac_cells_top10'].sum()),
        frac_top10_tokens_detected_gt50pct=float(
            stats.loc[stats['detection_rate'] > 0.5, 'frac_cells_top10'].sum()
            / stats['frac_cells_top10'].sum()),
        spearman_top10_vs_eta2_niche=float(
            stats['frac_cells_top10'].corr(stats['eta2_niche'], method='spearman')),
        spearman_top10_vs_detection=float(
            stats['frac_cells_top10'].corr(stats['detection_rate'], method='spearman')),
        top20_genes_top10=stats.head(20)[['gene', 'frac_cells_top10',
                                          'detection_rate', 'eta2_niche',
                                          'eta2_section']].round(3)
                                         .to_dict(orient='records'),
    )
    json.dump(summary, open(out_dir / f'{args.dataset}_token_summary.json', 'w'),
              indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != 'top20_genes_top10'},
                     indent=2))
    print(stats.head(20).round(3).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    main(parser.parse_args())
