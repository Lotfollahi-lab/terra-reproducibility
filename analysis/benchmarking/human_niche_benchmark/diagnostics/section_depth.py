"""Sequencing depth per tissue section.

TERRA ranks genes by raw count with a 256-token budget per cell, so sections
sequenced at different depths give the same niche different token sequences.
This reports, per dataset and section, counts and detected genes per cell and
the fraction of cells over the token budget, plus how strongly depth follows
section vs. ground-truth niche (eta squared of log10 counts per cell).

Output: ``<results>/token_diagnostics/section_depth.csv`` (+ printed tables).
Run in the banksy environment (any scanpy environment works).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from bench_utils import DATASETS, SECTION_KEY, load_benchmark_dataset, method_output_dir  # noqa: E402

BUDGET = 256   # TERRA tokens per cell with 10 neighbours (2816 / 11)


def eta_squared(values, groups):
    df = pd.DataFrame(dict(v=values, g=groups))
    between = df.groupby('g')['v'].agg(['mean', 'size'])
    total = ((df['v'] - df['v'].mean()) ** 2).sum()
    return float((between['size'] * (between['mean'] - df['v'].mean()) ** 2).sum()
                 / total)


def main():
    out_dir = method_output_dir('token_diagnostics')
    rows, summary = [], []
    for dataset in DATASETS:
        adata = load_benchmark_dataset(dataset)
        counts = np.asarray(adata.X.sum(1)).ravel()
        genes = np.asarray((adata.X > 0).sum(1)).ravel()
        obs = pd.DataFrame(dict(section=adata.obs[SECTION_KEY].astype(str).values,
                                niche=adata.obs['niche_gt'].astype(str).values,
                                counts=counts, genes=genes))
        for section, g in obs.groupby('section'):
            rows.append(dict(dataset=dataset, section=section, n_cells=len(g),
                             median_counts=g['counts'].median(),
                             median_genes=g['genes'].median(),
                             frac_over_budget=(g['genes'] > BUDGET).mean()))
        per_section = pd.DataFrame([r for r in rows if r['dataset'] == dataset])
        log_counts = np.log10(np.maximum(counts, 1))
        summary.append(dict(
            dataset=dataset, n_sections=len(per_section),
            median_counts_max_over_min=(per_section['median_counts'].max()
                                        / per_section['median_counts'].min()),
            eta2_log_counts_section=(eta_squared(log_counts, obs['section'])
                                     if len(per_section) > 1 else None),
            eta2_log_counts_niche=eta_squared(log_counts, obs['niche'])))
        del adata

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / 'section_depth.csv', index=False)
    pd.DataFrame(summary).to_csv(out_dir / 'section_depth_summary.csv', index=False)
    print(table.round(3).to_string(index=False))
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
