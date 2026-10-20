"""Check that import_fm.novae_batch_effect_correction equals Novae's own
batch-effect correction as patched in mouse_benchmarking/novae_benchmarking.ipynb
(cells "_fixed_dcps" and "_patched_batch_effect_correction", copied verbatim
below), on toy data with several sections, uneven and missing domains, and
cells without a domain / valid neighbourhood. Exits non-zero on a mismatch.

    python checks/test_novae_correction.py     (novae env)
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import novae.utils.correct as _correct
from novae._constants import Keys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from import_fm import novae_batch_effect_correction  # noqa: E402

# --- the notebook's patch (mouse_benchmarking/novae_benchmarking.ipynb) ---
_orig_dcps = _correct._domains_counts_per_slide


def _fixed_dcps(*args, **kwargs):
    result = _orig_dcps(*args, **kwargs)
    return result.loc[~result.index.duplicated(keep="first")]


_correct._domains_counts_per_slide = _fixed_dcps


def _patched_batch_effect_correction(adatas, obs_key):
    adata_indices, slides_obs_indices = _correct._slides_indices(adatas)
    domains_counts_per_slide = _correct._domains_counts_per_slide(adatas, obs_key)
    domains = domains_counts_per_slide.columns[:-1]
    valid_mask = domains_counts_per_slide[domains].notna().any(axis=0)
    domains = domains[valid_mask]
    valid_mask2 = (domains_counts_per_slide[domains] > 0).any(axis=0)
    domains = domains[valid_mask2]
    ref_slide_ids = domains_counts_per_slide[domains].idxmax(axis=0)

    def _centroid_reference(domain, slide_id, obs_key):
        adata_ref_index = domains_counts_per_slide[Keys.ADATA_INDEX].loc[slide_id]
        if isinstance(adata_ref_index, pd.Series):
            adata_ref_index = int(adata_ref_index.iloc[0])
        else:
            adata_ref_index = int(adata_ref_index)
        adata_ref = adatas[adata_ref_index]
        where = (adata_ref.obs[Keys.SLIDE_ID] == slide_id) & (adata_ref.obs[obs_key] == domain)
        return adata_ref.obsm[Keys.REPR][where].mean(0)

    centroids_reference = pd.DataFrame({
        domain: _centroid_reference(domain, slide_id, obs_key)
        for domain, slide_id in ref_slide_ids.items()})
    for adata in adatas:
        adata.obsm[Keys.REPR_CORRECTED] = adata.obsm[Keys.REPR].copy()
    for adata_index, obs_indices in zip(adata_indices, slides_obs_indices):
        adata = adatas[adata_index]
        for domain in domains:
            if adata.obs[Keys.SLIDE_ID].iloc[obs_indices[0]] == ref_slide_ids.loc[domain]:
                continue
            indices_domain = obs_indices[adata.obs.iloc[obs_indices][obs_key] == domain]
            if len(indices_domain) == 0:
                continue
            centroid_reference = centroids_reference[domain].values
            centroid = adata.obsm[Keys.REPR][indices_domain].mean(0)
            adata.obsm[Keys.REPR_CORRECTED][indices_domain] += centroid_reference - centroid


# --- toy data: 3 sections (one batch each, as in the paper notebook), 5 domains,
# one section missing two domains, ~5 % cells without a domain ---
rng = np.random.default_rng(0)
sections, domains, embs = [], [], []
for section, n, shift in [('A', 300, 0.0), ('B', 200, 2.0), ('C', 250, -1.0)]:
    probs = [.3, .3, .2, .1, .1] if section != 'C' else [.4, .4, .2, 0, 0]
    d = rng.choice(['d0', 'd1', 'd2', 'd3', 'd4'], n, p=probs).astype(object)
    d[rng.random(n) < 0.05] = np.nan
    sections += [section] * n
    domains += list(d)
    embs.append(rng.normal(size=(n, 8)) + shift)
X = np.vstack(embs).astype(np.float32)
sections = np.array(sections)
domains = pd.Series(domains)

adatas = []
for section in ['A', 'B', 'C']:
    m = sections == section
    obs = pd.DataFrame({Keys.SLIDE_ID: pd.Categorical([section] * m.sum()),
                        'dom': pd.Categorical(domains[m].values),
                        Keys.IS_VALID_OBS: ~pd.isna(domains[m].values)},
                       index=[f'{section}{i}' for i in range(m.sum())])
    adata = ad.AnnData(np.zeros((m.sum(), 1)), obs=obs)
    adata.obsm[Keys.REPR] = X[m].copy()
    adatas.append(adata)
_patched_batch_effect_correction(adatas, 'dom')
reference = np.vstack([a.obsm[Keys.REPR_CORRECTED] for a in adatas])

ours = novae_batch_effect_correction(
    X, sections, domains.fillna('-1').astype(str).values)
max_diff = float(np.abs(ours - reference).max())
shifted = float(np.mean(np.abs(ours - X).sum(1) > 0))
print(f'max |ours - notebook patch| = {max_diff:.3g}; cells shifted: {shifted:.1%}')
ok = max_diff < 1e-5 and shifted > 0
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
