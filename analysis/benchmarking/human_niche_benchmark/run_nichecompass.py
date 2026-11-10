"""NicheCompass niche embeddings for the human niche benchmark.

Mirrors analysis/benchmarking/mouse_benchmarking/nichecompass_benchmarking.ipynb
(same model/training hyperparameters), with human gene programs (OmniPath,
NicheNet v2, MEBOCOST) loaded from local files. Spatial graph: 10-NN per section
with self loop, made symmetric (required by NicheCompass). The section is the
categorical covariate (embedding injected into the gene expression decoder, no
edges between sections). Integration is native, no post-hoc alignment.

Run in the nichecompass environment on a GPU node (run_method.lsf, PYTHON_ENV=PYTHON_NICHECOMPASS).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import squidpy as sq
import nichecompass
from nichecompass.models import NicheCompass
from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
                                extract_gp_dict_from_mebocost_ms_interactions,
                                extract_gp_dict_from_nichenet_lrt_interactions,
                                extract_gp_dict_from_omnipath_lr_interactions,
                                filter_and_combine_gp_dict_gps_v2)

sys.path.append(str(Path(__file__).parent))
from bench_utils import (CACHE_FOLDER_PATH, N_NEIGHS,  # noqa: E402
                         SECTION_KEY, Timer,
                         git_commit, load_benchmark_dataset, save_embedding)

# Gene-program resources (OmniPath, NicheNet, MEBOCOST) as local files
NICHECOMPASS_GP_DATA_PATH = '/nfs/team361/sb75/terra-reproducibility/artifacts/nichecompass/data'

METHOD = 'nichecompass'
EMB_KEY = 'nichecompass_latent'

GP_DATA = NICHECOMPASS_GP_DATA_PATH
# Checkout the nichecompass package is imported from (recorded with its commit)
NICHECOMPASS_SOURCE = Path(nichecompass.__file__).resolve().parents[2]
GP_FILES = dict(
    omnipath_lr=f'{GP_DATA}/gene_programs/omnipath_lr_network.csv',
    nichenet_lr=f'{GP_DATA}/gene_programs/nichenet_lr_network_v2_human.csv',
    nichenet_lt=f'{GP_DATA}/gene_programs/nichenet_ligand_target_matrix_v2_human.csv',
    mebocost=f'{GP_DATA}/gene_programs/metabolite_enzyme_sensor_gps',
    orthologs=f'{GP_DATA}/gene_annotations/human_mouse_gene_orthologs.csv')

# Hyperparameters from the mouse notebook.
PARAMS = dict(
    conv_layer_encoder='gatv2conv',
    active_gp_thresh_ratio=0.01,
    n_epochs=400,
    n_epochs_all_gps=25,
    lr=0.001,
    lambda_edge_recon=500000.,
    lambda_gene_expr_recon=300.,
    lambda_l1_masked=0.,
    lambda_l1_addon=30.,
    edge_batch_size=4096,
    n_sampled_neighbors=4)


def clean_omnipath_network() -> tuple[str, int]:
    """Drop OmniPath rows without a ligand/receptor gene symbol.

    36 of ~10.9k rows in the local file have an empty
    ``genesymbol_intercell_target``; NicheCompass crashes on these NaNs. They
    cannot map to any gene, so dropping them does not change the gene programs.
    """
    df = pd.read_csv(GP_FILES['omnipath_lr'], index_col=0)
    keep = (df['genesymbol_intercell_source'].notna()
            & df['genesymbol_intercell_target'].notna())
    path = Path(CACHE_FOLDER_PATH) / 'omnipath_lr_network_clean.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(path)
    return str(path), int((~keep).sum())


def build_gene_programs():
    omnipath_path, n_dropped = clean_omnipath_network()
    print(f'OmniPath: dropped {n_dropped} rows without gene symbols.')
    omnipath = extract_gp_dict_from_omnipath_lr_interactions(
        species='human', load_from_disk=True, save_to_disk=False,
        lr_network_file_path=omnipath_path,
        gene_orthologs_mapping_file_path=GP_FILES['orthologs'],
        plot_gp_gene_count_distributions=False)
    nichenet = extract_gp_dict_from_nichenet_lrt_interactions(
        species='human', version='v2', keep_target_genes_ratio=1.,
        max_n_target_genes_per_gp=250, load_from_disk=True, save_to_disk=False,
        lr_network_file_path=GP_FILES['nichenet_lr'],
        ligand_target_matrix_file_path=GP_FILES['nichenet_lt'],
        gene_orthologs_mapping_file_path=GP_FILES['orthologs'],
        plot_gp_gene_count_distributions=False)
    mebocost = extract_gp_dict_from_mebocost_ms_interactions(
        dir_path=GP_FILES['mebocost'], species='human',
        plot_gp_gene_count_distributions=False)
    gp_dict = filter_and_combine_gp_dict_gps_v2(
        [omnipath, nichenet, mebocost], verbose=False)
    print(f'Gene programs: OmniPath {len(omnipath)}, NicheNet {len(nichenet)}, '
          f'MEBOCOST {len(mebocost)} -> combined {len(gp_dict)}')
    return gp_dict, n_dropped


def main(args):
    pd.options.future.infer_string = False   # as in the mouse notebook
    adata = load_benchmark_dataset(args.dataset, subsample=args.subsample)
    dataset = args.dataset + (f'_sub{args.subsample}' if args.subsample else '')
    adata.uns['benchmark']['dataset_name'] = dataset
    print(adata)

    n_sections = adata.obs[SECTION_KEY].nunique()
    params = dict(PARAMS, n_epochs=args.n_epochs or PARAMS['n_epochs'])

    with Timer() as t_embed:
        gp_dict, n_omnipath_dropped = build_gene_programs()
        adata.layers['counts'] = adata.X.copy()

        sq.gr.spatial_neighbors(adata, library_key=SECTION_KEY,
                                coord_type='generic', spatial_key='spatial',
                                n_neighs=N_NEIGHS, set_diag=True)
        adj = adata.obsp['spatial_connectivities']
        adata.obsp['spatial_connectivities'] = adj.maximum(adj.T)

        add_gps_from_gp_dict_to_adata(
            gp_dict=gp_dict, adata=adata,
            gp_targets_mask_key='nichecompass_gp_targets',
            gp_targets_categories_mask_key='nichecompass_gp_targets_categories',
            gp_sources_mask_key='nichecompass_gp_sources',
            gp_sources_categories_mask_key='nichecompass_gp_sources_categories',
            gp_names_key='nichecompass_gp_names',
            min_genes_per_gp=2, min_source_genes_per_gp=1,
            min_target_genes_per_gp=1, max_genes_per_gp=None,
            max_source_genes_per_gp=None, max_target_genes_per_gp=None)
        n_gps = len(adata.uns['nichecompass_gp_names'])
        print(f'Gene programs retained for this gene panel: {n_gps}')

        model = NicheCompass(
            adata,
            counts_key='counts',
            adj_key='spatial_connectivities',
            cat_covariates_embeds_injection=['gene_expr_decoder'],
            cat_covariates_keys=[SECTION_KEY],
            cat_covariates_no_edges=[True],
            cat_covariates_embeds_nums=[n_sections],
            gp_names_key='nichecompass_gp_names',
            active_gp_names_key='nichecompass_active_gp_names',
            gp_targets_mask_key='nichecompass_gp_targets',
            gp_targets_categories_mask_key='nichecompass_gp_targets_categories',
            gp_sources_mask_key='nichecompass_gp_sources',
            gp_sources_categories_mask_key='nichecompass_gp_sources_categories',
            latent_key=EMB_KEY,
            conv_layer_encoder=params['conv_layer_encoder'],
            active_gp_thresh_ratio=params['active_gp_thresh_ratio'])
        model.train(n_epochs=params['n_epochs'],
                    n_epochs_all_gps=params['n_epochs_all_gps'],
                    lr=params['lr'],
                    lambda_edge_recon=params['lambda_edge_recon'],
                    lambda_gene_expr_recon=params['lambda_gene_expr_recon'],
                    lambda_l1_masked=params['lambda_l1_masked'],
                    lambda_l1_addon=params['lambda_l1_addon'],
                    edge_batch_size=params['edge_batch_size'],
                    n_sampled_neighbors=params['n_sampled_neighbors'],
                    use_cuda_if_available=True,
                    verbose=False)
        adata.obsm[EMB_KEY] = np.asarray(adata.obsm[EMB_KEY], dtype=np.float32)

    # Gene-program masks are large; keep only embeddings.
    for key in [k for k in adata.varm if k.startswith('nichecompass')]:
        del adata.varm[key]
    for key in [k for k in adata.uns if k.startswith('nichecompass')]:
        del adata.uns[key]
    save_embedding(adata, METHOD, EMB_KEY, dict(
        n_gene_programs=n_gps, n_omnipath_rows_dropped=n_omnipath_dropped,
        embedding_seconds=t_embed.seconds,
        params=dict(params, n_neighs=N_NEIGHS, adjacency='symmetric',
                    gene_programs='OmniPath + NicheNet v2 + MEBOCOST (human)',
                    nichecompass_source=(f'{NICHECOMPASS_SOURCE} @ '
                                         f'{git_commit(NICHECOMPASS_SOURCE)}'),
                    integration='native (section covariate)')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--subsample', type=int, default=None,
                        help='Random subset of cells (smoke tests only).')
    parser.add_argument('--n_epochs', type=int, default=None,
                        help='Override n_epochs (smoke tests only).')
    main(parser.parse_args())
