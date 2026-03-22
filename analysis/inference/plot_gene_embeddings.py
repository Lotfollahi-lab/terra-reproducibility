import pickle

import anndata as ad
import scanpy as sc
import rapids_singlecell as rsc

gene_emb_path = f'/nfs/team361/sb75/nichejepa-reproducibility/artifacts/{args.cohort_name}/{args.load_timestamp}/ep3_5k_{args.tissue}_{args.tissue}_avg_gene_embed.pkl'

with open(gene_emb_path, 'rb') as f:
    output_gene_embed = pickle.load(f)

neighborhood_gene_emb = output_gene_embed['neighborhood_gene_emb_average_per_data']

# Filter missing genes
mask = ~np.isnan(neighborhood_gene_emb).any(axis=1)
neighborhood_gene_emb = neighborhood_gene_emb[mask]

adata_tmp = sc.read_h5ad('/nfs/team361/sb75/DATASETS/silver/xhe1008-1-20b_1p/section020_SOB26.h5ad')

adata.obs_names = adata_tmp.var_names[mask]

rsc.pp.neighbors(adata, use_rep='X')
rsc.tl.umap(adata)

latent_leiden_resolution = 1.22 # brain
latent_leiden_resolution = 1.55 # breast
latent_leiden_resolution = 1.36 # cervix
latent_leiden_resolution = 1.7 # lung
latent_leiden_resolution = 1.7 # ovary
latent_leiden_resolution = 1.4 # prostate --> 1 gene missing
latent_leiden_resolution = 1.572 # pancreas
latent_leiden_resolution = 1.46 # skin
latent_cluster_key = f'clusters_res_{latent_leiden_resolution}'

rsc.tl.leiden(adata,
              key_added=latent_cluster_key,
              resolution=latent_leiden_resolution)

print(adata.obs.head(5))

sc.pl.umap(adata,
           color=latent_cluster_key,
           show=False)

adata.write(f'/nfs/team361/sb75/nichejepa-reproducibility/artifacts/{args.cohort_name}/{args.load_timestamp}/ep3_5k_{args.tissue}_avg_neighborhood_gene_embed.h5ad')