import argparse
import sys
import os
import warnings
import pandas as pd
import scanpy as sc
import json
import matplotlib.image as mpimg
import glob

from terra import (
    embed_dataset,
    harmonize_adata,
    tokenize_adata,
)

# Parse arguments
parser = argparse.ArgumentParser(description='Process a Visium .h5ad sample for NicheJEPA')
parser.add_argument('--h5ad', dest='h5ad', required=False, help='Path to a single .h5ad file to process')
parser.add_argument('h5ad_pos', nargs='?', help='(positional) Path to a single .h5ad file to process')
args = parser.parse_args()
h5ad_path = args.h5ad or args.h5ad_pos

# Data paths
DATA_PATH = f"/lustre/scratch126/cellgen/lotfollahi/zh4/NicheJEPA/Data/Visium_Embryo_Figure/Zero_Shot_Inference/QC_Data"
HARMONISED_DATA_PATH = f"/lustre/scratch126/cellgen/lotfollahi/zh4/NicheJEPA/Data/Visium_Embryo_Figure/Zero_Shot_Inference/Harmonised_data"
TOKENIZED_DATA_PATH = f"/lustre/scratch126/cellgen/lotfollahi/zh4/NicheJEPA/Data/Visium_Embryo_Figure/Zero_Shot_Inference/Tokenised_data"
EMBEDDINGS_PATH = f"/lustre/scratch126/cellgen/lotfollahi/zh4/NicheJEPA/Data/Visium_Embryo_Figure/Zero_Shot_Inference/Final/Non_finetuned_norm"

#Set model paths
model_folder_path = '/lustre/scratch126/cellgen/lotfollahi/zh4/NicheJEPA/Data/Visium_Embryo_Figure/Finetuned_models/01112025_081731_614'

#Set latent cluster key
latent_cluster_key = 'predicted_niche'

#Ignore resource warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

#Backup embeddings pathway
embed_save_dir = EMBEDDINGS_PATH
os.makedirs(embed_save_dir, exist_ok=True)

# Dataset preprocessing
# Determine data file path and sample name
data_file = h5ad_path
sample_name = os.path.basename(data_file).replace('.h5ad', '').replace('_QC', '')
dataset_name = sample_name
data_dir = os.path.dirname(data_file)
print(f"Loading data from: {data_file}")

# Load the file based on extension
if data_file.endswith('.h5ad'):
    adata = sc.read_h5ad(data_file)
else:
    adata = sc.read_10x_h5(data_file)

# Check if spatial coordinates already exist
print(f"  Shape: {adata.obsm['spatial'].shape}")
print(f"  Coordinate range: {adata.obsm['spatial'].min()} to {adata.obsm['spatial'].max()}")

# Print basic info
print(f"\nDataset shape: {adata.shape}")
print(adata.obs.head())

#Harmonise data
adata = harmonize_adata(adata)

# Check for missing x,y coordinates and remove if any
nan_mask = adata.obs[['x', 'y']].isna().any(axis=1)
n_missing = int(nan_mask.sum())
if n_missing > 0:
    print(f"Found {n_missing} observations with missing x or y coordinates in {sample_name}; removing them.")
    adata = adata[~nan_mask].copy()
    print(f"New dataset shape after removing missing coords: {adata.shape}")
else:
    print(f"All observations have valid x,y coordinates in {sample_name}.")

#Save harmonized data
os.makedirs(HARMONISED_DATA_PATH, exist_ok=True)
adata.write_h5ad(f'{HARMONISED_DATA_PATH}/{dataset_name}_harmonized.h5ad')

#Tokenize data
dataset = tokenize_adata(adata,
                         model_folder_path,
                         cache_directory_path='/tmp',
                         nproc = 4,
                         processing_mode = 'parallel')

#Shard dataset
num_shards = 32

# Set dataset path
save_dataset_path = f'{TOKENIZED_DATA_PATH}/{dataset_name}_tokenised'
os.makedirs(save_dataset_path, exist_ok=True)

# Save dataset
dataset.save_to_disk(
            save_dataset_path,
            num_shards=num_shards)

print("Tokenized dataset saved!")

# Embed Data

#Embedding parameters
emb_layer = None
batch_size = 128
pin_memory = False
num_workers = 12
agg_excluded_tokens = None
top_k = None

#Embed dataset
output_embed = embed_dataset(
    dataset=dataset,
    model_folder_path=model_folder_path,
    emb_layer=emb_layer,
    agg_excluded_genes=agg_excluded_tokens, 
    top_k=top_k,
    batch_size=batch_size,
    pin_memory=pin_memory,
    num_workers=num_workers)

#Save embeddings to anndata object
adata.obsm['cell_emb'] = output_embed['cell_emb']
adata.obsm['neighborhood_emb'] = output_embed['neighborhood_emb']

# Backup AnnData with embeddings to Lustre 
embed_file = os.path.join(embed_save_dir, f"{dataset_name}_adata_embeddings.h5ad")
adata.write(embed_file)
print(f"Saved AnnData backup to: {embed_file}")

print("Processing completed!")