    #Tokeniser of Visium data for NicheJEPA finetuning

###Setup
#Set utils path
import sys, os
sys.path.append(os.environ.get(
    'TERRA_SRC',
    '/lustre/scratch126/cellgen/lotfollahi/zh4/Codebase_of_Zichen/terra-main 3/src'))

#Import libraries
import os
import warnings
import gc
from pathlib import Path

import pandas as pd
import scanpy as sc
from app.inference import (harmonize_adata,
                           tokenize_adata)
import numpy as np

#Set model paths 
model_folder_path = '/lustre/scratch126/cellgen/lotfollahi/zh4/NicheJEPA/Data/Visium_Embryo_Figure/Finetuned_models/01112025_081731_614'

#Set latent cluster key
latent_cluster_key = 'predicted_niche'

#Ignore resource warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

#Set directories
raw_data_dir = Path('/lustre/scratch126/cellgen/lotfollahi/zh4/NicheJEPA/Data/Visium_Embryo_Figure/Finetuning/QC_Data')
harmonised_dir = Path('/lustre/scratch126/cellgen/lotfollahi/zh4/NicheJEPA/Data/Visium_Embryo_Figure/Finetuning/Harmonised_data')
tokenised_dir = Path('/lustre/scratch126/cellgen/lotfollahi/zh4/NicheJEPA/Data/Visium_Embryo_Figure/Finetuning/Tokenised_data')

#Create directories if not exist
harmonised_dir.mkdir(parents=True, exist_ok=True)
tokenised_dir.mkdir(parents=True, exist_ok=True)

#Get list of h5ad files
h5ad_files = sorted(list(raw_data_dir.glob('*.h5ad')))
print(f"Found {len(h5ad_files)} h5ad files: {[f.stem for f in h5ad_files]}")

# Get sample index from command line argument 
if len(sys.argv) > 1:
    sample_index = int(sys.argv[1]) - 1
    if sample_index < 0 or sample_index >= len(h5ad_files):
        print(f"Invalid sample index {sample_index + 1}, must be between 1 and {len(h5ad_files)}")
        sys.exit(1)
    h5ad_file = h5ad_files[sample_index]
else:
    print("No sample index provided, processing all samples")
    for h5ad_file in h5ad_files:
        pass

# Process the single sample
sample_name = h5ad_file.stem
print(f"\nProcessing {sample_name}")

###Load data
adata = sc.read_h5ad(h5ad_file)
print(f"Dataset shape: {adata.shape}")

###Harmonize data
adata = harmonize_adata(adata)

#Check for missing x,y coordinates and remove if any
nan_mask = adata.obs[['x', 'y']].isna().any(axis=1)
n_missing = int(nan_mask.sum())
if n_missing > 0:
    print(f"Found {n_missing} observations with missing x or y coordinates in {sample_name}; removing them.")
    adata = adata[~nan_mask].copy()
    print(f"New dataset shape after removing missing coords: {adata.shape}")
else:
    print(f"All observations have valid x,y coordinates in {sample_name}.")

#Save harmonized data
harmonised_path = harmonised_dir / f'{sample_name}_harmonized.h5ad'
adata.write_h5ad(harmonised_path)
print(f"Saved harmonized data to {harmonised_path}")

###Tokenize data
dataset = tokenize_adata(adata,
                         model_folder_path,
                         cache_directory_path='/tmp',
                         nproc = 4,
                         processing_mode = 'parallel')

#Shard dataset
num_shards = 32

# Set dataset path
save_dataset_path = tokenised_dir / f'{sample_name}_tokenised'
save_dataset_path.mkdir(exist_ok=True)

# Save dataset
dataset.save_to_disk(
            str(save_dataset_path),
            num_shards=num_shards)

print(f"Saved tokenized data to {save_dataset_path}")

#Clear memory
del adata
del dataset
gc.collect()

print("Finished processing sample!")