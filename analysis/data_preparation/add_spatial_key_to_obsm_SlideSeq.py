import numpy as np
import scanpy as sc

from pathlib import Path

if __name__ == "__main__":
    input_path = Path("/lustre/scratch126/cellgen/team361/DATASETS/silver/smb13-111b_1p")

    for file_path in input_path.glob("*.h5ad"):
        # Load the data
        adata = sc.read(file_path)
        
        # Add spatial coordinates
        adata.obsm["spatial"] = np.vstack([adata.obs['Raw_Slideseq_X'].to_numpy(),adata.obs['Raw_Slideseq_Y'].to_numpy()]).T
        
        # Save the data
        adata.write(file_path)
        
        print(f"File: {file_path}")
        print(adata)
        print("")