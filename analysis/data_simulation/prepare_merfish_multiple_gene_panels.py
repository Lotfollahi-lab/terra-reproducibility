import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import scanpy as sc
import anndata as ad


def create_dataset_with_multiple_gene_panels(input_directory, 
                                                output_directory,
                                                n_gene_panels,
                                                gene_panel_list,
                                                seed=42):
    file_paths = list(input_directory.glob(f"**/*.h5ad"))
    gene_panel_idx = np.random.randint(0, n_gene_panels, size=len(file_paths))
    
    for file_path, panel_idx in zip(file_paths, gene_panel_idx):
        print(f"Loading data from {file_path}")
        adata = sc.read_h5ad(file_path)
        print(f"Old adata shape: {adata.shape}")
        
        # Subset the adata object to include only the genes in the gene_panel
        gene_panel = gene_panel_list[panel_idx]
        new_adata = adata[:, adata.var['ensembl_id'].isin(gene_panel)].copy()
        print(f"New adata shape: {new_adata.shape}")
         
        # Write new_adata to output_directory with same file name
        output_file_path = output_directory / file_path.name
        new_adata.write(output_file_path)
        print(f"Writing to {output_file_path}")
        print()
    
    print(adata.var_names[:20])
    print(gene_panel[:20])


def create_gene_panels(gene_list, 
                       n_gene_panels=2,
                       shared_genes_ratio=0.4,
                       subset_ratios=[0.4, 0.2],
                       seed=42):
    if subset_ratios is None or len(subset_ratios) != n_gene_panels:
        raise ValueError("subset_ratios must be a list of size equal to n_gene_panels")
    
    if not np.isclose(sum(subset_ratios), 1 - shared_genes_ratio):
        print(sum(subset_ratios), 1 - shared_genes_ratio)
        raise ValueError("The sum of subset_ratios must be equal to 1 - shared_genes_ratio")
    
    n_genes = len(gene_list)
    print(f"Total Number of Genes: {n_genes}")
    print()
    
    gene_idx = np.arange(n_genes, dtype=int)
    np.random.seed(seed)
    np.random.shuffle(gene_idx)
    
    gene_panel_list = []
    
    shared_genes_count = int(n_genes * shared_genes_ratio)
    shared_genes = np.array(gene_list)[gene_idx[:shared_genes_count]]
    
    start_idx = shared_genes_count
    for i in range(n_gene_panels):
        if i == n_gene_panels - 1:
            gene_panel = np.array(gene_list)[gene_idx[start_idx:]]
        else:
            subset_count = int(n_genes * subset_ratios[i])
            gene_panel = np.array(gene_list)[gene_idx[start_idx:start_idx + subset_count]]
            start_idx += subset_count
        
        gene_panel = np.concatenate([shared_genes, gene_panel])
        gene_panel_list.append(gene_panel.tolist())
        
        print(f"Panel {i+1}: {len(gene_panel)} genes")
        print()
    
    # Assert that the union of gene_panels is the same as the gene_list
    union_genes = np.concatenate(gene_panel_list)
    assert set(union_genes) == set(gene_list), "The union of gene_panels is not the same as the gene_list"
    
    return gene_panel_list


def get_gene_list(data_dir):
    file_paths = list(data_dir.glob(f"**/*.h5ad"))
    print(f"Loading data from {file_paths[0]}")
    adata = sc.read_h5ad(file_paths[0])    
    gene_list = adata.var["ensembl_id"].values.tolist()
    return gene_list
    

def main():
   # Configure data paths
    data_directory_path = Path('/lustre/scratch126/cellgen/team361/am84/NicheJEPA/datasets/am84/st_data/')
    st_data_silver_folder_path = data_directory_path / "silver"
    dataset_name = 'merfish_mouse_brain_8-batch'
    
    n_gene_panels = 2
    input_directory = st_data_silver_folder_path / dataset_name
    output_directory = st_data_silver_folder_path / f"{dataset_name}_{n_gene_panels}-gene-panels"
    os.makedirs(output_directory, exist_ok=True)

    shared_genes_ratio = 0.4
    subset_ratios = [0.4, 0.2]
    seed = 42
    
    gene_list = get_gene_list(input_directory)
    
    gene_panel_list = create_gene_panels(gene_list, 
                                            n_gene_panels,
                                            shared_genes_ratio,
                                            subset_ratios,
                                            seed)

    create_dataset_with_multiple_gene_panels(input_directory, 
                                                output_directory,
                                                n_gene_panels,
                                                gene_panel_list,
                                                seed)

if __name__ == '__main__':
    
    main()