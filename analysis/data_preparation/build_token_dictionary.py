import argparse
import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
import scanpy as sc
from pyensembl import EnsemblRelease


def retrieve_special_token_values(
    data_directory_path: Path,
    dataset_names: list[str]) -> Tuple[list[str],
                                       list[str],
                                       list[str],
                                       list[str],
                                       list[str]]:
    """
    Process each batch of AnnData included in `dataset_names` in the data
    directory to retrieve special token values. These will be added to the
    token dict.

    Parameters
    ----------
    data_directory_path:
        The path to the input directory.
    dataset_names:
        Names of datasets included in corpus and considered in generation of
        token dictionary.

    Returns:
    ----------
    species_key_list:
        A list of species, one per unique species across batches.
    tissue_key_list:
        A list of tissues, one per unique tissue across batches.
    assay_key_list:
        A list of assays, one per unique assay across batches.
    gene_panel_key_list:
        A list of gene panels, one per unique gene panel size across batches.
    batch_key_list:
        A list of batches.
    """
    file_paths = []
    for dataset_name in dataset_names:
        silver_dataset_path = Path(
            data_directory_path) / 'silver' / dataset_name
        dataset_file_paths = list(silver_dataset_path.glob(f"**/*.h5ad"))
        file_paths.extend(dataset_file_paths)

    if len(file_paths) == 0:
        raise FileNotFoundError(
            "No '.h5ad' files of included datasets found in "
            f"{data_directory_path}.")
            
    # Initialize lists for capturing special token values
    species_key_list = []
    tissue_key_list = []
    assay_key_list = []
    gene_panel_key_list = []
    batch_key_list = []
                
    for file_path in file_paths:
        print(f"Loading data from {file_path}.")
        adata = sc.read_h5ad(file_path)

        # Capture species key for the current batch
        # e.g. species_key = "homo_sapiens"...
        species_key = f"spv_{adata.uns['species']}"
        if species_key not in species_key_list:
            species_key_list.append(species_key)       

        # Capture tissue key for the current batch
        # e.g. tissue_key = "brain"...
        tissue_key = f"spv_{adata.uns['tissue']}"
        if tissue_key not in tissue_key_list:
            tissue_key_list.append(tissue_key)

        # Capture assay key for the current batch
        # e.g. assay_key = "merfish"...
        assay_key = f"spv_{adata.uns['assay']}"
        if assay_key not in assay_key_list:
            assay_key_list.append(assay_key)

        # Capture gene panel key for the current batch
        # e.g. gene_panel_key = "gene_panel5000"...
        gene_panel_key = f"spv_gene_panel{len(adata.var_names)}"
        if gene_panel_key not in gene_panel_key_list:
            gene_panel_key_list.append(gene_panel_key)

        # Capture batch_id_key for the current batch
        # e.g. batch_id_key = "0_batch1"...
        dataset_id = adata.uns['dataset_id']
        batch_id = adata.uns['batch']
        batch_key = f"spv_{dataset_id}_{batch_id}"    
        batch_key_list.append(batch_key)

    return species_key_list, \
           tissue_key_list, \
           assay_key_list, \
           gene_panel_key_list, \
           batch_key_list


def build_token_dictionary(token_dictionary_file_path: str,
                           ensembl_release: int,
                           species_key_list: list[str],
                           tissue_key_list: list[str],
                           assay_key_list: list[str],
                           gene_panel_key_list: list[str],
                           batch_key_list: list[str],
                          ) -> dict:
    """
    Build token dictionary.
    
    Parameters:
    ----------
    token_dictionary_file_path:
        The file path to save the token dictionary.
    ensembl_release:
        Ensembl release used to retrieve gene tokens.
    species_key_list:
        A list of species, one per unique species across batches.
    tissue_key_list:
        A list of tissues, one per unique tissue across batches.
    assay_key_list:
        A list of assays, one per unique assay across batches.
    gene_panel_key_list:
        A list of gene panels, one per gene panel size across batches.
    batch_key_list:
        A list of batches.

    Returns:
    ----------
    token_dict:
        The token dictionary.
    """
    print(f"Creating token dict and saving to {token_dictionary_file_path}.")
    
    # Initialize with mandatory tokens
    token_dict = {'<pad>': 0, '<mask>': 1}
    
    # Add cls tokens
    # e.g. in the case of cell graph tokenizer cls_0: 2 is for cell, cls_1: 3 is
    # for 1st neighbor, cls_2: 4 is for 2nd neighbor, ..., up to 100 neighbors
    # e.g. in the case of cell neighborhood tokenizer cls_0: 2 is for cell,
    # cls_1: 3 is for neighborhood 
    token_dict.update({f'<cls_{i}>' : i + 2 for i in range(100)})

    # Add other special tokens
    curr_token_dict_size = len(token_dict)
    token_dict.update({'spt_species': curr_token_dict_size,
                       'spt_tissue': curr_token_dict_size + 1,
                       'spt_assay': curr_token_dict_size + 2,
                       'spt_gene_panel': curr_token_dict_size + 3,
                       'spt_batch': curr_token_dict_size + 4, 
                       })
    
    # Add gene tokens (protein-coding and miRNA genes from Ensembl)
    # e.g. token_dict = {'ENSG00000139618': 107, ...}
    gene_ids = []
    for species_key in species_key_list:
        species = species_key[4:]
        ensembl = EnsemblRelease(ensembl_release, species=species)
        ensembl.download()
        ensembl.index()
        all_genes = ensembl.genes()
        protein_coding_genes = [
            gene for gene in all_genes if gene.biotype == "protein_coding"]
        mirna_genes = [gene for gene in all_genes if gene.biotype == "miRNA"]
        all_relevant_genes = protein_coding_genes + mirna_genes
        species_gene_ids = [gene.gene_id for gene in all_relevant_genes]
        gene_ids.extend(species_gene_ids)

    curr_token_dict_size = len(token_dict)
    token_dict.update({
        gene_id: idx + curr_token_dict_size for
        idx, gene_id in enumerate(gene_ids)})

    # Add 1 token per species in species_key_list
    # e.g. token_dict = {'homo_sapiens': 107 + len(gene_list)}
    curr_token_dict_size = len(token_dict)
    token_dict.update({
        species_key: idx + curr_token_dict_size for
        idx, species_key in enumerate(species_key_list)})

    # Add 1 token per tissue in tissue_key_list
    # e.g. token_dict = {'brain': 107 + len(gene_list) +
    # len(species_key_list), ...}
    curr_token_dict_size = len(token_dict)
    token_dict.update({
        tissue_key: idx + curr_token_dict_size for
        idx, tissue_key in enumerate(tissue_key_list)})

    # Add 1 token per assay in assay_key_list
    # e.g. token_dict = {'merfish': 107 + len(gene_list) +
    # len(species_key_list) + len(tissue_key_list), ...}
    curr_token_dict_size = len(token_dict)
    token_dict.update({
        assay_key: idx + curr_token_dict_size for
        idx, assay_key in enumerate(assay_key_list)})

    # Add 1 token per gene_panel_id in gene_panel_id_key_list
    # e.g. token_dict = {'gene_panel0': 107 + len(gene_list) +
    # len(species_key_list) + len(tissue_key_list) + len(assay_key_list), ...}
    curr_token_dict_size = len(token_dict)
    token_dict.update({
        gene_panel_key: idx + curr_token_dict_size for
        idx, gene_panel_key in enumerate(gene_panel_key_list)})
    
    # Add 1 token per batch_id in batch_id_key_list
    # e.g. token_dict = {'0_batch1': 107 + len(gene_list) +
    # len(species_key_list) + len(tissue_key_list) + len(assay_key_list) +
    # len(gene_panel_id_key_list), ...}
    curr_token_dict_size = len(token_dict)
    token_dict.update({
        batch_key: idx + curr_token_dict_size for
        idx, batch_key in enumerate(batch_key_list)})

    # Add 1000 additional dummy tokens
    curr_token_dict_size = len(token_dict)
    token_dict.update({
        f'dummy_{idx}': idx + curr_token_dict_size for
        idx in range(1000)})

    with open(token_dictionary_file_path, "wb") as file:
        pickle.dump(token_dict, file)

    print(token_dict)
    print("")
    
    return token_dict


def main(args):
    # Retrieve list of datasets included in corpus
    datasets_df = pd.read_csv(
        args.cohort_dataset_names_file_path,
        index_col=0,
        header=None)
    dataset_names = datasets_df.index.tolist()

    # Retrieve special token values from all datasets included in corpus
    species_key_list, \
    tissue_key_list, \
    assay_key_list, \
    gene_panel_key_list, \
    batch_key_list = retrieve_special_token_values(
        args.data_directory_path,
        dataset_names)

    # Build token dictionary
    token_dict = build_token_dictionary(
        token_dictionary_file_path=args.token_dictionary_file_path,
        ensembl_release=args.ensembl_release,
        species_key_list=species_key_list,
        tissue_key_list=tissue_key_list,
        assay_key_list=assay_key_list,
        gene_panel_key_list=gene_panel_key_list,
        batch_key_list=batch_key_list)

    with open(args.token_dictionary_file_path, "wb") as file:
        pickle.dump(token_dict, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument(
        '--data_directory_path',
        type=str,
        default="/lustre/scratch126/cellgen/team361/DATASETS",
        help='Data directory path.')
    parser.add_argument(
        '--cohort_dataset_names_file_path',
        type=str,
        default="/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/analysis/data_preparation/hst_corpus_70m_dataset_names.txt",
        help='Path to the file containing datasets of the cohort.')  
    parser.add_argument(
        '--token_dictionary_file_path',
        type=str,
        default="/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/artifacts/data/token_dictionary_homo_sapiens.pkl",
        help='Token dictionary file path.')
    parser.add_argument(
        '--ensembl_release',
        type=int,
        default=111,
        help='Ensembl release for retrieving gene IDs.')                    
    args = parser.parse_args()
    
    main(args)
