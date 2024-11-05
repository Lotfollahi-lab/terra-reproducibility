# nichejepa-reproducibility

## Datasets

`/lustre/scratch126/cellgen/team361/DATASETS/` houses all the data across experiments. An experiment is defined as a collection of anndata objects that may be from the same or different datasets, species, tissues, gene panels, and batches. 
- `silver` -- contains preprocessed AnnData files
- `gold` -- contains tokenized datasets
- `tokenizer` -- contains intermediate tokenizer files such as token dictionaries

### Preparing Semi-Real Experiment data comprising of Multiple Gene Panels:
For each batch of AnnData in the input directory, this script creates `n_gene_panels` new batches of AnnData, one per new gene panel. The new gene panels are created by randomly selecting genes from the original gene list according to the following procedure:
1. A fixed number of genes are shared between all gene panels.
2. The remaining genes are divided into `n_gene_panels` subsets according to the `subset_ratios` parameter.
3. The shared genes and the genes in each subset are concatenated to form the gene panels.

Usage:
```
python analysis/data_simulation/create_semi-synthetic-data_multiple-gene-panels.py --config_file [CONFIG-FILE]
```
To create `mmb0-300k-2b_2p` from `mmb0-300k-2b_1p`, use `config/multiple-gene-panels/multiple-gene-panels_mmb0-300k-2b_1p_to_mmb0-300k-2b_2p.yaml`
- `mmb0-300k-2b_1p` -- Merfish Mouse Brain | Batches 217, 218 | 1 gene panel
- `mmb0-300k-2b_2p` -- Merfish Mouse Brain | Batches 217, 218 | 2 gene panel (semi-real)


### Preparing Semi-Real Experiment data comprising of Multiple Assays with Shared Genes:
"""
For each batch of AnnData in the input directory, this script creates a list of shared gene symbols across all batches. It then creates a new batch of AnnData for each batch in the input directory with only the shared genes. The new batches are saved in the output directory.

Usage:
```
python create_semi-synthetic-data_multiple-assays.py --config_file [CONFIG-FILE]
```

To create `mmb0-1b_smb1-1b_1p` from `mmb0-1b_smb1-1b_2p`, use `config/multiple-assays/multiple-assays_mmb0-1b_smb1-1b_2p_to_mmb0-1b_smb1-1b_1p.yaml`
- `mmb0-1b_smb1-1b_2p` -- Merfish Mouse Brain | Batch 82 | Starmap Plus Mouse CNS | Batch 15 | 2 gene panels
- `mmb0-1b_smb1-1b_1p` -- Merfish Mouse Brain | Batch 82 | Starmap Plus Mouse CNS | Batch 15 | 1 gene panels (semi-real, shared genes only)
Use `config/multiple-assays/mmb70-1b_smb71-1b_smb13-1b_3p_to_mmb70-1b_smb71-1b_smb13-1b_1p.yaml` for creating a shared gene panel from the following three datasets (CAST paper):
- `mmb70-1b` -- Merfish Mouse Brain | co1_slice37 in co1_sample13
- `smb71-1b` -- STARmap plus Mouse Brain | S1–S8, S64_1 and S64_2 from SCP1375
- `smb13-1b` -- Slide-seq Mouse Brain | Puck/Slice 42

## Tokenization

### CellNeighborhoodTokenizer:

Execute the following to create a Hugging Face dataset created using the Cell Neighborhood Tokenization from a preprocessed `anndata` object:
```
python analysis/tokenization/create_cell_tokenized_dataset.py --config_file [CONFIG-FILE] --override [OPTIONS]
```
Sample config file is available at `config/cell-neighborhood-tokenizer/cell-neighborhood-tokenizer_sss2-1b_1p_read-depth_shifted-log.yaml`. The override flag allows you to override two things: (1) the number of processes parameter from the config file by using `nproc=[NPROC]`, and (2) whether to reuse the intermediate files such as token dictionary or rebuild them first before recreating the tokenized data by using `overwrite=True`. Other parameters from the config file remain unaffected. If in doubt, it may be safer to overwrite.

If you are on `farm22`, then the easier alternative is to simply use the wrapper script as follows:
```
./scripts/wrapper_create_cell_neighborhood_tokenized_dataset.sh [EXPERIMENT-NAME]
```
This will create an LSF job with the appropriate requirements and fetch the corresponding config file. Currently, the following experiment names are supported:
- `mmb0-8b_1p` -- Merfish Mouse Brain | Batches 1-8 | 1 gene panel
- `mmb0-100k-1b_1p` -- Merfish Mouse Brain | Batch 224 | 1 gene panel
- `mmb0-300k-2b_1p` -- Merfish Mouse Brain | Batches 217, 218 | 1 gene panel
- `mmb0-300k-2b_2p` -- Merfish Mouse Brain | Batches 217, 218 | 2 gene panel (semi-real)
- `mmb0-239b_1p` -- Merfish Mouse Brain | All Batches 1-239 | 1 gene panel
- `mmb0-1b_smb1-1b_2p` -- Merfish Mouse Brain | Batch 82 | Starmap Plus Mouse CNS | Batch 15 | 2 gene panels
- `mmb0-1b_smb1-1b_1p` -- Merfish Mouse Brain | Batch 82 | Starmap Plus Mouse CNS | Batch 15 | 1 gene panels (semi-real, shared genes only)
- `smb1-20b_1p` -- Starmap Plus Mouse CNS | Batches 1-20 | 1 gene panel
- `sss2-1b_1p` -- Simulated 10k | Batch 1 | 1 gene panel
- `sss3-1b_1p` -- Simulated 100k | Batch 1 | 1 gene panel
- `xhs1000-39b_1p` -- Xenium Human Skin | Batch 39 | 1 gene panel


### Adding A New Experiment:
To tokenize a new experiment, follow the steps below:
1. Choose an `[EXPERIMENT-NAME]` (e.g.`mmb0-8b_1p`)
2. Create a folder at the following location: `/lustre/scratch126/cellgen/team361/DATASETS/silver/[EXPERIMENT-NAME]`. Add the appropriate (harmonized) AnnData batch files.
3. Create a config file at the following location with this naming convention: `config/cell-neighborhood-tokenizer/cell-neighborhood-tokenizer_[EXPERIMENT-NAME]_read-depth_shifted-log.yaml`
4. Update the `experiment_name`, `experiment_description`, and potentially `nproc` and `num_shards`. Ideally, you shouldn't need to update any of the other parameters here if you just want to repeat the same tokenization pipeline on a new experiment.
5. Update `scripts/wrapper_create_cell_neighborhood_tokenized_dataset.sh` by adding the `RAM`, `TIME`, `CORES`, and `QUEUE` as LSF job parameters as a new case titled `[EXPERIMENT-NAME]`.


### CellGraphTokenizer:

Executing Cell Graph Tokenization from a preprocessed `anndata` object occurs in exactly the same way as the Cell Neighborhood Tokenization described above:
```
python analysis/tokenization/create_cell_tokenized_dataset.py --config_file [CONFIG-FILE] --override [OPTIONS]
```
The only difference is that the config files are stored in `config/cell-graph-tokenizer`.
The corresponding wrapper script is available at `scripts/wrapper_create_cell_graph_tokenized_dataset.sh`.


## Analysis

### Compute Dataset Statistics:
Execute the following to compute batchwise statistics of a collection of `anndata` objects:
```
python analysis/statistics/compute_anndata_statistics.py --config_file [CONFIG-FILE] --override [OPTIONS]
```
Sample config file is available at `config/anndata_statistics_sim1_1105fts_10000locs.yaml`. The override flag allows you to override the number of processes parameter from the config file by using the `--override mode=[sequential | parallel] nproc=[NPROC]`. Other parameters from the config file remain unaffected.