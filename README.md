# terra-reproducibility

## Datasets

List of datasets: https://docs.google.com/spreadsheets/d/1bfdBZ1MZEVKz2-4Ge89zwe5jVT4qnXAndStmXmJehTo/edit?gid=1836124407#gid=1836124407

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
Sample config file is available at `config/cell-neighborhood-tokenizer/cell-neighborhood-tokenizer_sss2-1b_1p`. The override flag allows you to override `overwrite` parameter which is set to `True` by default in the config files. The `overwrite` parameter controls whether a previously created token dictionary should be reused or if it should be recreated from scratch.

If you are on `farm22`, then the easier alternative is to simply use the wrapper script as follows:
```
./scripts/wrapper_create_cell_neighborhood_tokenized_dataset.sh [EXPERIMENT-NAME]
```
This will create an LSF job with the appropriate requirements and fetch the corresponding config file. Currently, the following experiment names are supported:
- `mmb0-100k-1b_1p` -- Merfish Mouse Brain | Batch 224 | 1 gene panel
- `mmb0-300k-2b_1p` -- Merfish Mouse Brain | Batches 217, 218 | 1 gene panel
- `mmb0-300k-2b_2p` -- Merfish Mouse Brain | Batches 217, 218 | 2 gene panel (semi-real)
- `mmb0-239b_1p` -- Merfish Mouse Brain | All Batches 1-239 | 1 gene panel
- `mmb0-1b_smb1-1b_1p` -- Merfish Mouse Brain | Batch 82 | Starmap Plus Mouse CNS | Batch 15 | 1 gene panels (semi-real, shared genes only)
- `mmb0-1b_smb1-1b_2p` -- Merfish Mouse Brain | Batch 82 | Starmap Plus Mouse CNS | Batch 15 | 2 gene panels
- `smb1-20b_1p` -- Starmap Plus Mouse CNS | Batches 1-20 | 1 gene panel
- `sss2-1b_1p` -- Simulated 10k | Batch 1 | 1 gene panel
- `sss3-1b_1p` -- Simulated 100k | Batch 1 | 1 gene panel
- `xhs1000-39b_1p` -- Xenium Human Skin | Batch 39 | 1 gene panel


### Adding A New Experiment:
To tokenize a new experiment, follow the steps below:
1. Choose an `[EXPERIMENT-NAME]` (e.g.`mmb0-8b_1p`)
2. Create a folder at the following location: `/lustre/scratch126/cellgen/team361/DATASETS/silver/[EXPERIMENT-NAME]`. Add the appropriate (harmonized) AnnData batch files.
3. Create a config file at the following location with this naming convention: `config/cell-neighborhood-tokenizer/cell-neighborhood-tokenizer_[EXPERIMENT-NAME].yaml`
4. Update the `experiment_name`, `experiment_description`, and potentially `num_shards`. Ideally, you shouldn't need to update any of the other parameters here if you just want to repeat the same tokenization pipeline on a new experiment.
5. Update `scripts/wrapper_create_cell_neighborhood_tokenized_dataset.sh` by adding the `RAM`, `TIME`, `CORES`, and `QUEUE` as LSF job parameters as a new case titled `[EXPERIMENT-NAME]`.


### CellGraphTokenizer:

Executing Cell Graph Tokenization from a preprocessed `anndata` object occurs in exactly the same way as the Cell Neighborhood Tokenization described above:
```
python analysis/tokenization/create_cell_tokenized_dataset.py --config_file [CONFIG-FILE] --override [OPTIONS]
```
The only difference is that the config files are stored in `config/cell-graph-tokenizer`.
The corresponding wrapper script is available at `scripts/wrapper_create_cell_graph_tokenized_dataset.sh`.


## Analysis

### Compute Cohort Statistics:
Execute the following to compute batchwise statistics of a collection of datasets comprising of one or more `anndata` objects:
```
python analysis/statistics/collect_cohort_statistics.py --cohort_name [COHORT_NAME]
```

`COHORT_NAME` can be any folder in the `silver` directory (e.g. `human_cohort1_50m`, `sss2-1b_1p`, etc.). The output will be stored in `analysis/statistics/results/[COHORT_NAME]_statistics.csv`. 
This can also be achieved by an LSF job using `./scripts/collect_cohort_statistics.sh [COHORT_NAME]`.


### Tokenization
Set tokenization configs for each dataset.
```
python analysis/tokenization/set_tokenization_config_params.py \
--cohort_dataset_names_file_path '/lustre/scratch126/cellgen/team361/sb75/terra-reproducibility/analysis/data_preparation/hst_corpus_70m/hst_corpus_70m_dataset_names.txt' \
--tokenizer "cell-graph-tokenizer" \
--model_input_size 1792 \
--rank_cell_norm_method None \
--rank_gene_norm_method None \
--rank_count_norm_method None \
--count_cell_norm_method None \
--count_gene_norm_method None \
--count_count_norm_method 'shifted_log' \
--include_zero_expr_genes \
--n_neighs 6
```

Run tokenizers
```
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh mhb14-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh mhc15-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh mhl16-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh mhl17-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh mhm18-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh mho19-4b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh mhp20-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh mhu21-3b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh mhh22-3b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb23-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb24-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb25-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb26-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb27-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhl28-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhl29-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhp30-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhs31-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhh32-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhln33-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhp34-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhp35-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhs36-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb37-3b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhc38-4b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhk39-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhl40-2b_2p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhl41-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb42-3b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xht43-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhsc44-6b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhc45-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhl46-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xho47-S1-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xho47-S2-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhp48-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xho50-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhc51-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhd52-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhl53-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xho54-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhc55-3b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh chb56-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh chl57-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh chl59-8b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh chp60-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh chc61-9b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhs62-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh mhl63-10b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh shp75-4b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb76-16b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh  chr78-11b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhk79-4b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh mhl80-8b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb81-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhl82-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh mhb83-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhc1001-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT3-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT4-3b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT5-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT6-1-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT6-2-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT6-3-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT6-4-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT9-6b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT10-7b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT12-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT13-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT14-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT15-10b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT16-15b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT17-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT18-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT19-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT20-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT21-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT23-9b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1002-AT101-6b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhb1003-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhln1004-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhl1005-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhsc1006-2b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhk1007-CV1-41b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhk1007-CV2-41b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhk1007-CV3-41b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhk1007-CV4-41b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhk1007-CV5-41b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhk1007-CV6-41b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhk1007-CV7-41b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhk1007-CV8-41b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhk1007-CV9-41b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhe1008-1-20b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhe1008-2-20b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhe1008-3-20b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhe1008-4-20b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhe1008-5-20b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhe1008-6-20b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhe1008-7-20b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhe1008-8-20b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhe1008-9-20b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhe1008-10-20b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhs49-1b_1p
./analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh xhs1000-39b_1p
```

Check which datasets have been tokenized
```
for dir in */; do
  if ! find "$dir" -type d -name "*None_None_None_shifted_log_knn_6_1792_incl_zero_expr*" | grep -q .; then
    echo "$dir"
  fi
done
```

### Data Preparation
Split dataset
```
python analysis/data_preparation/split_dataset.py \
--tokenized_data_folder_path '/lustre/scratch126/cellgen/team361/DATASETS/gold/cell-graph-tokenizer/hst_corpus_70m/hst_corpus_70m_32_gene_corrected_read_depth_nonzero_mean_None_None_None_shifted_log_knn_6.dataset' \
--cohort_name "hst_corpus_70m"
```

Save number of nonzero tokens
```
python analysis/data_preparation/save_dataset_n_nonzero_tokens.py \
--tokenized_data_folder_path '/lustre/scratch126/cellgen/team361/DATASETS/gold/cell-graph-tokenizer/hst_corpus_70m/hst_corpus_70m_32_gene_corrected_read_depth_nonzero_mean_None_None_None_shifted_log_knn_6.dataset' \
--cohort_name "hst_corpus_70m"
```