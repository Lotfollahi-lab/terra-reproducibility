# nichejepa-reproducibility

## Tokenization

### CellNeighborhoodRankTokenizer:

Execute the following to create a Hugging Face dataset created using the Cell Neighborhood Rank Tokenization from a preprocessed `anndata` object:
```
python analysis/tokenization/create_cell_neighborhood_tokenized_dataset.py --config_file [CONFIG-FILE] --override [OPTIONS]
```
Sample config file is available at `config/cell-neighborhood-tokenization_sim1_1105fts_10000locs_read-depth_shifted-log.yaml`. The override flag allows you to override the number of processes parameter from the config file by using the `--override nproc=[NPROC]`. Other parameters from the config file remain unaffected.

## Analysis

### Compute Dataset Statistics:
Execute the following to compute batchwise statistics of a collection of `anndata` objects:
```
python analysis/statistics/compute_anndata_statistics.py --config_file [CONFIG-FILE] --override [OPTIONS]
```
Sample config file is available at `config/anndata_statistics_sim1_1105fts_10000locs.yaml`. The override flag allows you to override the number of processes parameter from the config file by using the `--override mode=[sequential | parallel] nproc=[NPROC]`. Other parameters from the config file remain unaffected.

## Scripts:

To use the resources on `farm22`, use the following wrapper script to load all relevant packages, environments, paths, etc. Sample usage is as follows
```
./scripts/wrapper_neighborhood_tokenizer_profile-nprocs.sh [DATASET-NAME] [TOKENIZATION] [CORES]
```
Datasets currently supported include: `sim1_1105fts_10000locs`, `sim1_1105genes_100klocs_strongincrements`, `merfish_mouse_brain_8-batch`, `merfish_mouse_brain_16-batch`, `merfish_mouse_brain_64-batch`. 
Choose between `neighborhood` and `graph` options for `[TOKENIZATION]`.