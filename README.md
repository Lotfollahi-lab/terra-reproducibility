# nichejepa-reproducibility

## Datasets:

### `sim1`:

Execute the following to create a Hugging Face tokenized data object from a preprocessed `anndata` object:
```
python analysis/tokenization/create_cell_neighborhood_tokenized_dataset.py --config_file config/cell-neighborhood-tokenization_sim1_shifted-log.yaml
```

### `MERFISH Mouse Brain (8 batches)`:

Execute the following to create a Hugging Face tokenized data object from 8 preprocessed `anndata` objects stored in 8 `.h5ad` files:
```
python analysis/tokenization/create_cell_neighborhood_tokenized_dataset.py --config_file config/cell-neighborhood-tokenization_sim1_shifted-log.yaml
```

## Scripts:
This wrapper script will run the neighborhood tokenization on farm by loading all relevant packages, environments, paths, etc.
Dataset-specific config files can be edited for customised runs and number of cores can be overridden directly from commandline.
```
./scripts/wrapper_neighborhood_tokenizer_profile-nprocs.sh [DATASET-NAME] [CORES]
```
Supports `sim1` or `merfish_mouse_brain` datasets.