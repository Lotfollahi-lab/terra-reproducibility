import pandas as pd
import datasets
from pathlib import Path
import os
import tracemalloc


tokenizer='cell-graph-tokenizer'

num_shards = 32
norm_factor = "None"
norm_method = "shifted_log"
n_neighs = 10
hparam_prefix = '_'.join([str(s) for s in [num_shards, norm_factor, norm_method, "knn", n_neighs]])
print(hparam_prefix)

data_directory_path = Path('/lustre/scratch126/cellgen/team361/DATASETS')
gold_data_path = data_directory_path / 'gold' / tokenizer

cohort_name = 'human_cohort1_50m'
cohort_dir = gold_data_path / cohort_name
cohort_dataset_dir = cohort_dir / f"{cohort_name}_{hparam_prefix}"
output_path = str(
    (Path(cohort_dir) / f"{cohort_name}_{hparam_prefix}").with_suffix(
        '.dataset'))

os.makedirs(cohort_dir, exist_ok=True)

ds_list = []

# Read metadata csv and filter out non-xenium datasets
dataset_names_df = pd.read_csv('/lustre/scratch126/cellgen/team361/am84/NicheJEPA/reproducibility/analysis/tokenization/human_cohort1_50m_dataset_names.txt', index_col=0, header=None)

# valid_datasets = ['chr78-11b_1p', 'xhs1000-39b_1p']
# pattern = '|'.join(valid_datasets)
# dataset_names_df = dataset_names_df[dataset_names_df.index.str.contains(pattern)]

print(dataset_names_df)

for dataset_name, _ in dataset_names_df.iterrows():
    print('Processing: ', dataset_name)
    
    gold_dataset_path = gold_data_path / dataset_name
    dir_substring = f"{dataset_name}_{hparam_prefix}"
    
    gold_dataset_dir = [p for p in gold_data_path.rglob("*") if p.is_dir() and dir_substring in p.name][0]
    
    print(gold_dataset_dir)

    # Read the dataset

    ds = datasets.load_from_disk(str(gold_dataset_dir))
    ds_list.append(ds)
    

# Save the dataset
print("Concatenating datasets...")
tracemalloc.start()
concatenated_dataset = datasets.concatenate_datasets(ds_list)
concatenated_dataset.save_to_disk(output_path, num_shards=num_shards)
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
