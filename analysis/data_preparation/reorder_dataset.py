from datasets import load_from_disk
import numpy as np
import pickle

print("Start script.")

# Load Huggingface dataset
data_path = '/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/hst_corpus_90m/hst_corpus_90m_32_None_None_None_None_None_shifted_log_knn_10.dataset'
ds = load_from_disk(data_path)

# Retrieve indices for reordering
with open('/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer/hst_corpus_90m_32_None_None_None_None_None_shifted_log_knn_10_precomputed_loader_idx.pkl', "rb") as f: 
    indices = pickle.load(f)

ds_all_indices = list(range(len(ds)))

# Add test and val indices at end of training indices
seen = set(indices)
to_add = []
for idx in ds_all_indices:
    if idx not in seen:
        to_add.append(idx)
        seen.add(idx)

indices.extend(to_add)

cache_name = "/lustre/scratch126/cellgen/lotfollahi/DATASETS/tokenizer/hst_corpus_90m_32_None_None_None_None_None_shifted_log_knn_10_loader_reidx.arrow"
ds_ordered = ds.select(indices,
                       indices_cache_file_name=cache_name)

print("Start flattening.")

# Materialize the new physical order
ds_ordered = ds_ordered.flatten_indices(num_proc=128, writer_batch_size=1000)

print("Flattening done.")

new_path = data_path.replace('.dataset', '_ordered.dataset')
ds_ordered.save_to_disk(new_path, num_shards=4000)
print("Saved to:", new_path)

