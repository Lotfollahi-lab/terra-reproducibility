import argparse
import pickle
import random
from collections import defaultdict

from datasets import load_from_disk


def extract_prefix(item):
    return item.rsplit("_", 1)[0] # Extract prefix, e.g., "23_batch0_0" -> "23_batch0"

def main(args):
    # Load dataset and retrieve cell IDs
    dataset = load_from_disk(args.tokenized_data_folder_path)
    cell_ids = dataset['cell_id']

    # Step 1: Exclude items with prefixes in val_test_batch_ids
    # Filtering the items and keeping track of indices
    filtered_items = [(index, item) # Keep track of the original index
                      for index, item in enumerate(cell_ids)
                      if extract_prefix(item) not in args.val_test_batch_ids]

    # Step 2: Group items by prefix
    grouped_items = defaultdict(list)
    for index, item in filtered_items:
        prefix = extract_prefix(item)
        grouped_items[prefix].append((index, item))

    # Step 3: Stratified sampling
    train_subset_indices = []
    remaining_items_indices = []

    # Fix random seed for reproducibility
    random.seed(42)
    for prefix, group in grouped_items.items():
        random.shuffle(group) # Shuffle within the prefix
        split_idx = int(len(group) * 0.19) # 19% split
        train_subset_indices.extend([index for index, _ in group[:split_idx]]) # Add indices of 19% to train_subset
        remaining_items_indices.extend([index for index, _ in group[split_idx:]]) # Add indices of remaining items

    # Outputs
    print("Train Subset Indices (19%):", len(train_subset_indices))
    print("Remaining Items Indices (81%):", len(remaining_items_indices))
    
    # Save results
    with open("precomputed_split_60m_10m_train_subset_indices.pkl", "wb") as f:
        pickle.dump(train_subset_indices, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--tokenized_data_folder_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/team361/DATASETS/gold/cell-graph-tokenizer/human_cohort2_60m/human_cohort2_60m_32_None_None_None_gene_corrected_read_depth_None_shifted_log_knn_10.dataset',
                        help='Tokenized data directory path.')
    parser.add_argument('--val_test_batch_ids',
                        type=str,
                        nargs='+',
                        default=['49_batch0',
                                 '59_batch0',
                                 '59_batch1',
                                 '59_batch2',
                                 '59_batch3',
                                 '59_batch4',
                                 '59_batch5',
                                 '59_batch6',
                                 '59_batch7',
                                 '78_batch0',
                                 '78_batch1',
                                 '78_batch2',
                                 '78_batch3',
                                 '78_batch4',
                                 '78_batch5',
                                 '78_batch6',
                                 '78_batch7',
                                 '78_batch8',
                                 '78_batch9',
                                 '78_batch10',
                                 '1000_batch4',
                                 '1000_batch6',
                                 '1000_batch11',
                                 '1000_batch12',
                                 '1000_batch18',
                                 '1000_batch19',
                                 '1000_batch20',
                                 '1000_batch21',
                                 '1000_batch22',
                                 '1000_batch23',
                                 '1000_batch24',
                                 '1000_batch25',
                                 '1000_batch26',
                                 '1000_batch27',
                                 '1000_batch28',
                                 '1000_batch29',
                                 '1000_batch32',
                                 '1000_batch34',
                                 '1000_batch36',
                                 '1000_batch38',
                                 '1002_batch3',
                                 '1002_batch4',
                                 '1002_batch16',
                                 '1002_batch18',
                                 '1002_batch31',
                                 '1002_batch34',
                                 '1002_batch23',
                                 '1002_batch26',
                                 '1002_batch27',
                                 '1002_batch27',
                                 '1002_batch39',
                                 '1002_batch40',
                                 '1002_batch45',
                                 '1002_batch46',
                                 '1002_batch53',
                                 '1002_batch55',
                                 '1007_batch0',
                                 '1007_batch1',
                                 '1007_batch2',
                                 '1007_batch3',
                                 '1007_batch4',
                                 '1007_batch5',
                                 '1007_batch6',
                                 '1007_batch7'],
                        help='Validation and test batch IDs.')
    args = parser.parse_args()
    
    main(args)
