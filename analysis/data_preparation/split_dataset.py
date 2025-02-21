import argparse
import pickle

from datasets import load_from_disk


def main(args):
    # Load dataset and retrieve cell IDs
    dataset = load_from_disk(args.tokenized_data_folder_path)
    cell_ids = dataset['cell_id']

    validation_indices = []
    test_indices = []

    print('done0')
    print(len(cell_ids))

    # Single loop to determine validation and test indices
    for index, cell_id in enumerate(cell_ids):
        if any(batch_id == f"{cell_id.split('_')[0]}_{cell_id.split('_')[1]}" for batch_id in args.val_batch_ids):
            validation_indices.append(index)
        elif any(batch_id == f"{cell_id.split('_')[0]}_{cell_id.split('_')[1]}" for batch_id in args.test_batch_ids):
            test_indices.append(index)
    print('done1')

    # Another loop to determine train indices
    excluded_indices = set(validation_indices).union(test_indices)
    print('done2')
    print(len(excluded_indices))
    print(len(validation_indices)+len(test_indices))
    print(len(validation_indices))
    print(len(test_indices))
    train_indices = [index for index in range(len(cell_ids)) if index not in excluded_indices]
    print('done3')

    # Extract unique identifiers (first and second part before the first underscore) for each index list and store in sets
    train_unique = {
        f"{cell_ids[i].split('_')[0]}_{cell_ids[i].split('_')[1]}" for i in train_indices}
    validation_unique = {
        f"{cell_ids[i].split('_')[0]}_{cell_ids[i].split('_')[1]}" for i in validation_indices}
    test_unique = {
        f"{cell_ids[i].split('_')[0]}_{cell_ids[i].split('_')[1]}" for i in test_indices}
    print("Train unique identifiers:", train_unique)
    print("Validation unique identifiers:", validation_unique)
    print("Test unique identifiers:", test_unique)
    print('done4')

    # Save the splits as pickle files
    with open(f'/lustre/scratch126/cellgen/team361/DATASETS/tokenizer/{args.tokenized_data_folder_path.rsplit("/", 1)[-1].removesuffix(".dataset")}_precomputed_split_validation.pkl', 'wb') as f:
        pickle.dump(validation_indices, f)
    with open(f'/lustre/scratch126/cellgen/team361/DATASETS/tokenizer/{args.tokenized_data_folder_path.rsplit("/", 1)[-1].removesuffix(".dataset")}_precomputed_split_test.pkl', 'wb') as f:
        pickle.dump(test_indices, f)
    with open(f'/lustre/scratch126/cellgen/team361/DATASETS/tokenizer/{args.tokenized_data_folder_path.rsplit("/", 1)[-1].removesuffix(".dataset")}_precomputed_split_train.pkl', 'wb') as f:
        pickle.dump(train_indices, f)
    print("Saved train, validation, and test dataset splits.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--tokenized_data_folder_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/team361/DATASETS/gold/cell-graph-tokenizer/hst_corpus_70m/hst_corpus_70m_32_None_None_None_None_None_shifted_log_knn_10.dataset',
                        help='Tokenized data directory path.')
    parser.add_argument('--cohort_name',
                        type=str,
                        default="hst_corpus_70m",
                        help='Name of the cohort.')
    parser.add_argument('--val_batch_ids',
                        type=str,
                        nargs='+',
                        default=['1000_batch23',
                                 '1000_batch24',
                                 '1000_batch25',
                                 '1000_batch28',
                                 '1000_batch29'],
                        help='Validation batch IDs.')
    parser.add_argument('--test_batch_ids',
                        type=str,
                        nargs='+',
                        default=['49_batch0',
                                 '59_batch0',
                                 '59_batch3',
                                 '59_batch6',
                                 '59_batch7',
                                 '59_batch2',
                                 '59_batch4',
                                 '59_batch1',
                                 '59_batch5',
                                 '78_batch3',
                                 '78_batch10',
                                 '78_batch1',
                                 '78_batch0',
                                 '78_batch2',
                                 '78_batch6',
                                 '78_batch7',
                                 '78_batch9',
                                 '78_batch8',
                                 '78_batch4',
                                 '78_batch5',
                                 '1000_batch4',
                                 '1000_batch6',
                                 '1000_batch11',
                                 '1000_batch12',
                                 '1000_batch18',
                                 '1000_batch19',
                                 '1000_batch21',
                                 '1000_batch22',
                                 '1000_batch34',
                                 '1000_batch38',
                                 '1000_batch20',
                                 '1000_batch26',
                                 '1000_batch27',
                                 '1000_batch32',
                                 '1000_batch36',
                                 '1002_batch3',
                                 '1002_batch15'
                                 '1002_batch27',
                                 '1002_batch39',
                                 '1002_batch45',
                                 '1002_batch26',
                                 '1002_batch4',
                                 '1002_batch16'
                                 '1002_batch18',
                                 '1002_batch23',
                                 '1002_batch31',
                                 '1002_batch34',
                                 '1002_batch53',
                                 '1002_batch55',
                                 '1002_batch40',
                                 '1002_batch46',
                                 '1007_batch1',
                                 '1007_batch14',
                                 '1007_batch20',
                                 '1007_batch36',
                                 '1007_batch17',
                                 '1007_batch33',
                                 '1007_batch4',
                                 '1007_batch2',
                                 '1007_batch0',
                                 '1007_batch27',
                                 '1007_batch3',
                                 '1007_batch12',
                                 '1007_batch19',
                                 '1007_batch24',
                                 '1007_batch6',
                                 '1007_batch25',
                                 '1007_batch34',
                                 '1007_batch30',
                                 '1007_batch39',
                                 '1007_batch22'],
                        help='Test batch IDs.')
    args = parser.parse_args()
    
    main(args)
