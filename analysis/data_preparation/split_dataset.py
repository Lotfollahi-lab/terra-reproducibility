import argparse
import pickle

from datasets import load_from_disk
from terra.datasets.utils import save_precomputed_split

from heldout_batch_ids import TEST_BATCH_IDS, VAL_BATCH_IDS


def main(args):
    # Load dataset and retrieve cell IDs
    dataset = load_from_disk(args.tokenized_data_folder_path)

    if args.full_dataset:
        indices = list(range(len(dataset)))
        # The split file records the dataset it was made for, so training
        # refuses it for another tokenization (TERRA save_precomputed_split)
        save_precomputed_split(
            dataset, indices,
            f'{args.tokenizer_folder_path}/{args.tokenized_data_folder_path.rsplit("/", 1)[-1].removesuffix(".dataset")}_precomputed_split_full.pkl')
        print("Saved full dataset split.")

    else:       
        # Only the cell_id column is read, so padded and compact rows
        # (TERRA storage_format 'compact') are split alike
        cell_ids = list(dataset['cell_id'])

        validation_indices = []
        test_indices = []

        print('done0')
        print(len(cell_ids))

        # Single loop to determine validation and test indices. A cell's
        # batch id is '<dataset_id>_<batch>', the first two fields of its
        # cell id; validation takes precedence over test, as before. (Set
        # lookups instead of comparing every cell with every listed id: the
        # same indices, minutes instead of hours for 110M cells.)
        val_batch_ids = set(args.val_batch_ids)
        test_batch_ids = set(args.test_batch_ids)
        batch_ids = set()
        for index, cell_id in enumerate(cell_ids):
            fields = cell_id.split('_')
            batch_id = f"{fields[0]}_{fields[1]}"
            batch_ids.add(batch_id)
            if batch_id in val_batch_ids:
                validation_indices.append(index)
            elif batch_id in test_batch_ids:
                test_indices.append(index)
        print('done1')

        # Every listed held-out batch must match cells (a listed id that
        # matches nothing, e.g. a batch number that does not exist, leaves
        # the batch meant to be held out in training)
        unmatched = sorted(
            (set(args.val_batch_ids) | set(args.test_batch_ids)) - batch_ids)
        if unmatched:
            raise ValueError(
                f'{len(unmatched)} listed held-out batch ids match no cell of '
                f'the dataset: {unmatched}')

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
        #train_unique = {
        #    f"{cell_ids[i].split('_')[0]}_{cell_ids[i].split('_')[1]}" for i in train_indices}
        #validation_unique = {
        #    f"{cell_ids[i].split('_')[0]}_{cell_ids[i].split('_')[1]}" for i in validation_indices}
        #test_unique = {
        #    f"{cell_ids[i].split('_')[0]}_{cell_ids[i].split('_')[1]}" for i in test_indices}
        #print("Train unique identifiers:", train_unique)
        #print("Validation unique identifiers:", validation_unique)
        #print("Test unique identifiers:", test_unique)
        #print('done4')

        # Save the splits as pickle files
        # The training split records the held-out batches it excludes, so
        # training need not scan every cell id to check it
        save_precomputed_split(
            dataset, train_indices,
            f'{args.tokenizer_folder_path}/{args.tokenized_data_folder_path.rsplit("/", 1)[-1].removesuffix(".dataset")}_precomputed_split_train.pkl',
            excluded_batch_ids=list(args.val_batch_ids) + list(args.test_batch_ids))
        save_precomputed_split(
            dataset, validation_indices,
            f'{args.tokenizer_folder_path}/{args.tokenized_data_folder_path.rsplit("/", 1)[-1].removesuffix(".dataset")}_precomputed_split_validation.pkl')
        save_precomputed_split(
            dataset, test_indices,
            f'{args.tokenizer_folder_path}/{args.tokenized_data_folder_path.rsplit("/", 1)[-1].removesuffix(".dataset")}_precomputed_split_test.pkl')
        print("Saved train, validation, and test dataset splits.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--tokenizer_folder_path',
                        type=str,
                        default='/nfs/team361/sb75/DATASETS/tokenizer',
                        help='Tokenizer folder path.')
    parser.add_argument('--tokenized_data_folder_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/hst_corpus_110m/hst_corpus_110m_32_None_None_None_None_None_shifted_log_knn_10.dataset',
                        help='Tokenized data directory path.')
    parser.add_argument('--cohort_name',
                        type=str,
                        default="hst_corpus_110m",
                        help='Name of the cohort.')
    parser.add_argument('--val_batch_ids',
                        type=str,
                        nargs='+',
                        default=VAL_BATCH_IDS,
                        help='Validation batch IDs.')
    parser.add_argument('--test_batch_ids',
                        type=str,
                        nargs='+',
                        default=TEST_BATCH_IDS,
                        help='Test batch IDs.')
    parser.add_argument("--full_dataset",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Flag whether full dataset index should be saved instead of splits.')                         
    args = parser.parse_args()
    
    main(args)
