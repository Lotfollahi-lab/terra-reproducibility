import argparse
import pickle

from datasets import load_from_disk


# Helper function to save n_nonzero_tokens
def save_n_nonzero_tokens(dataset, indices, output_path):
    # Select entries based on indices
    selected_data = dataset.select(indices)
    # Extract 'n_nonzero_tokens' and save as a list
    n_nonzero_tokens = selected_data['n_nonzero_tokens']
    with open(output_path, 'wb') as f:
        pickle.dump(n_nonzero_tokens, f)


def main(args):
    # Load dataset
    dataset = load_from_disk(args.tokenized_data_folder_path)

    # Load split indices
    with open(args.train_indices_path, 'rb') as f:
        train_indices = pickle.load(f)
    with open(args.validation_indices_path, 'rb') as f:
        validation_indices = pickle.load(f)
    with open(args.test_indices_path, 'rb') as f:
        test_indices = pickle.load(f)

    # Save n_nonzero_tokens for train, validation, and test
    save_n_nonzero_tokens(
        dataset,
        validation_indices,
        'n_nonzero_tokens_validation.pkl')
    print('done1')
    save_n_nonzero_tokens(dataset, train_indices, 'n_nonzero_tokens_train.pkl')
    print('done2')
    save_n_nonzero_tokens(dataset, test_indices, 'n_nonzero_tokens_test.pkl')
    print('done3')

    print("Saved n_nonzero_tokens for train, validation, and test datasets.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--tokenized_data_folder_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/team361/DATASETS/gold/cell-graph-tokenizer/human_cohort2_60m/human_cohort2_60m_32_None_None_None_gene_corrected_read_depth_None_shifted_log_knn_10.dataset',
                        help='Tokenized data directory path.')
    parser.add_argument('--train_indices_path',
                        type=str,
                        default='precomputed_split_train.pkl',
                        help='Path to pickle file containing train indices.')
    parser.add_argument('--validation_indices_path',
                        type=str,
                        default='precomputed_split_validation.pkl',
                        help='Path to pickle file containing validation indices.')
    parser.add_argument('--test_indices_path',
                        type=str,
                        default='precomputed_split_test.pkl',
                        help='Path to pickle file containing test indices.')
    args = parser.parse_args()
    
    main(args)
