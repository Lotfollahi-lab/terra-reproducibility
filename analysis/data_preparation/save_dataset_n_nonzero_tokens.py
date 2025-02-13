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
    with open(f'/lustre/scratch126/cellgen/team361/DATASETS/tokenizer/{args.tokenized_data_folder_path.rsplit("/", 1)[-1].removesuffix(".dataset")}_precomputed_split_train.pkl', 'rb') as f:
        train_indices = pickle.load(f)
    with open(f'/lustre/scratch126/cellgen/team361/DATASETS/tokenizer/{args.tokenized_data_folder_path.rsplit("/", 1)[-1].removesuffix(".dataset")}_precomputed_split_validation.pkl', 'rb') as f:
        validation_indices = pickle.load(f)
    with open(f'/lustre/scratch126/cellgen/team361/DATASETS/tokenizer/{args.tokenized_data_folder_path.rsplit("/", 1)[-1].removesuffix(".dataset")}_precomputed_split_test.pkl', 'rb') as f:
        test_indices = pickle.load(f)

    # Save n_nonzero_tokens for train, validation, and test
    save_n_nonzero_tokens(
        dataset,
        validation_indices,
        f'/lustre/scratch126/cellgen/team361/DATASETS/tokenizer/{args.tokenized_data_folder_path.rsplit("/", 1)[-1].removesuffix(".dataset")}_n_nonzero_tokens_validation.pkl')
    print('done1')
    save_n_nonzero_tokens(
        dataset,
        train_indices,
        f'/lustre/scratch126/cellgen/team361/DATASETS/tokenizer/{args.tokenized_data_folder_path.rsplit("/", 1)[-1].removesuffix(".dataset")}_n_nonzero_tokens_train.pkl')
    print('done2')
    save_n_nonzero_tokens(
        dataset,
        test_indices,
        f'/lustre/scratch126/cellgen/team361/DATASETS/tokenizer/{args.tokenized_data_folder_path.rsplit("/", 1)[-1].removesuffix(".dataset")}_n_nonzero_tokens_test.pkl')
    print('done3')
    print("Saved n_nonzero_tokens for train, validation, and test datasets.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--tokenized_data_folder_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/team361/DATASETS/gold/cell-graph-tokenizer/hst_corpus_70m/hst_corpus_70m_32_None_None_None_None_None_shifted_log_knn_6.dataset',
                        help='Tokenized data directory path.')
    parser.add_argument('--cohort_name',
                        type=str,
                        default="hst_corpus_70m",
                        help='Name of the cohort.')
    args = parser.parse_args()
    
    main(args)
