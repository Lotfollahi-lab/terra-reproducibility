import argparse
import pickle

from datasets import load_from_disk


def main(args):
    # Load dataset and retrieve cell IDs
    dataset = load_from_disk(args.tokenized_data_folder_path)

    cell_ids = list(dataset['cell_id'])

    keep_mask = [
        (cell_id.split('_')[0] + '_' + cell_id.split('_')[1]
        ) in args.keep_batch_ids for cell_id in cell_ids]
    keep_idx = [i for i, b in enumerate(keep_mask) if b]

    with open(f'{tokenizer_folder_path}/{args.dataset_name}_precomputed_split.pkl', 'wb') as f:
        pickle.dump(keep_idx, f)
    print(f"Saved dataset indices for dataset {args.dataset_name}.")


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
    parser.add_argument('--dataset_name',
                        type=str,
                        default="xhs1000_test",
                        help='Name of the dataset.')
    parser.add_argument('--keep_batch_ids',
                        type=str,
                        nargs='+',
                        default=['1000_batch11',
                                 '1000_batch19',
                                 '1000_batch32'],
                        help='Validation batch IDs.')        
    args = parser.parse_args()
    
    main(args)
