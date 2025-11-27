import argparse

from datasets import concatenate_datasets, load_from_disk


def main(args):
    print("Start script.")

    # Load Huggingface dataset
    ds = load_from_disk(args.dataset_path)

    start = 70438096
    end = 70853334

    ds1 = ds.select(range(0, start))
    ds2 = ds.select(range(end, len(ds)))

    ds = concatenate_datasets([ds1, ds2])

    new_path = args.dataset_path.replace('.dataset', '_fixed.dataset')
    ds.save_to_disk(new_path)
    print("Saved to:", new_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--dataset_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/hst_corpus_110m/hst_corpus_110m_32_None_None_None_None_None_shifted_log_knn_10.dataset',
                        help='Path where the Huggingface dataset will be loaded from.')
    args = parser.parse_args()

    main(args)

