import argparse

from datasets import load_from_disk


def main(args):
    print("Start script.")

    # Load Huggingface dataset
    ds = load_from_disk(args.dataset_path)

    ds = ds.remove_columns("seg_tokens")
    ds = ds.remove_columns("batch_token")
    ds = ds.remove_columns("gene_panel_token")
    ds = ds.remove_columns("assay_token")
    ds = ds.remove_columns("species_token")
    ds = ds.remove_columns("tissue_token")

    def keep_every_256(batch, n_segments=11):
        return {
            "rel_x_coord": [
                x[::256] if len(x) > n_segments else x
                for x in batch["rel_x_coord"]
            ],
            "rel_y_coord": [
                y[::256] if len(y) > n_segments else y
                for y in batch["rel_y_coord"]
            ],
        }

    ds = ds.map(
        keep_every_256,
        batched=True,
        num_proc=12
    )

    new_path = args.dataset_path.replace('.dataset', '_fixed.dataset')
    ds.save_to_disk(new_path)
    print("Saved to:", new_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--dataset_path',
                        type=str,
                        default='/lustre/scratch126/cellgen/lotfollahi/DATASETS/gold/cell-graph-tokenizer/hst_corpus_90m/hst_corpus_90m_32_None_None_None_None_None_shifted_log_knn_10.dataset',
                        help='Path where the Huggingface dataset will be loaded from.')
    args = parser.parse_args()

    main(args)

