import argparse
import time
from datetime import timedelta

import scanpy as sc


# Setup argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Compute UMAP.')
    parser.add_argument('--n_pcs', type=int, default=50,
                        help='Number of principal components.')
    return parser.parse_args()


def main(args):
    artifacts_folder_path = '/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/artifacts'
    load_timestamp = '19022025_023508_679'
    step_size = '2000000'
    read_checkpoint = 'ep0'
    emb_key = 'neighborhood_emb_layer_6'


    adata_list = []

    for j in range(9):
        for i in range(4):
            adata = sc.read_h5ad(f'{artifacts_folder_path}/hst_corpus_70m/{load_timestamp}/adata_{read_checkpoint}_{step_size}_{i}.h5ad')
            if emb_key == 'cell_emb_layer_6':
                del adata.obsm['neighborhood_emb_layer_6']
            elif emb_key == 'neighborhood_emb_layer_6':
                del adata.obsm['cell_emb_layer_6']
            adata_list.append(adata)
            del adata

    adata_full = sc.concat(adata_list, join='outer', index_unique=None)

    start_time = time.time()
    print("Start time:", start_time)

    if args.n_pcs > 0:
        # Compute neighbors and UMAP 
        sc.pp.neighbors(adata_full,
                        n_neighbors=15,
                        use_rep=emb_key,
                        n_pcs=args.n_pcs,
                        key_added='neighborhood')
    else:
        # Compute neighbors and UMAP 
        sc.pp.neighbors(adata_full,
                        n_neighbors=15,
                        use_rep=emb_key,
                        key_added='neighborhood')        

    intermediate_time = time.time()
    print("Intermediate time:", intermediate_time)

    elapsed_time = intermediate_time - start_time
    print("Elapsed time:", str(timedelta(seconds=elapsed_time)))

    adata.write(f'{artifacts_folder_path}/temp1.h5ad')

    sc.tl.umap(adata_full,
               neighbors_key='neighborhood')

    end_time = time.time()
    print("End time:", end_time)

    elapsed_time = end_time - start_time
    print("Elapsed time:", str(timedelta(seconds=elapsed_time)))

    adata.write(f'{artifacts_folder_path}/temp2.h5ad')


if __name__ == '__main__':

    args = parse_arguments()
    main(args)