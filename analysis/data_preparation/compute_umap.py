import argparse
import time
from datetime import timedelta

import scanpy as sc


# Define color palette
custom_palette_300 = {
    "0": "#F24848",
    "1": "#48F2F2",
    "2": "#F24C48",
    "3": "#48EFF2",
    "4": "#F24F48",
    "5": "#48EAF2",
    "6": "#F25548",
    "7": "#48E5F2",
    "8": "#F25B48",
    "9": "#48E0F2",
    "10": "#F26048",
    "11": "#48DBF2",
    "12": "#F26648",
    "13": "#48D5F2",
    "14": "#F26B48",
    "15": "#48D0F2",
    "16": "#F27048",
    "17": "#48CAF2",
    "18": "#F27548",
    "19": "#48C5F2",
    "20": "#F27A48",
    "21": "#48BFF2",
    "22": "#F27F48",
    "23": "#48BAF2",
    "24": "#F28348",
    "25": "#48B4F2",
    "26": "#F28848",
    "27": "#48AEF2",
    "28": "#F28C48",
    "29": "#48A8F2",
    "30": "#F29048",
    "31": "#48A2F2",
    "32": "#F29448",
    "33": "#489CF2",
    "34": "#F29848",
    "35": "#4897F2",
    "36": "#F29C48",
    "37": "#4891F2",
    "38": "#F29F48",
    "39": "#488BF2",
    "40": "#F2A348",
    "41": "#4885F2",
    "42": "#F2A648",
    "43": "#4880F2",
    "44": "#F2AA48",
    "45": "#487AF2",
    "46": "#F2AD48",
    "47": "#4874F2",
    "48": "#F2B048",
    "49": "#486EF2",
    "50": "#F2B348",
    "51": "#4869F2",
    "52": "#F2B648",
    "53": "#4863F2",
    "54": "#F2B948",
    "55": "#485DF2",
    "56": "#F2BB48",
    "57": "#4858F2",
    "58": "#F2BE48",
    "59": "#4852F2",
    "60": "#F2C148",
    "61": "#484CF2",
    "62": "#F2C348",
    "63": "#4847F2",
    "64": "#F2C548",
    "65": "#4841F2",
    "66": "#F2C748",
    "67": "#483BF2",
    "68": "#F2C948",
    "69": "#4836F2",
    "70": "#F2CB48",
    "71": "#4830F2",
    "72": "#F2CC48",
    "73": "#482AF2",
    "74": "#F2CE48",
    "75": "#4824F2",
    "76": "#F2D048",
    "77": "#481FF2",
    "78": "#F2D248",
    "79": "#4819F2",
    "80": "#F2D348",
    "81": "#4813F2",
    "82": "#F2D548",
    "83": "#480DF2",
    "84": "#F2D748",
    "85": "#4808F2",
    "86": "#F2D848",
    "87": "#4802F2",
    "88": "#F2DA48",
    "89": "#47FCF2",
    "90": "#F2DB48",
    "91": "#47F6F2",
    "92": "#F2DC48",
    "93": "#47F1F2",
    "94": "#F2DE48",
    "95": "#47EBF2",
    "96": "#F2DF48",
    "97": "#47E5F2",
    "98": "#F2E048",
    "99": "#47E0F2",
    "100": "#F2E148",
    "101": "#47DAF2",
    "102": "#F2E248",
    "103": "#47D5F2",
    "104": "#F2E348",
    "105": "#47CFF2",
    "106": "#F2E448",
    "107": "#47CAF2",
    "108": "#F2E548",
    "109": "#47C4F2",
    "110": "#F2E648",
    "111": "#47BFF2",
    "112": "#F2E748",
    "113": "#47B9F2",
    "114": "#F2E848",
    "115": "#47B3F2",
    "116": "#F2E948",
    "117": "#47ADF2",
    "118": "#F2EA48",
    "119": "#47A8F2",
    "120": "#F2EB48",
    "121": "#47A2F2",
    "122": "#F2EC48",
    "123": "#479CF2",
    "124": "#F2ED48",
    "125": "#4797F2",
    "126": "#F2EE48",
    "127": "#4791F2",
    "128": "#F2EF48",
    "129": "#478BF2",
    "130": "#F2F048",
    "131": "#4786F2",
    "132": "#F2F148",
    "133": "#4780F2",
    "134": "#F2F248",
    "135": "#477AF2",
    "136": "#F2F348",
    "137": "#4774F2",
    "138": "#F2F448",
    "139": "#476EF2",
    "140": "#F2F548",
    "141": "#4769F2",
    "142": "#F2F648",
    "143": "#4763F2",
    "144": "#F2F748",
    "145": "#475DF2",
    "146": "#F2F848",
    "147": "#4758F2",
    "148": "#F2F948",
    "149": "#4752F2",
    "150": "#F2FA48",
    "151": "#474CF2",
    "152": "#F2FB48",
    "153": "#4747F2",
    "154": "#F2FC48",
    "155": "#4741F2",
    "156": "#F2FD48",
    "157": "#473BF2",
    "158": "#F2FE48",
    "159": "#4736F2",
    "160": "#F2FF48",
    "161": "#4730F2",
    "162": "#F30048",
    "163": "#472AF2",
    "164": "#F30148",
    "165": "#4724F2",
    "166": "#F30248",
    "167": "#471FF2",
    "168": "#F30348",
    "169": "#4719F2",
    "170": "#F30448",
    "171": "#4713F2",
    "172": "#F30548",
    "173": "#470DF2",
    "174": "#F30648",
    "175": "#4708F2",
    "176": "#F30748",
    "177": "#4702F2",
    "178": "#F30848",
    "179": "#46FCF2",
    "180": "#F30948",
    "181": "#46F6F2",
    "182": "#F30A48",
    "183": "#46F1F2",
    "184": "#F30B48",
    "185": "#46EBF2",
    "186": "#F30C48",
    "187": "#46E5F2",
    "188": "#F30D48",
    "189": "#46E0F2",
    "190": "#F30E48",
    "191": "#46DAF2",
    "192": "#F30F48",
    "193": "#46D5F2",
    "194": "#F31048",
    "195": "#46CFF2",
    "196": "#F31148",
    "197": "#46CAF2",
    "198": "#F31248",
    "199": "#46C4F2",
    "200": "#F31348",
    "201": "#46BFF2",
    "202": "#F31448",
    "203": "#46B9F2",
    "204": "#F31548",
    "205": "#46B3F2",
    "206": "#F31648",
    "207": "#46ADF2",
    "208": "#F31748",
    "209": "#46A8F2",
    "210": "#F31848",
    "211": "#46A2F2",
    "212": "#F31948",
    "213": "#469CF2",
    "214": "#F31A48",
    "215": "#4697F2",
    "216": "#F31B48",
    "217": "#4691F2",
    "218": "#F31C48",
    "219": "#468BF2",
    "220": "#F31D48",
    "221": "#4686F2",
    "222": "#F31E48",
    "223": "#4680F2",
    "224": "#F31F48",
    "225": "#467AF2",
    "226": "#F32048",
    "227": "#4674F2",
    "228": "#F32148",
    "229": "#466EF2",
    "230": "#F32248",
    "231": "#4669F2",
    "232": "#F32348",
    "233": "#4663F2",
    "234": "#F32448",
    "235": "#465DF2",
    "236": "#F32548",
    "237": "#4658F2",
    "238": "#F32648",
    "239": "#4652F2",
    "240": "#F32748",
    "241": "#464CF2",
    "242": "#F32848",
    "243": "#4647F2",
    "244": "#F32948",
    "245": "#4641F2",
    "246": "#F32A48",
    "247": "#463BF2",
    "248": "#F32B48",
    "249": "#4636F2",
    "250": "#F32C48",
    "251": "#4630F2",
    "252": "#F32D48",
    "253": "#462AF2",
    "254": "#F32E48",
    "255": "#4624F2",
    "256": "#F32F48",
    "257": "#461FF2",
    "258": "#F33048",
    "259": "#4619F2",
    "260": "#F33148",
    "261": "#4613F2",
    "262": "#F33248",
    "263": "#460DF2",
    "264": "#F33348",
    "265": "#4608F2",
    "266": "#F33448",
    "267": "#4602F2",
    "268": "#F33548",
    "269": "#45FCF2",
    "270": "#F33648",
    "271": "#45F6F2",
    "272": "#F33748",
    "273": "#45F1F2",
    "274": "#F33848",
    "275": "#45EBF2",
    "276": "#F33948",
    "277": "#45E5F2",
    "278": "#F33A48",
    "279": "#45E0F2",
    "280": "#F33B48",
    "281": "#45DAF2",
    "282": "#F33C48",
    "283": "#45D5F2",
    "284": "#F33D48",
    "285": "#45CFF2",
    "286": "#F33E48",
    "287": "#45CAF2",
    "288": "#F33F48",
    "289": "#45C4F2",
    "290": "#F34048",
    "291": "#45BFF2",
    "292": "#F34148",
    "293": "#45B9F2",
    "294": "#F34248",
    "295": "#45B3F2",
    "296": "#F34348",
    "297": "#45ADF2",
    "298": "#F34448",
    "299": "#45A8F2",
    "-1" : "#E1D9D1",
    "None" : "#E1D9D1"
}


# Setup argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Compute UMAP.')
    parser.add_argument('--load_timestamp',
                        type=str,
                        default='19022025_023508_679',
                        help='Load timestamp of the model.')
    parser.add_argument('--read_checkpoint',
                        type=str,
                        default='ep0',
                        help='Read checkpoint of the model.')
    parser.add_argument('--corpus_size',
                        type=int,
                        default=72000000,
                        help='Size of the entire corpus.')
    parser.add_argument('--step_size',
                        type=int,
                        default=2000000,
                        help='Step size during inference.')
    parser.add_argument('--subsample',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='If `True`, subsample adata stratified by batch.')
    parser.add_argument('--sample_frac',
                        type=float,
                        default=0.2,
                        help='Subsample fraction for batch-stratified split if \
                              specified.')
    parser.add_argument('--random_state',
                        type=int,
                        default=42,
                        help='Random state.')
    parser.add_argument('--emb_key',
                        type=str,
                        default='neighborhood_emb_layer_6',
                        help='`adata.obsm` key where embeddings are stored.')
    parser.add_argument('--n_pcs',
                        type=int,
                        default=50,
                        help='Number of principal components.')
    parser.add_argument('--dim_red_method',
                        type=str,
                        default="umap",
                        help='Dimensionality reduction method. Either `umap` or `tsne`')
    return parser.parse_args()


def main(args):
    artifacts_folder_path = '/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/artifacts'
    steps = int(args.corpus_size / args.step_size)

    adata_list = []

    for i in range(steps):
        try:
            adata = sc.read_h5ad(f'{artifacts_folder_path}/hst_corpus_70m/{load_timestamp}/adata_{read_checkpoint}_{step_size}_{i}.h5ad')
            if emb_key == 'cell_emb_layer_6':
                del adata.obsm['neighborhood_emb_layer_6']
            elif emb_key == 'neighborhood_emb_layer_6':
                del adata.obsm['cell_emb_layer_6']
            adata_list.append(adata)
            del adata
        except:
            print(f'Could not find file with file name adata_{read_checkpoint}_{step_size}_{i}.h5ad.')

    adata_full = sc.concat(adata_list, join='outer', index_unique=None)

    start_time = time.time()
    print("Start time:", start_time)

    if args.subsample:
        # Perform stratified subsampling based on batch
        adata_full.obs[['dataset', 'batch', 'cell']] = [
            s.split('_') for s in adata_full.obs.index]
        adata_full.obs['dataset_batch'] = adata_full.obs[
            'dataset'].astype(str) + adata_full.obs['batch'].astype(str)
        adata_full.obs['cell'] = adata_full.obs['cell'].astype(int)

        # Stratified sampling: sample 20% from each batch
        sampled_df = adata_full.obs.groupby(
            ['dataset', 'batch'], group_keys=False).apply(
                lambda x: x.sample(
                    frac=sample_frac, replace=False, random_state=random_state))
        adata_full = adata_full[sampled_df.index]
        del(sampled_df)

    if args.n_pcs > 0:
        # Compute neighbors and UMAP 
        sc.pp.neighbors(adata_full,
                        n_neighbors=15,
                        use_rep=args.emb_key,
                        n_pcs=args.n_pcs,
                        key_added='neighborhood')
    else:
        # Compute neighbors and UMAP 
        sc.pp.neighbors(adata_full,
                        n_neighbors=15,
                        use_rep=args.emb_key,
                        key_added='neighborhood')        

    intermediate_time = time.time()
    print("Intermediate time:", intermediate_time)

    elapsed_time = intermediate_time - start_time
    print("Elapsed time:", str(timedelta(seconds=elapsed_time)))

    adata_full.write(f'{artifacts_folder_path}/temp1.h5ad')

    if args.dim_red_method == 'umap':
        sc.tl.umap(adata_full,
                   neighbors_key='neighborhood')
    #elif args.dim_red_method == 'tsne':
    #    sc.tl.tsne(adata_full,
    #               use_rep=args.emb_key)

        dataset_batch_palette = {new_key: value for new_key, value in zip(
            adata_full.obs['dataset_batch'].unique().tolist(),
            custom_palette_300.values())}

        labels = ['assay', 'batch', 'tissue', 'gene_panel_size']
        # Plot (aggregated) neighborhood gene embeddings, colored by labels
        for label in labels:
            sc.pl.umap(
                adata_full,
                color=label,
                palette=(dataset_batch_palette if label == 'dataset_batch' else None),
                show=False)
            plt.savefig(
                f'{artifacts_folder_path}/hst_corpus_70m/{load_timestamp}/hst_corpus_70m_{sample_frac}_subsample_{label}_umap.svg',
                dpi=300,
                bbox_inches='tight')

    end_time = time.time()
    print("End time:", end_time)

    elapsed_time = end_time - start_time
    print("Elapsed time:", str(timedelta(seconds=elapsed_time)))

    adata_full.write(f'{artifacts_folder_path}/temp2.h5ad')


if __name__ == '__main__':

    args = parse_arguments()
    main(args)