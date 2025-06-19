import argparse
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


n_layer = 12

# Define color palette
custom_palette_300 = {
    "0": "#5E67F7",
    "1": "#F54349",
    "2": "#D28E59",
    "3": "#920828",
    "4": "#07E10B",
    "5": "#13C35C",
    "6": "#91C83F",
    "7": "#61C7CC",
    "8": "#175DA6",
    "9": "#278729",
    "10": "#9A4B9B",
    "11": "#3516AA",
    "12": "#F223BF",
    "13": "#459A8C",
    "14": "#DADD17",
    "15": "#EF7E60",
    "16": "#5A58D5",
    "17": "#C22A30",
    "18": "#E96B13",
    "19": "#00DABE",
    "20": "#4DD8C1",
    "21": "#73A341",
    "22": "#3FD824",
    "23": "#9A0EA1",
    "24": "#338B6F",
    "25": "#D2C71E",
    "26": "#D660CF",
    "27": "#5346EE",
    "28": "#583581",
    "29": "#FC0F7F",
    "30": "#CC3335",
    "31": "#63CB9F",
    "32": "#88582B",
    "33": "#46FDF2",
    "34": "#DB8042",
    "35": "#0BDC84",
    "36": "#A5BD18",
    "37": "#F97487",
    "38": "#6F5ADA",
    "39": "#131591",
    "40": "#6EFD5F",
    "41": "#66AE4A",
    "42": "#A14E9D",
    "43": "#13B305",
    "44": "#E333E9",
    "45": "#46AAEE",
    "46": "#09868F",
    "47": "#CDE363",
    "48": "#9F07F3",
    "49": "#AB3324",
    "50": "#6647D1",
    "51": "#691E84",
    "52": "#8BB61B",
    "53": "#61FAAE",
    "54": "#E9A034",
    "55": "#E526AC",
    "56": "#1C871F",
    "57": "#3F8F73",
    "58": "#1DEBF4",
    "59": "#F93802",
    "60": "#9B4E21",
    "61": "#546BBB",
    "62": "#6C40FE",
    "63": "#76DD67",
    "64": "#F549F5",
    "65": "#B9062B",
    "66": "#C99961",
    "67": "#B64489",
    "68": "#21CE92",
    "69": "#10FCD1",
    "70": "#080ADF",
    "71": "#917816",
    "72": "#1082C9",
    "73": "#3A867D",
    "74": "#68F8CD",
    "75": "#E7E24A",
    "76": "#6A1B84",
    "77": "#F69054",
    "78": "#6E79F8",
    "79": "#ED3145",
    "80": "#55F474",
    "81": "#93B74F",
    "82": "#C96CF1",
    "83": "#8B083E",
    "84": "#3A6285",
    "85": "#069839",
    "86": "#018BB5",
    "87": "#23C8DE",
    "88": "#3E1AC0",
    "89": "#F97C95",
    "90": "#CCF55C",
    "91": "#B8506E",
    "92": "#C1BE16",
    "93": "#1AEB78",
    "94": "#78AE2E",
    "95": "#479067",
    "96": "#645FC3",
    "97": "#6DFD31",
    "98": "#A12D00",
    "99": "#E560DB",
    "100": "#DD7231",
    "101": "#020486",
    "102": "#E706AA",
    "103": "#29A5FD",
    "104": "#57FED0",
    "105": "#DCC95E",
    "106": "#1CF164",
    "107": "#0B34F5",
    "108": "#515CCD",
    "109": "#833F16",
    "110": "#11966B",
    "111": "#3BC321",
    "112": "#E87970",
    "113": "#DD3BA9",
    "114": "#B3E21F",
    "115": "#901983",
    "116": "#79A1F2",
    "117": "#E38A15",
    "118": "#0A33AE",
    "119": "#16E695",
    "120": "#8A20F4",
    "121": "#5B5CDE",
    "122": "#FA71E8",
    "123": "#1BFCEA",
    "124": "#47CA46",
    "125": "#86390A",
    "126": "#237888",
    "127": "#99B022",
    "128": "#76409A",
    "129": "#BE4D5B",
    "130": "#C3E759",
    "131": "#1B19DC",
    "132": "#B61AC7",
    "133": "#E369BD",
    "134": "#460384",
    "135": "#8858EB",
    "136": "#66CEFB",
    "137": "#1BAF95",
    "138": "#03CED9",
    "139": "#B2C10C",
    "140": "#8C366C",
    "141": "#41629A",
    "142": "#62A046",
    "143": "#B3F754",
    "144": "#CE2194",
    "145": "#7204CD",
    "146": "#2676E2",
    "147": "#A01F08",
    "148": "#04D665",
    "149": "#DA454E",
    "150": "#A16828",
    "151": "#391780",
    "152": "#6ABDFE",
    "153": "#0DF9B3",
    "154": "#FE2B02",
    "155": "#B0229D",
    "156": "#9BEC72",
    "157": "#933AE8",
    "158": "#48C83B",
    "159": "#F36AC5",
    "160": "#024BA9",
    "161": "#A3254D",
    "162": "#268D93",
    "163": "#87A80C",
    "164": "#34F8A3",
    "165": "#EB2D0C",
    "166": "#4A00FE",
    "167": "#60BCCF",
    "168": "#E1B000",
    "169": "#842CA0",
    "170": "#6550ED",
    "171": "#854F0C",
    "172": "#F08345",
    "173": "#39C36B",
    "174": "#FA16E9",
    "175": "#7BFE0D",
    "176": "#C75D83",
    "177": "#261090",
    "178": "#27ACC7",
    "179": "#1F07F0",
    "180": "#71B9FD",
    "181": "#A99C13",
    "182": "#29DA04",
    "183": "#713AD5",
    "184": "#0BAC50",
    "185": "#B13335",
    "186": "#65C64F",
    "187": "#FC0A0D",
    "188": "#D6639D",
    "189": "#A211A1",
    "190": "#2B529F",
    "191": "#48CCBA",
    "192": "#B7E83C",
    "193": "#0528E3",
    "194": "#E149F2",
    "195": "#2B9072",
    "196": "#84014C",
    "197": "#BF6D1E",
    "198": "#0394CA",
    "199": "#6EE96F",
    "200": "#EE1797",
    "201": "#5C2FAD",
    "202": "#91AF0E",
    "203": "#63A2F9",
    "204": "#1FFC47",
    "205": "#ADEB3D",
    "206": "#0B4680",
    "207": "#610CFB",
    "208": "#F57302",
    "209": "#057ACA",
    "210": "#6DF6B9",
    "211": "#449C63",
    "212": "#A42BD4",
    "213": "#1DD1D0",
    "214": "#891539",
    "215": "#6B9CEE",
    "216": "#D2B656",
    "217": "#9D8A09",
    "218": "#15DD2C",
    "219": "#22FC90",
    "220": "#6EF13F",
    "221": "#571A9E",
    "222": "#0E3DC0",
    "223": "#4CFCFB",
    "224": "#DF3D75",
    "225": "#8E55D7",
    "226": "#B83A21",
    "227": "#308E45",
    "228": "#35B29C",
    "229": "#00C5F2",
    "230": "#08DE16",
    "231": "#AA9739",
    "232": "#652E84",
    "233": "#1D0996",
    "234": "#62D535",
    "235": "#C85483",
    "236": "#3366B3",
    "237": "#9C4129",
    "238": "#63F2C7",
    "239": "#12915B",
    "240": "#F8A545",
    "241": "#24B4E8",
    "242": "#F35C1C",
    "243": "#11EA51",
    "244": "#900262",
    "245": "#A3CA48",
    "246": "#CCF602",
    "247": "#526BBF",
    "248": "#D01C8B",
    "249": "#58F307",
    "250": "#2A4582",
    "251": "#B78B12",
    "252": "#1D8118",
    "253": "#EEB574",
    "254": "#19918A",
    "255": "#6BFAB4",
    "256": "#731C80",
    "257": "#994207",
    "258": "#0F3EEE",
    "259": "#B00433",
    "260": "#D63D77",
    "261": "#6537D0",
    "262": "#1CE0B9",
    "263": "#BA961D",
    "264": "#49D94C",
    "265": "#F6CD73",
    "266": "#04C0FC",
    "267": "#80C904",
    "268": "#FBC710",
    "269": "#212E8E",
    "270": "#5EEDF4",
    "271": "#BC3FD0",
    "272": "#D73936",
    "273": "#11CD10",
    "274": "#7F5801",
    "275": "#66C55E",
    "276": "#A6007A",
    "277": "#3B8DA7",
    "278": "#F65B91",
    "279": "#D9D40C",
    "280": "#0259D3",
    "281": "#4924D7",
    "282": "#082783",
    "283": "#A48835",
    "284": "#BAE661",
    "285": "#64E1A7",
    "286": "#47CD41",
    "287": "#9A3C1F",
    "288": "#3D96AB",
    "289": "#BD4975",
    "290": "#EA24F7",
    "291": "#B97BFB",
    "292": "#FA1981",
    "293": "#202491",
    "294": "#588C09",
    "295": "#918D40",
    "296": "#55DCDE",
    "297": "#7A48FB",
    "298": "#2F59E9",
    "299": "#41AF73",
    "-1" : "#E1D9D1",
    "None" : "#E1D9D1"
}


# Setup argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Compute UMAP.')
    parser.add_argument('--load_timestamp',
                        type=str,
                        default='06052025_160815_325',
                        help='Load timestamp of the model.')
    parser.add_argument('--read_checkpoint',
                        type=str,
                        default='ep9',
                        help='Read checkpoint of the model.')
    parser.add_argument('--dataset_name',
                        type=str,
                        default='hst_corpus_70m', # '1000-1007-1008', 'hst_corpus_70m'
                        help='Name of the corpus.')
    parser.add_argument('--corpus_size',
                        type=int,
                        default=72000000, # 11036633, 72000000
                        help='Size of the entire corpus.')
    parser.add_argument('--step_size',
                        type=int,
                        default=2000000,
                        help='Step size during inference.')
    parser.add_argument('--subsample',
                        action=argparse.BooleanOptionalAction,
                        default=True, # True, False
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
    parser.add_argument('--emb_keys',
                        type=str,
                        nargs='+',
                        default=[f'neighborhood_emb_layer_{n_layer}', f'cell_emb_layer_{n_layer}'], # [f'neighborhood_emb_layer_{n_layer}', f'cell_emb_layer_{n_layer}']
                        help='`adata.obsm` keys where embeddings are stored.')
    parser.add_argument('--n_pcs',
                        type=int,
                        default=50,
                        help='Number of principal components.')
    parser.add_argument('--dim_red_method',
                        type=str,
                        default="umap",
                        help='Dimensionality reduction method. Either `umap` or `tsne`')
    parser.add_argument('--plot_labels',
                        type=str,
                        nargs='+',
                        default=['assay', 'tissue'], # ['assay', 'batch', 'tissue', 'gene_panel_size']
                        help='Labels to be plotted.')
    return parser.parse_args()


def main(args):

    start_time = time.time()
    print("Start time:", start_time)

    artifacts_folder_path = '/nfs/team361/sb75/nichejepa-reproducibility/artifacts' # '/lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/artifacts'
    steps = int(np.ceil(args.corpus_size / args.step_size))

    adata_list = []

    for i in range(steps):
        try:
            adata = sc.read_h5ad(f'{artifacts_folder_path}/hst_corpus_80m/{args.load_timestamp}/adata_{args.read_checkpoint}{f"_{args.dataset_name}" if args.dataset_name != "hst_corpus_80m" else ""}_{args.step_size}_{i}.h5ad')
            if f'neighborhood_emb_layer_{n_layer}' not in args.emb_keys:
                del adata.obsm[f'neighborhood_emb_layer_{n_layer}']
            if f'cell_emb_layer_{n_layer}' not in args.emb_keys:
                del adata.obsm[f'cell_emb_layer_{n_layer}']
            adata_list.append(adata)
            del adata
        except:
            raise ValueError(
                f'Could not find file with file name adata_{args.read_checkpoint}{f"_{args.dataset_name}" if args.dataset_name != "hst_corpus_70m" else ""}_{args.step_size}_{i}.h5ad.')

    adata_full = sc.concat(adata_list, join='outer', index_unique=None)

    adata_full.obs[['dataset', 'batch', 'cell']] = [s.split('_') for s in adata_full.obs.index]
    adata_full.obs['dataset_batch'] = adata_full.obs['dataset'].astype(str) + adata_full.obs['batch'].astype(str)
    adata_full.obs['cell'] = adata_full.obs['cell'].astype(int)

    adata_full.write(
        f'{artifacts_folder_path}/hst_corpus_70m/{args.load_timestamp}/adata_{args.read_checkpoint}{f"_{args.dataset_name}" if args.dataset_name != "hst_corpus_70m" else ""}_combined.h5ad')

    if args.subsample:
        # Perform stratified subsampling based on batch
        sampled_df = adata_full.obs.groupby(['dataset', 'batch'], group_keys=False).apply(
                lambda x: x.sample(frac=args.sample_frac, replace=False, random_state=args.random_state))
        adata_full = adata_full[sampled_df.index]
        del(sampled_df)

    # Compute neighbors and UMAP 
    if f'neighborhood_emb_layer_{n_layer}' in args.emb_keys:
        sc.pp.neighbors(adata_full,
                        n_neighbors=15,
                        use_rep=f'neighborhood_emb_layer_{n_layer}',
                        n_pcs=(args.n_pcs if args.n_pcs > 0 else None),
                        key_added='neighborhood')
    if f'cell_emb_layer_{n_layer}' in args.emb_keys:
        sc.pp.neighbors(adata_full,
                        n_neighbors=15,
                        use_rep=f'cell_emb_layer_{n_layer}',
                        n_pcs=(args.n_pcs if args.n_pcs > 0 else None),
                        key_added='cell')    

    intermediate_time = time.time()
    print("Intermediate time:", intermediate_time)

    elapsed_time = intermediate_time - start_time
    print("Elapsed time:", str(timedelta(seconds=elapsed_time)))

    adata_full.write(
        f'{artifacts_folder_path}/hst_corpus_70m/{args.load_timestamp}/adata_{args.read_checkpoint}{f"_{args.dataset_name}" if args.dataset_name != "hst_corpus_70m" else ""}_combined.h5ad')

    if args.dim_red_method == 'umap':
        if f'neighborhood_emb_layer_{n_layer}' in args.emb_keys:
            sc.tl.umap(adata_full,
                       neighbors_key='neighborhood',
                       key_added='neighborhood_umap')
        if f'cell_emb_layer_{n_layer}' in args.emb_keys:
            sc.tl.umap(adata_full,
                       neighbors_key='cell',
                       key_added='cell_umap')

        adata_full.write(
                f'{artifacts_folder_path}/hst_corpus_70m/{args.load_timestamp}/adata_{args.read_checkpoint}{f"_{args.dataset_name}" if args.dataset_name != "hst_corpus_70m" else ""}_combined.h5ad')

        dataset_batch_palette = {new_key: value for new_key, value in zip(
            adata_full.obs['dataset_batch'].unique().tolist(),
            custom_palette_300.values())}

        # Plot (aggregated) neighborhood gene embeddings, colored by labels
        for label in args.plot_labels:
            if f'neighborhood_emb_layer_{n_layer}' in args.emb_keys:
                adata_full.obsm['X_umap'] = adata_full.obsm['neighborhood_umap']
                sc.pl.umap(
                    adata_full,
                    neighbors_key='neighborhood',
                    color=label,
                    palette=(dataset_batch_palette if label == 'dataset_batch' else None),
                    show=False)
                plt.savefig(
                    f'{artifacts_folder_path}/hst_corpus_70m/{args.load_timestamp}/{f"{args.dataset_name}_" if args.dataset_name != "hst_corpus_70m" else ""}{label}_neighborhood_umap.png',
                    dpi=300,
                    bbox_inches='tight')
            if f'cell_emb_layer_{n_layer}' in args.emb_keys:
                adata_full.obsm['X_umap'] = adata_full.obsm['cell_umap']
                sc.pl.umap(
                    adata_full,
                    neighbors_key='cell',
                    color=label,
                    palette=(dataset_batch_palette if label == 'dataset_batch' else None),
                    show=False)
                plt.savefig(
                    f'{artifacts_folder_path}/hst_corpus_70m/{args.load_timestamp}/{f"{args.dataset_name}_" if args.dataset_name != "hst_corpus_70m" else ""}{label}_cell_umap.png',
                    dpi=300,
                    bbox_inches='tight')

    end_time = time.time()
    print("End time:", end_time)

    elapsed_time = end_time - start_time
    print("Elapsed time:", str(timedelta(seconds=elapsed_time)))

    adata_full.write(
        f'{artifacts_folder_path}/hst_corpus_70m/{args.load_timestamp}/adata_{args.read_checkpoint}{f"_{args.dataset_name}" if args.dataset_name != "hst_corpus_70m" else ""}_combined.h5ad')


if __name__ == '__main__':

    args = parse_arguments()
    main(args)