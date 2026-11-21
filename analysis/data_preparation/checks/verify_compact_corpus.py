"""Prove that a compact dataset gives the same model inputs as the padded one.

usage: verify_compact_corpus.py DATA_DIR PADDED_NAME COMPACT_NAME STORED_SEQ STORED_NEIGHS [N_SUB [TOKEN_DICT]]

DATA_DIR holds <PADDED_NAME>.dataset and <COMPACT_NAME>.dataset (e.g. two
tokenizations of one pilot dataset on the farm: storage_format padded and
compact, each written by create_tokenized_dataset.py into its own data
directory and linked into DATA_DIR under two names); STORED_SEQ is the
stored genes per cell (512) and STORED_NEIGHS the stored neighbours (16);
N_SUB (default 300) rows are compared for the costlier settings; TOKEN_DICT
defaults to DATA_DIR/td.pkl (only its size is used). Needs the TERRA with
terra.datasets.compact. Ends with 'ALL CHECKS PASSED' (exit code 0); any
difference raises (exit code 1).

Size: every row is compared for one setting and several layouts, and the
padded gene tokens are loaded whole (rows x model_input_size x 8 bytes, a
few times), so use a small pilot dataset (up to ~50k cells: ~4 GB per copy
at 8704 tokens).

Every layout reachable from the stored one with the configs' slicing
(STORED_SEQ x STORED_NEIGHS, 256 genes, 10 and 6 neighbours) is compared:
  1. CellGraphDataset.__getitem__ outputs (every key, dtype, shape, bits) for
     a grid of model settings (gt_type, cell_pos_enc, special tokens, nz_spc,
     pad_special_tokens, sep_gene_tokens_neb, sampling strategies);
  2. lengths: the compact dataset's derived lengths (no list) equal the
     per-layout list of the padded dataset (count of non-zero tokens in the
     layout's columns, as save_layout_n_nonzero_tokens.py computes it) and,
     for the stored layout, n_nonzero_tokens;
  3. CustomDistributedLengthGroupedSampler order: padded + per-layout list vs
     compact without a list; BlockMaskCollator batches under the same seeds;
  4. a DataLoader with 2 spawn workers, distributed length-grouped sampler;
  5. a precomputed split (save_precomputed_split / prepare_dataset, indices
     mapping): items and lengths;
  6. inference: embed._truncation_settings and _init_inference_loader batches
     (embed_dataset / gene_embed_dataset / token distances) and the loader
     infer() builds, for models trained on each layout;
  7. to_padded(compact) == padded (torch rows; raw float64 columns).
Datasets are loaded with terra.datasets.prepare_dataset (train_mode=True for
training, False for inference), as train() and test_model_checkpoint do.
"""
import itertools
import json
import pickle
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

from terra.datasets import prepare_dataset, to_padded
from terra.datasets.cell_datasets import init_cell_dataset
from terra.datasets.dataloaders import (CustomDistributedLengthGroupedSampler,
                                        init_dataloader_and_sampler)
from terra.datasets.utils import save_precomputed_split
from terra.inference.embed import _init_inference_loader, _truncation_settings
from terra.masks.block_masking import BlockMaskCollator

import argparse

_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('data_dir', help='Folder holding both .dataset folders.')
_ap.add_argument('padded_name', help='Padded dataset name (without .dataset).')
_ap.add_argument('compact_name', help='Compact dataset name (without .dataset).')
_ap.add_argument('stored_seq', type=int, help='Stored genes per cell (512).')
_ap.add_argument('stored_neighs', type=int, help='Stored neighbours (16).')
_ap.add_argument('n_sub', type=int, nargs='?', default=300,
                 help='Rows compared for the costlier settings.')
_ap.add_argument('token_dict', nargs='?', default=None,
                 help='Token dictionary (default DATA_DIR/td.pkl).')
_args = _ap.parse_args()
data = Path(_args.data_dir)
P, C = _args.padded_name, _args.compact_name
STORED_SEQ, STORED_NEIGHS = _args.stored_seq, _args.stored_neighs
N_SUB = _args.n_sub
TOKEN_DICT = Path(_args.token_dict) if _args.token_dict else data / 'td.pkl'
with open(TOKEN_DICT, 'rb') as f:
    VOCAB = len(pickle.load(f))
report = {'pair': [P, C], 'checks': {}}
T0 = time.time()


def bits_equal(a, b, path='') -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), (path, list(a), list(b))
        for k in a:
            bits_equal(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            bits_equal(x, y, f'{path}[{i}]')
    elif isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), path
        assert a.dtype == b.dtype and a.shape == b.shape, (path, a.dtype, b.dtype, a.shape, b.shape)
        if a.is_floating_point():
            itype = {torch.float32: torch.int32, torch.float64: torch.int64,
                     torch.float16: torch.int16, torch.bfloat16: torch.int16}[a.dtype]
            assert torch.equal(a.contiguous().view(itype), b.contiguous().view(itype)), path
        else:
            assert torch.equal(a, b), path
    else:
        assert type(a) is type(b) and a == b, (path, a, b)


def load(name, cell_pos_enc, train_mode, split=None):
    args = {'data': {'tokenized_data_folder_path': str(data / f'{name}.dataset')},
            'meta': {'cell_pos_enc': cell_pos_enc}}
    if split is not None:
        args['data']['precomputed_split'] = split
    ds, _, _ = prepare_dataset(args, train_mode=train_mode)
    return ds


# layouts reachable from the stored one: (seq_len_cell, n_neighs, truncate, tokenized_seq_len_cell)
layouts = []
for seq in sorted({STORED_SEQ, 256}, reverse=True):
    for nn in sorted({STORED_NEIGHS, 10, 6}, reverse=True):
        if seq > STORED_SEQ or nn > STORED_NEIGHS:
            continue
        layouts.append(dict(seq_len_cell=seq, n_neighs=nn,
                            truncate_neighbors=nn != STORED_NEIGHS,
                            tokenized_seq_len_cell=STORED_SEQ))
report['layouts'] = layouts

settings = []
for gt, pe in itertools.product(['combined', 'counts', 'rank'], ['coord', 'segment']):
    settings.append(dict(gt_type=gt, cell_pos_enc=pe, special_tokens=[], nz_spc=False,
                         pad_special_tokens=False, sep_gene_tokens_neb=False,
                         sampling_strategy=None))
settings += [
    dict(gt_type='combined', cell_pos_enc='coord', special_tokens=['batch', 'assay', 'tissue', 'gene_panel'],
         nz_spc=True, pad_special_tokens=False, sep_gene_tokens_neb=False, sampling_strategy=None),
    dict(gt_type='rank', cell_pos_enc='coord', special_tokens=['batch', 'assay'],
         nz_spc=False, pad_special_tokens=True, sep_gene_tokens_neb=True, sampling_strategy=None),
    dict(gt_type='counts', cell_pos_enc='coord', special_tokens=['assay'],
         nz_spc=False, pad_special_tokens=False, sep_gene_tokens_neb=False,
         sampling_strategy='norm_value_rank_sampling'),
    dict(gt_type='combined', cell_pos_enc='coord', special_tokens=[],
         nz_spc=False, pad_special_tokens=False, sep_gene_tokens_neb=False,
         sampling_strategy='rand_sampling'),
]

cache = {}


def get_ds(name, pe, train_mode):
    key = (name, pe in ('segment', 'none'), train_mode)
    if key not in cache:
        cache[key] = load(name, pe, train_mode)
    return cache[key]


def make(ds, lay, st, include_cell_id=False, n_nonzero_tokens_list=None):
    return init_cell_dataset(
        tokenizer_type='cell_graph', dataset=ds, vocab_size=VOCAB,
        seq_len_cell=lay['seq_len_cell'],
        seq_len_neighborhood=lay['seq_len_cell'] * lay['n_neighs'],
        gt_type=st['gt_type'], cell_pos_enc=st['cell_pos_enc'],
        special_tokens=st['special_tokens'], sampling_strategy=st['sampling_strategy'],
        n_nonzero_tokens_list=n_nonzero_tokens_list, include_cell_id=include_cell_id,
        sep_gene_tokens_neb=st['sep_gene_tokens_neb'], nz_spc=st['nz_spc'],
        pad_special_tokens=st['pad_special_tokens'],
        truncate_neighbors=lay['truncate_neighbors'],
        tokenized_seq_len_cell=lay['tokenized_seq_len_cell'])


def padded_layout_list(lay, rows=None):
    """Per-layout n_nonzero list of the padded dataset (save_layout_n_nonzero_tokens.py)."""
    raw = load_from_disk(str(data / f'{P}.dataset'))
    if rows is not None:
        raw = raw.select(rows)
    tokens = np.asarray(raw.with_format('numpy')['gene_tokens'])
    stride = lay['tokenized_seq_len_cell']
    if lay['seq_len_cell'] == stride and not lay['truncate_neighbors']:
        return [int(x) for x in raw['n_nonzero_tokens']]
    cols = np.concatenate([np.arange(s * stride, s * stride + lay['seq_len_cell'])
                           for s in range(lay['n_neighs'] + 1)])
    return np.count_nonzero(tokens[:, cols], axis=1).tolist()


def coll_for(lay, n_spc=0):
    return BlockMaskCollator(n_targets=1, n_contexts=1, n_segments=lay['n_neighs'] + 1,
                             seq_len_cell=lay['seq_len_cell'],
                             seq_len_neighborhood=lay['seq_len_cell'] * lay['n_neighs'],
                             n_special_tokens=n_spc, per_block_mask_ratio=0.8,
                             sample_segments=True, sample_gene_masks=True)


def main():
    n_rows = len(get_ds(P, 'coord', True))
    assert n_rows == len(get_ds(C, 'coord', True))
    rng = np.random.default_rng(0)
    sub = sorted(rng.choice(n_rows, min(N_SUB, n_rows), replace=False).tolist())

    # 1. __getitem__ for every layout x setting (all rows for the first setting)
    n_items = n_raised = 0
    raised = {}
    for li, lay in enumerate(layouts):
        for si, st in enumerate(settings):
            dp = make(get_ds(P, st['cell_pos_enc'], True), lay, st)
            dc = make(get_ds(C, st['cell_pos_enc'], True), lay, st)
            assert dc.compact and not dp.compact
            rows = range(n_rows) if si == 0 else sub
            for i in rows:
                out = []
                for d in (dp, dc):
                    np.random.seed(i); random.seed(i); torch.manual_seed(i)
                    try:
                        out.append(d[i])
                    except Exception as e:  # noqa: BLE001
                        # a bug of the padded path must be the same bug
                        out.append(('raised', type(e).__name__, str(e)))
                if isinstance(out[0], tuple) or isinstance(out[1], tuple):
                    assert out[0] == out[1], (li, si, i, out[0], out[1])
                    raised.setdefault(f'S{si}', set()).add(out[0][1] + ': ' + out[0][2][:80])
                    n_raised += 1
                    continue
                bits_equal(out[0], out[1], f'L{li}S{si}row{i}')
                n_items += 1
    report['checks']['1_getitem_items_compared'] = n_items
    report['checks']['1_getitem_rows_raising_identically_in_both'] = n_raised
    report['checks']['1_getitem_errors'] = {k: sorted(v) for k, v in raised.items()}
    print('1 done', round(time.time() - T0), flush=True)

    # 2. lengths, 3. sampler order and collator batches (released training setting)
    st = settings[0]
    coll_batches = 0
    lengths_report = {}
    for lay in layouts:
        key = f"{lay['n_neighs']}x{lay['seq_len_cell']}"
        layout_list = padded_layout_list(lay)
        dp = make(get_ds(P, 'coord', True), lay, st, n_nonzero_tokens_list=layout_list)
        dc = make(get_ds(C, 'coord', True), lay, st)
        derived = [int(x) for x in dc.n_nz_tokens]
        assert derived == layout_list, key
        stored_nnz = [int(x) for x in make(get_ds(P, 'coord', True), lay, st).n_nz_tokens]
        lengths_report[key] = dict(
            mean=float(np.mean(derived)),
            equal_to_stored_n_nonzero=derived == stored_nnz)
        if lay['seq_len_cell'] == STORED_SEQ and not lay['truncate_neighbors']:
            assert derived == stored_nnz
        sp = CustomDistributedLengthGroupedSampler(dp, batch_size=32, num_replicas=2, rank=1, seed=3)
        sc = CustomDistributedLengthGroupedSampler(dc, batch_size=32, num_replicas=2, rank=1, seed=3)
        for ep in range(2):
            sp.set_epoch(ep); sc.set_epoch(ep)
            assert list(sp) == list(sc), key
        coll = coll_for(lay)
        sp.set_epoch(0)
        order = list(sp)
        for b in range(0, min(len(order), 640), 32):
            idx = order[b:b + 32]
            torch.manual_seed(b); np.random.seed(b)
            ca = coll([dp[i] for i in idx])
            torch.manual_seed(b); np.random.seed(b)
            cb = coll([dc[i] for i in idx])
            bits_equal(ca, cb, f'collate{b}')
            coll_batches += 1
    report['checks']['2_derived_lengths_equal_padded_layout_lists'] = lengths_report
    report['checks']['3_sampler_orders_equal'] = True
    report['checks']['3_collator_batches_compared'] = coll_batches
    print('2-3 done', round(time.time() - T0), flush=True)

    # 4. DataLoader with 2 spawn workers, distributed length-grouped sampler (train path)
    dl_batches = 0
    for lay in (layouts[0], layouts[-1]):
        loaders = []
        for name in (P, C):
            nz = padded_layout_list(lay) if name == P else None
            ds = make(get_ds(name, 'coord', True), lay, st, n_nonzero_tokens_list=nz)
            g = torch.Generator(); g.manual_seed(7)
            loader, sampler = init_dataloader_and_sampler(
                cell_dataset=ds, batch_size=32, distributed=True, world_size=1, rank=0,
                mega_batch_mult_max=1000, collate_fn=coll_for(lay), num_workers=2,
                drop_last=True, generator=g, multiprocessing_context='spawn',
                persistent_workers=False)
            sampler.set_epoch(1)
            loaders.append(loader)
        for ba, bb in zip(*loaders):
            bits_equal(ba, bb, f'loader{dl_batches}')
            dl_batches += 1
    report['checks']['4_dataloader_batches_compared_2_workers'] = dl_batches
    print('4 done', round(time.time() - T0), flush=True)

    # 5. precomputed split (indices mapping), as train() loads it
    tmp = Path(tempfile.mkdtemp(dir=data))
    split_rows = sorted(rng.choice(n_rows, n_rows * 3 // 4, replace=False).tolist())
    raw_p = load_from_disk(str(data / f'{P}.dataset'))
    raw_c = load_from_disk(str(data / f'{C}.dataset'))
    save_precomputed_split(raw_p, split_rows, str(tmp / 'split_p.pkl'))
    save_precomputed_split(raw_c, split_rows, str(tmp / 'split_c.pkl'))
    split_items = 0
    for lay in layouts:
        sp_ds = load(P, 'coord', True, split=str(tmp / 'split_p.pkl'))
        sc_ds = load(C, 'coord', True, split=str(tmp / 'split_c.pkl'))
        layout_list = padded_layout_list(lay, rows=split_rows)
        dp = make(sp_ds, lay, st, n_nonzero_tokens_list=layout_list)
        dc = make(sc_ds, lay, st)
        assert [int(x) for x in dc.n_nz_tokens] == layout_list
        for i in range(0, len(dp), 7):
            bits_equal(dp[i], dc[i], f'split{i}')
            split_items += 1
    report['checks']['5_precomputed_split_items_compared'] = split_items
    print('5 done', round(time.time() - T0), flush=True)

    # 6. inference on the pre-tokenized corpus (prepare_dataset train_mode=False)
    inf_batches = 0
    for lay in layouts:
        for st_i in (settings[0], settings[6], settings[1]):
            n_seg = lay['n_neighs'] + 1
            cfg = {'meta': {'special_tokens': st_i['special_tokens'], 'gt_type': st_i['gt_type'],
                            'cell_pos_enc': st_i['cell_pos_enc']},
                   'data': {'n_segments': n_seg, 'seq_len_cell': lay['seq_len_cell'],
                            'seq_len_neighborhood': lay['seq_len_cell'] * lay['n_neighs'],
                            'tokenizer_type': 'cell_graph', 'sep_gene_tokens_neb': False,
                            'truncate_neighbors': lay['truncate_neighbors'],
                            'tokenized_seq_len_cell': lay['tokenized_seq_len_cell'],
                            # model config of a model trained on this corpus with this layout
                            'model_input_size': lay['seq_len_cell'] * n_seg,
                            'n_neighs': lay['n_neighs']},
                   'mask': {'n_targets': 1, 'n_contexts': 1, 'per_block_mask_ratio': 0.8}}
            dsp = get_ds(P, st_i['cell_pos_enc'], False)
            dsc = get_ds(C, st_i['cell_pos_enc'], False)
            assert _truncation_settings(dsp, cfg['data']) == _truncation_settings(dsc, cfg['data'])
            for pad_spc in (True, False):
                la = _init_inference_loader(dsp, cfg, VOCAB, pad_special_tokens=pad_spc,
                                            batch_size=64, pin_memory=False, num_workers=0)
                lb = _init_inference_loader(dsc, cfg, VOCAB, pad_special_tokens=pad_spc,
                                            batch_size=64, pin_memory=False, num_workers=0)
                for ba, bb in zip(la, lb):
                    bits_equal(ba, bb, 'inference')
                    inf_batches += 1
            # the loader infer() builds (infer.py:378-406)
            tr, tl = _truncation_settings(dsp, cfg['data'])
            loaders = []
            for ds in (dsp, dsc):
                cd = init_cell_dataset(
                    dataset=ds, vocab_size=VOCAB, seq_len_cell=lay['seq_len_cell'],
                    seq_len_neighborhood=lay['seq_len_cell'] * lay['n_neighs'],
                    tokenizer_type='cell_graph', gt_type=st_i['gt_type'],
                    cell_pos_enc=st_i['cell_pos_enc'], special_tokens=st_i['special_tokens'],
                    sampling_strategy=None, n_nonzero_tokens_list=[], include_cell_id=True,
                    sep_gene_tokens_neb=False, nz_spc=st_i['nz_spc'], pad_special_tokens=True,
                    truncate_neighbors=tr, tokenized_seq_len_cell=tl)
                coll = BlockMaskCollator(
                    n_targets=1, n_contexts=1, n_segments=n_seg,
                    seq_len_cell=lay['seq_len_cell'],
                    seq_len_neighborhood=lay['seq_len_cell'] * lay['n_neighs'],
                    n_special_tokens=len(st_i['special_tokens']), per_block_mask_ratio=0.8,
                    sample_segments=False, sample_gene_masks=False,
                    special_token_pad_ratio=1.0)
                loader, _ = init_dataloader_and_sampler(
                    cell_dataset=cd, batch_size=64, distributed=False, world_size=1, rank=0,
                    collate_fn=coll, pin_memory=False, num_workers=0, drop_last=False,
                    persistent_workers=False, mega_batch_mult_max=1000)
                loaders.append(loader)
            for ba, bb in zip(*loaders):
                bits_equal(ba, bb, 'infer_loader')
                inf_batches += 1
    report['checks']['6_inference_batches_compared'] = inf_batches
    print('6 done', round(time.time() - T0), flush=True)

    # 7. to_padded(compact) == padded
    back = to_padded(raw_c)
    assert sorted(back.column_names) == sorted(raw_p.column_names)
    assert back.features == raw_p.features
    assert np.array_equal(np.asarray(back['gene_tokens']), np.asarray(raw_p['gene_tokens']))
    ge_b, ge_p = np.asarray(back['gene_expr']), np.asarray(raw_p['gene_expr'])
    report['checks']['7_to_padded_gene_expr_float64_identical'] = bool(
        np.array_equal(ge_b.view(np.int64), ge_p.view(np.int64)))
    rx_b, rx_p = np.asarray(back['rel_x_coord']), np.asarray(raw_p['rel_x_coord'])
    report['checks']['7_to_padded_rel_coords_float64_identical'] = bool(np.array_equal(rx_b, rx_p))
    report['checks']['7_to_padded_rel_coords_float32_identical'] = bool(
        np.array_equal(rx_b.astype(np.float32).view(np.int32), rx_p.astype(np.float32).view(np.int32)))
    back.set_format('torch'); raw_p.set_format('torch')
    for i in sub:
        bits_equal(back[i], raw_p[i], f'to_padded{i}')
    report['checks']['7_to_padded_torch_rows_equal'] = len(sub)
    report['seconds'] = round(time.time() - T0)
    print(json.dumps(report))
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
