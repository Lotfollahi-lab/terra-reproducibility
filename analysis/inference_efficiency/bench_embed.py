"""
Time embed_dataset of one TERRA source tree on a tokenized dataset and save
its outputs.

The tree is chosen with PYTHONPATH=<tree>/src (setup_variants.sh); with
TERRA_TREE=<tree>/src set as well, the script stops if `terra` was imported
from anywhere else. All variants embed the same tokenized dataset
(prepare_data.py), so compare.py can compare their outputs bit for bit.

usage: PYTHONPATH=<tree>/src python bench_embed.py --bundle DIR --dataset DIR
           --out PREFIX [--batch-size 128] [--num-workers 12] [--pin-memory]
           [--reps 2] [--warmup-cells 1024] [--allow-cpu]

Writes PREFIX.npz (cell_emb, neighborhood_emb, spatial_cell_emb of the last
repetition) and PREFIX.json (per repetition: the embedding-loop time that
embed_dataset logs, the wall time including model loading, and the peak GPU
memory).
"""
import argparse
import json
import os
import resource
import time

import datasets
import numpy as np
import torch
from datasets import load_from_disk

import terra
import terra.inference.embed as embed_module
from terra import embed_dataset


def check_terra_tree():
    """Stop if `terra` is not the tree named by TERRA_TREE (e.g. if the
    venv's own TERRA came first on sys.path)."""
    expected = os.environ.get('TERRA_TREE')
    found = os.path.realpath(os.path.dirname(terra.__file__))
    if expected and os.path.commonpath(
            [found, os.path.realpath(expected)]) != os.path.realpath(expected):
        raise SystemExit(f'terra was imported from {found}, not from '
                         f'TERRA_TREE={expected}')
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--reps', type=int, default=2)
    parser.add_argument('--warmup-cells', type=int, default=1024)
    parser.add_argument('--allow-cpu', action='store_true')
    args = parser.parse_args()
    terra_path = check_terra_tree()
    cuda = torch.cuda.is_available()
    if not cuda and not args.allow_cpu:
        raise SystemExit('No CUDA device: this benchmark is meant for a GPU '
                         'node (pass --allow-cpu to run it anyway).')

    # The loop time that embed_dataset measures and logs (no model loading)
    loop_times = []
    log_inference_time = embed_module.log_inference_time

    def record(elapsed_s, *a, **k):
        loop_times.append(elapsed_s)
        return log_inference_time(elapsed_s, *a, **k)
    embed_module.log_inference_time = record

    dataset = load_from_disk(args.dataset)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=args.pin_memory)

    # Warm-up: CUDA context, kernel selection, allocator
    if args.warmup_cells:
        embed_dataset(dataset.select(range(min(args.warmup_cells,
                                               dataset.num_rows))),
                      args.bundle, **kwargs)
    reps, outputs = [], None
    for rep in range(args.reps):
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        outputs = embed_dataset(dataset, args.bundle, **kwargs)
        if cuda:
            torch.cuda.synchronize()
        wall = time.perf_counter() - start
        reps.append({
            'loop_s': loop_times[-1],
            'wall_s': wall,
            'peak_gpu_mem_gb': (torch.cuda.max_memory_allocated() / 1e9
                                if cuda else None)})
        print(f'rep {rep}: loop {reps[-1]["loop_s"]:.1f} s '
              f'({1000 * reps[-1]["loop_s"] / dataset.num_rows:.2f} s per '
              f'1000 cells), wall {wall:.1f} s, peak GPU memory '
              f'{reps[-1]["peak_gpu_mem_gb"]} GB', flush=True)

    info = {
        'terra': terra_path,
        'n_cells': dataset.num_rows,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'reps': reps,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'datasets': datasets.__version__,
        'gpu': torch.cuda.get_device_name(0) if cuda else None,
        # CPU threads of the host-side work inside the timed loop
        'torch_threads': torch.get_num_threads(),
        'cpus_available': (len(os.sched_getaffinity(0))
                           if hasattr(os, 'sched_getaffinity')
                           else os.cpu_count()),
        # ru_maxrss is in kB on Linux
        'peak_host_rss_gb': resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1e6,
    }
    # Written under temporary names and renamed, the JSON last: a variant
    # counts as done only once PREFIX.json exists
    np.savez(args.out + '.partial.npz',
             **{key: np.asarray(value) for key, value in outputs.items()})
    os.replace(args.out + '.partial.npz', args.out + '.npz')
    with open(args.out + '.partial.json', 'w') as f:
        json.dump(info, f, indent=2)
    os.replace(args.out + '.partial.json', args.out + '.json')


if __name__ == '__main__':
    main()
