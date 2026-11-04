"""
Compare the benchmark variants with the base runs and write a report.

usage: python compare.py RESULTS_DIR

Reads RESULTS_DIR/emb__<model>__<variant>.{npz,json} (bench_embed.py) and
RESULTS_DIR/tok__<model>__np<nproc>__<variant>/ (bench_tokenize.py); writes
RESULTS_DIR/report.md and report.json and prints the report.

Embeddings: for each output, whether it is bit-identical to the base run,
otherwise the largest absolute difference, the largest difference relative to
the largest base value, and the smallest per-cell cosine similarity; speed is
the embedding-loop time per 1000 cells (mean over repetitions), and memory the
peak GPU memory. 'base_rerun' repeats the base run: if it is not bit-identical
to 'base', the GPU itself is not deterministic and only the size of the
differences can be judged.
Tokenization: whether the dataset of each prototype variant equals the base
dataset of the same nproc (rows, columns, features, every Arrow column) and
the time per run.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_tokenize import compare_datasets  # noqa: E402

# Report order of the variants; others follow alphabetically
ORDER = ['base', 'base_rerun', 'p2', 'p1', 'p12', 'base_pin', 'p12_pin',
         'base_nw0', 'tok', 'tok_gen', 'tok_gen_fp']


def by_order(variant):
    return (ORDER.index(variant) if variant in ORDER else len(ORDER), variant)


def output_differences(base, other):
    rows = {}
    for key in sorted(base):
        a = base[key].astype(np.float64)
        if key not in other or other[key].shape != base[key].shape:
            rows[key] = {'comparable': False}
            continue
        b = other[key].astype(np.float64)
        row = {'comparable': True,
               'bit_identical': base[key].tobytes() == other[key].tobytes(),
               'nan_base': int(np.isnan(a).any(axis=-1).sum()),
               'nan_variant': int(np.isnan(b).any(axis=-1).sum())}
        both = np.isfinite(a) & np.isfinite(b)
        diff = np.abs(a - b)[both]
        row['max_abs_diff'] = float(diff.max()) if diff.size else 0.0
        scale = float(np.abs(a[both]).max()) if diff.size else 0.0
        row['max_rel_diff'] = row['max_abs_diff'] / scale if scale else 0.0
        if a.ndim == 2:
            ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
            norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            ok &= norms > 0
            cos = (a[ok] * b[ok]).sum(1) / norms[ok]
            row['min_cosine'] = float(cos.min()) if cos.size else None
        rows[key] = row
    return rows


def describe(diffs):
    if all(d.get('bit_identical') for d in diffs.values()):
        return 'bit-identical'
    parts = []
    for key, d in diffs.items():
        if not d.get('comparable'):
            parts.append(f'{key}: not comparable')
        elif d['bit_identical']:
            parts.append(f'{key}: bit-identical')
        else:
            cosine = (f'{d["min_cosine"]:.7f}'
                      if d.get('min_cosine') is not None else '-')
            parts.append(f'{key}: max abs diff {d["max_abs_diff"]:.2e} '
                         f'(rel {d["max_rel_diff"]:.1e}), min cosine '
                         f'{cosine}, cells with NaN {d["nan_base"]} -> '
                         f'{d["nan_variant"]}')
    return '; '.join(parts)


def timing(path):
    with open(path) as f:
        info = json.load(f)
    loops = [rep['loop_s'] for rep in info['reps']]
    peaks = [rep['peak_gpu_mem_gb'] for rep in info['reps']
             if rep['peak_gpu_mem_gb'] is not None]
    return {'s_per_1000_cells': 1000 * float(np.mean(loops)) / info['n_cells'],
            'loop_s': loops,
            'peak_gpu_mem_gb': max(peaks) if peaks else None,
            'gpu': info.get('gpu'), 'n_cells': info['n_cells'],
            'batch_size': info['batch_size'],
            'num_workers': info['num_workers'],
            'pin_memory': info['pin_memory'], 'terra': info['terra']}


def embedding_section(results, report, lines):
    for npz in sorted(glob.glob(os.path.join(results, 'emb__*__base.npz'))):
        model = os.path.basename(npz).split('__')[1]
        base = dict(np.load(npz))
        base_time = timing(npz[:-4] + '.json')
        lines += [f'## Embedding: {model}', '',
                  f'{base_time["n_cells"]} cells, batch size '
                  f'{base_time["batch_size"]}, GPU {base_time["gpu"]}. Speed '
                  'is the embedding loop (no model loading), mean over the '
                  'repetitions.', '',
                  '| variant | s / 1000 cells | speedup | peak GPU GB | '
                  'outputs vs base |', '|---|---|---|---|---|']
        report['embedding'][model] = {}
        variants = {
            os.path.basename(path)[:-4].split('__', 2)[2]: path
            for path in glob.glob(os.path.join(results,
                                               f'emb__{model}__*.npz'))}
        for variant in sorted(variants, key=by_order):
            path = variants[variant]
            t = timing(path[:-4] + '.json')
            diffs = ({} if variant == 'base'
                     else output_differences(base, dict(np.load(path))))
            report['embedding'][model][variant] = {'timing': t,
                                                   'outputs': diffs}
            verdict = '(reference)' if variant == 'base' else describe(diffs)
            speedup = base_time['s_per_1000_cells'] / t['s_per_1000_cells']
            peak = (f'{t["peak_gpu_mem_gb"]:.1f}'
                    if t['peak_gpu_mem_gb'] is not None else '-')
            lines.append(f'| {variant} | {t["s_per_1000_cells"]:.2f} | '
                         f'{speedup:.2f}x | {peak} | {verdict} |')
        rerun = report['embedding'][model].get('base_rerun')
        if rerun and not all(d.get('bit_identical')
                             for d in rerun['outputs'].values()):
            lines += ['', 'base_rerun is not bit-identical to base: the GPU '
                      'runs are not deterministic, so judge the variants by '
                      'the size of their differences relative to base_rerun.']
        lines.append('')


def tokenization_section(results, report, lines):
    for base_dir in sorted(glob.glob(os.path.join(results,
                                                  'tok__*__np*__base'))):
        _, model, nproc, _ = os.path.basename(base_dir).split('__')
        with open(os.path.join(base_dir, 'timing.json')) as f:
            base_info = json.load(f)
        base_seconds = float(np.mean(base_info['seconds']))
        lines += [f'## Tokenization: {model}, nproc {nproc[2:]}', '',
                  f'{base_info["n_cells"]} cells; time per tokenize_adata '
                  'call, mean over the repetitions.', '',
                  '| variant | s per run | speedup | dataset vs base |',
                  '|---|---|---|---|']
        report['tokenization'].setdefault(model, {})[nproc] = {}
        variants = {
            os.path.basename(path).split('__')[3]: path
            for path in glob.glob(os.path.join(results,
                                               f'tok__{model}__{nproc}__*'))}
        for variant in sorted(variants, key=by_order):
            run_dir = variants[variant]
            with open(os.path.join(run_dir, 'timing.json')) as f:
                seconds = float(np.mean(json.load(f)['seconds']))
            same = (None if variant == 'base' else compare_datasets(
                os.path.join(base_dir, 'dataset'),
                os.path.join(run_dir, 'dataset')))
            report['tokenization'][model][nproc][variant] = {
                'seconds': seconds, 'vs_base': same}
            verdict = ('(reference)' if same is None else
                       'identical' if same['identical'] else
                       f'DIFFERENT: {json.dumps(same)}')
            lines.append(f'| {variant} | {seconds:.1f} | '
                         f'{base_seconds / seconds:.2f}x | {verdict} |')
        lines.append('')


def main():
    results = sys.argv[1]
    report, lines = {'embedding': {}, 'tokenization': {}}, []
    embedding_section(results, report, lines)
    tokenization_section(results, report, lines)
    text = '\n'.join(lines)
    with open(os.path.join(results, 'report.md'), 'w') as f:
        f.write(text + '\n')
    with open(os.path.join(results, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(text)


if __name__ == '__main__':
    main()
