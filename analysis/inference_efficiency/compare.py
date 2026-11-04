"""
Compare the benchmark variants with the base runs and write a report.

usage: python compare.py RESULTS_DIR

Reads, from one run of run_efficiency.sh, RESULTS_DIR/emb__<model>__<variant>
.{npz,json} (bench_embed.py), RESULTS_DIR/tok__<model>__np<nproc>__<variant>/
timing.json (bench_tokenize.py), prepare__<model>.json (prepare_data.py) and
steps.tsv; writes RESULTS_DIR/report.md and report.json and prints the report.

Embeddings: for each output, whether it is bit-identical to the base run,
otherwise the largest absolute difference, the largest difference relative to
the largest base value, and the smallest per-cell cosine similarity; speed is
the embedding-loop time per 1000 cells (mean over repetitions; the loop
includes the DataLoader and the host-side aggregation), and memory the peak
GPU memory. 'base_rerun' repeats the base run: if it is not bit-identical to
'base', the GPU itself is not deterministic and only the size of the
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

# Report order of the variants; others follow alphabetically
ORDER = ['base', 'base_rerun', 'p2', 'p1', 'p12', 'base_pin', 'p12_pin',
         'base_nw0', 'tok', 'tok_gen', 'tok_gen_fp', 'final']


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
            **{key: info.get(key) for key in (
                'gpu', 'n_cells', 'batch_size', 'num_workers', 'pin_memory',
                'terra', 'torch', 'cuda', 'datasets', 'torch_threads',
                'cpus_available')}}


def input_line(results, model):
    path = os.path.join(results, f'prepare__{model}.json')
    if not os.path.isfile(path):
        return 'Input: unknown (no prepare.json).'
    with open(path) as f:
        prepared = json.load(f)
    inputs = prepared['inputs']
    section = (f", {inputs['sample_key']} = {inputs['sample']}"
               if inputs.get('sample_key') else '')
    return (f"Input: {inputs['h5ad']}{section}; {prepared['n_cells']} cells "
            f"(N_CELLS {inputs['n_cells']}), {prepared['n_genes_harmonized']} "
            'harmonized genes.')


def embedding_section(results, report, lines):
    models = sorted({os.path.basename(path).split('__')[1] for path in
                     glob.glob(os.path.join(results, 'emb__*'))})
    for model in models:
        done = {os.path.basename(path)[:-5].split('__', 2)[2]: path[:-5]
                for path in glob.glob(os.path.join(results,
                                                   f'emb__{model}__*.json'))
                if not path.endswith('.partial.json')}
        lines += [f'## Embedding: {model}', '', input_line(results, model)]
        report['embedding'][model] = {}
        if 'base' not in done:
            lines += ['', 'No complete base run: nothing to compare.', '']
            continue
        base = dict(np.load(done['base'] + '.npz'))
        base_time = timing(done['base'] + '.json')
        lines += [f'GPU {base_time["gpu"]}, torch {base_time["torch"]} (CUDA '
                  f'{base_time["cuda"]}), datasets {base_time["datasets"]}; '
                  f'batch size {base_time["batch_size"]}; '
                  f'{base_time["torch_threads"]} torch CPU threads of '
                  f'{base_time["cpus_available"]} CPUs. Speed: embedding loop '
                  'without model loading, mean over the repetitions (each '
                  'repetition in brackets).', '',
                  '| variant | s / 1000 cells | speedup | peak GPU GB | '
                  'outputs vs base |', '|---|---|---|---|---|']
        for variant in sorted(done, key=by_order):
            t = timing(done[variant] + '.json')
            diffs = ({} if variant == 'base' else
                     output_differences(base, dict(np.load(done[variant]
                                                           + '.npz'))))
            report['embedding'][model][variant] = {'timing': t,
                                                   'outputs': diffs}
            verdict = '(reference)' if variant == 'base' else describe(diffs)
            speedup = base_time['s_per_1000_cells'] / t['s_per_1000_cells']
            peak = (f'{t["peak_gpu_mem_gb"]:.1f}'
                    if t['peak_gpu_mem_gb'] is not None else '-')
            per_rep = ', '.join(f'{1000 * s / t["n_cells"]:.2f}'
                                for s in t['loop_s'])
            lines.append(f'| {variant} | {t["s_per_1000_cells"]:.2f} '
                         f'({per_rep}) | {speedup:.2f}x | {peak} | '
                         f'{verdict} |')
        rerun = report['embedding'][model].get('base_rerun')
        if rerun and not all(d.get('bit_identical')
                             for d in rerun['outputs'].values()):
            lines += ['', 'base_rerun is not bit-identical to base: the GPU '
                      'runs are not deterministic, so judge the variants by '
                      'the size of their differences relative to base_rerun.']
        lines.append('')


def tokenization_section(results, report, lines):
    groups = sorted({tuple(os.path.basename(path).split('__')[1:3]) for path
                     in glob.glob(os.path.join(results, 'tok__*__np*__*'))})
    for model, nproc in groups:
        runs = {os.path.basename(path).split('__')[3]: path for path in
                glob.glob(os.path.join(results, f'tok__{model}__{nproc}__*'))}
        infos = {}
        for variant, run_dir in runs.items():
            path = os.path.join(run_dir, 'timing.json')
            if os.path.isfile(path):
                with open(path) as f:
                    infos[variant] = json.load(f)
        lines += [f'## Tokenization: {model}, nproc {nproc[2:]}', '']
        report['tokenization'].setdefault(model, {})[nproc] = infos
        if 'base' not in infos:
            lines += ['No complete base run: nothing to compare.', '']
            continue
        base_seconds = float(np.mean(infos['base']['seconds']))
        lines += [f'{infos["base"]["n_cells"]} cells; time per tokenize_adata '
                  'call, mean over the repetitions.', '',
                  '| variant | s per run | speedup | dataset vs base |',
                  '|---|---|---|---|']
        for variant in sorted(runs, key=by_order):
            if variant not in infos:
                lines.append(f'| {variant} | - | - | no result (failed) |')
                continue
            seconds = float(np.mean(infos[variant]['seconds']))
            same = infos[variant].get('vs_base')
            verdict = ('(reference)' if variant == 'base' else
                       'not compared' if not same else
                       'identical' if same.get('identical') else
                       f'DIFFERENT: {json.dumps(same)}')
            lines.append(f'| {variant} | {seconds:.1f} | '
                         f'{base_seconds / seconds:.2f}x | {verdict} |')
        lines.append('')


def failed_steps(results, report, lines):
    path = os.path.join(results, 'steps.tsv')
    if not os.path.isfile(path):
        return
    with open(path) as f:
        failed = [line.rstrip('\n').split('\t') for line in f
                  if '\tFAILED' in line]
    report['failed_steps'] = failed
    if failed:
        lines += ['## Failed steps', '', 'See the job log for the errors.',
                  ''] + [f'- {name}: {status}' for name, status in failed] + ['']


def main():
    results = sys.argv[1]
    report, lines = {'embedding': {}, 'tokenization': {}}, []
    embedding_section(results, report, lines)
    tokenization_section(results, report, lines)
    failed_steps(results, report, lines)
    text = '\n'.join(lines)
    with open(os.path.join(results, 'report.md'), 'w') as f:
        f.write(text + '\n')
    with open(os.path.join(results, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(text)


if __name__ == '__main__':
    main()
