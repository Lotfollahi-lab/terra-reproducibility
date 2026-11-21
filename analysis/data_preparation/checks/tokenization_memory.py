"""
Estimate the memory of the per-dataset tokenization jobs and compare it with
the resource table of analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh.

The table was sized for earlier builds (10 x 256 genes, whole dataset in one
dict). With the hst_corpus_110m configs (16 x 512 genes, model_input_size 8704,
stream_per_file: true, processing_mode: parallel) the tokenizer holds, per
input file, a dict whose neighbourhood arrays take about
    24 bytes x (n_neighs + 1) x mean min(genes, seq_len_cell)   per cell
(three float64 arrays; measured 208 kB per cell for cells with >= 512 genes at
16 x 512), and a parallel job holds up to about 3 x cores files at once
(TERRA cell_tokenizers.py _iter_file_dicts: 2 x cores finished results waiting,
plus the files the cores are tokenizing). The estimate per dataset is
    base_gb + margin x (dicts of the min(files, 3 x cores) largest files)
Without --gpc_per_file every cell is assumed to fill its segments
(mean min(genes, seq_len_cell) = seq_len_cell: an upper bound); with the
per-file table of count_genes_per_cell.py the files' mean_filled_<seq_len_cell>
column is used. base_gb (the job's other memory: AnnData of the files being
tokenized, Arrow writer, interpreter) and margin are guesses: UNVERIFIED at
farm scale. Compare the estimates with the "Max Memory" of the first finished
jobs before trusting them.

Reads only HDF5 metadata (the shape of X) of the silver files; read-only.

Usage (farm, terra venv, from the terra-reproducibility root):
    python analysis/data_preparation/checks/tokenization_memory.py \
        --names analysis/data_preparation/hst_corpus_110m/hst_corpus_110m_dataset_names.txt \
        --silver ${TERRA_DATA_DIR}/silver [--gpc_per_file out/gpc_*_per_file.tsv]
Prints one row per dataset (files, cells, largest file, the table's cores /
queue / RAM, the estimate at the table's cores, OK or OVER) and, for every OVER
row, a wrapper command with fewer cores or more memory. Exit code 1 if any
dataset is OVER or has no silver file.
"""
import argparse
import csv
import math
import re
import sys
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[3]
WRAPPER = (REPO_ROOT / 'analysis' / 'tokenization'
           / 'wrapper_create_cell_graph_tokenized_dataset.sh')
# Largest memory per queue, from the wrapper's table (long up to 600G,
# hugemem up to 1536G); teramem beyond that. The farm's actual limits are
# unclear: check `bqueues -l <queue>` (MEMLIMIT) before relying on them
QUEUE_RAM_GB = (('long', 600), ('hugemem', 1536), ('teramem', 2900))


def parse_size_gb(text: str) -> float:
    """'512G' / '1536G' / '256GB' / '2T' -> GB."""
    m = re.fullmatch(r'\s*([\d.]+)\s*([GgTtMm])?[Bb]?\s*', text)
    if not m:
        raise ValueError(f'Cannot read the memory size {text!r}.')
    value, unit = float(m.group(1)), (m.group(2) or 'G').upper()
    return value * {'M': 1 / 1024, 'G': 1, 'T': 1024}[unit]


def parse_wrapper_table(path: Path) -> dict:
    """{dataset or '*': {'RAM': .., 'TIME': .., 'CORES': .., 'QUEUE': ..}}
    from the `case $EXPERIMENT_NAME in ... esac` block of the wrapper."""
    table, current = {}, None
    in_case = False
    for line in path.read_text().splitlines():
        stripped = line.split('#', 1)[0].strip()
        if stripped.startswith('case $EXPERIMENT_NAME in'):
            in_case = True
            continue
        if not in_case:
            continue
        if stripped == 'esac':
            break
        m = re.fullmatch(r'([\w.*-]+)\)', stripped)
        if m:
            current = m.group(1)
            table[current] = {}
            continue
        m = re.fullmatch(r'(RAM|TIME|CORES|QUEUE)="?([^"]*)"?', stripped)
        if m and current is not None:
            table[current][m.group(1)] = m.group(2)
        if stripped == ';;':
            current = None
    if '*' not in table:
        raise ValueError(f"No default row ('*)') in the table of {path}.")
    return table


def n_obs(path: Path) -> int:
    """Rows of X of an .h5ad file, from the HDF5 metadata only."""
    with h5py.File(path, 'r') as f:
        X = f['X']
        if isinstance(X, h5py.Dataset):
            return int(X.shape[0])
        shape = X.attrs['shape'] if 'shape' in X.attrs else X.attrs[
            'h5sparse_shape']
        return int(shape[0])


def read_gpc_per_file(paths, seq_len_cell: int) -> dict:
    """{file path: (cells, mean min(genes, seq_len_cell))} from the per-file
    tables of count_genes_per_cell.py."""
    column = f'mean_filled_{seq_len_cell}'
    out = {}
    for p in paths:
        with open(p) as f:
            for row in csv.DictReader(f, delimiter='\t'):
                if row.get('status') != 'OK':
                    continue
                if column not in row:
                    raise ValueError(f"'{p}' has no column {column}.")
                out[str(Path(row['path']))] = (int(row['cells']),
                                               float(row[column]))
    return out


def estimate_gb(file_dicts_gb: list, cores: int, base_gb: float,
                margin: float) -> float:
    """base + margin x the min(files, 3 x cores) largest file dicts."""
    held = sorted(file_dicts_gb, reverse=True)[:3 * max(cores, 1)]
    return base_gb + margin * sum(held)


def suggest(file_dicts_gb, cores, ram_gb, base_gb, margin):
    """Fewest changes that fit: the most cores <= `cores` within `ram_gb`,
    else the RAM needed at 1 core and the queue for it."""
    for c in range(cores, 0, -1):
        if estimate_gb(file_dicts_gb, c, base_gb, margin) <= ram_gb:
            return c, None
    need = estimate_gb(file_dicts_gb, 1, base_gb, margin)
    return 1, need


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--names', required=True,
                    help='Dataset names file (first comma-separated field).')
    ap.add_argument('--silver', required=True,
                    help='Silver root: <silver>/<dataset>/**/*.h5ad.')
    ap.add_argument('--gpc_per_file', nargs='*', default=[],
                    help='Per-file tables of count_genes_per_cell.py '
                         '(<out>_per_file.tsv), for the mean filled segment '
                         'length of each file (default: full segments).')
    ap.add_argument('--wrapper', default=str(WRAPPER),
                    help='The tokenization wrapper with the resource table.')
    ap.add_argument('--n_neighs', type=int, default=16)
    ap.add_argument('--seq_len_cell', type=int, default=512)
    ap.add_argument('--base_gb', type=float, default=16.0,
                    help='Memory besides the file dicts (guess; default 16).')
    ap.add_argument('--margin', type=float, default=1.3,
                    help='Factor on the file dicts (guess; default 1.3).')
    args = ap.parse_args()

    table = parse_wrapper_table(Path(args.wrapper))
    gpc = read_gpc_per_file(args.gpc_per_file, args.seq_len_cell)
    names = [line.split(',')[0].strip() for line in open(args.names)
             if line.strip()]
    bytes_per_entry = 24 * (args.n_neighs + 1)

    header = ['dataset', 'files', 'cells', 'largest_file', 'mean_filled',
              'cores', 'queue', 'ram_gb', 'estimate_gb', 'status']
    print('\t'.join(header))
    commands, n_bad = [], 0
    for name in names:
        row = table.get(name, table['*'])
        cores, queue = int(row['CORES']), row['QUEUE']
        ram_gb = parse_size_gb(row['RAM'])
        files = sorted((Path(args.silver) / name).glob('**/*.h5ad'))
        if not files:
            print('\t'.join([name, '0', '', '', '', str(cores), queue,
                             f'{ram_gb:.0f}', '', 'MISSING']))
            n_bad += 1
            continue
        sizes, fills = [], []
        for path in files:
            cells, filled = gpc.get(str(path), (None, None))
            if cells is None:
                cells, filled = n_obs(path), float(args.seq_len_cell)
            sizes.append(cells)
            fills.append(filled)
        dicts_gb = [c * (bytes_per_entry * m + 1024) / 1e9
                    for c, m in zip(sizes, fills)]
        est = estimate_gb(dicts_gb, cores, args.base_gb, args.margin)
        status = 'OK' if est <= ram_gb else 'OVER'
        mean_filled = (sum(c * m for c, m in zip(sizes, fills))
                       / max(sum(sizes), 1))
        print('\t'.join([name, str(len(files)), str(sum(sizes)),
                         str(max(sizes)), f'{mean_filled:.0f}', str(cores),
                         queue, f'{ram_gb:.0f}', f'{est:.0f}', status]))
        if status == 'OVER':
            n_bad += 1
            new_cores, need_gb = suggest(dicts_gb, cores, ram_gb,
                                         args.base_gb, args.margin)
            cmd = ('bash analysis/tokenization/'
                   f'wrapper_create_cell_graph_tokenized_dataset.sh {name}')
            if need_gb is None:
                # Fewer files held at once
                commands.append(f'{cmd} {new_cores} {queue} {row["RAM"]}')
            else:
                # Even one core exceeds the table: more memory, at 1 core
                q = next((q for q, limit in QUEUE_RAM_GB
                          if need_gb <= limit), None)
                ram = f'{int(math.ceil(need_gb / 64.0) * 64)}G'
                commands.append(
                    f'{cmd} 1 {q or "teramem"} {ram}'
                    + ('' if q else '   # above the largest queue limit '
                       'assumed here: check bqueues -l teramem'))
    if commands:
        print('\n# Datasets whose estimate exceeds the table (fewer cores, '
              'or more memory):', file=sys.stderr)
        for c in commands:
            print(c, file=sys.stderr)
    print(f'\n{len(names)} datasets, {n_bad} OVER or MISSING '
          f"({'per-file genes from count_genes_per_cell.py' if gpc else 'full segments assumed: upper bound'}).",
          file=sys.stderr)
    return 1 if n_bad else 0


if __name__ == '__main__':
    sys.exit(main())
