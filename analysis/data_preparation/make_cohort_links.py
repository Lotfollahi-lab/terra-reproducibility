#!/usr/bin/env python
"""
Make the cohort folder of a silver root (e.g. silver/hst_corpus_110m, which
evaluation reads as raw_data_folder_path) a folder of links to the dataset
folders of that root, one per dataset of the corpus list, instead of copies.
The links are relative (../<dataset>), so they never go stale and take no
space. TERRA follows them when it reads metadata (collect_adata_from_folder
with followlinks, TERRA feat/compact-tokenized-data 2026-11-21 or later):
jobs running older TERRA code would find no files in a folder of links.

Without --run it only prints what it would do. With --run:
- a cohort folder holding copies (anything that is not a link) is first
  renamed to <silver>/.<cohort>_copies_<date> (same file system: instant;
  move it back to undo). Delete that folder yourself once evaluation works;
- a link is made for every dataset of the corpus list whose folder exists
  in the silver root; links of datasets no longer in the list are removed;
- every link is checked to lead to a folder with .h5ad files.
Datasets of the list without a silver folder are reported and fail the run.

Example:
    python make_cohort_links.py --silver_dir /lustre/scratch126/cellgen/lotfollahi/DATASETS/silver
    python make_cohort_links.py --silver_dir /lustre/scratch126/cellgen/lotfollahi/DATASETS/silver --run
"""
import argparse
import datetime
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--silver_dir', required=True,
                   help='Silver root holding the dataset folders.')
    p.add_argument('--cohort_name', default='hst_corpus_110m')
    p.add_argument('--dataset_names_file_path', default=str(
        HERE / 'hst_corpus_110m' / 'hst_corpus_110m_dataset_names.txt'))
    p.add_argument('--run', action='store_true')
    args = p.parse_args()

    silver = Path(args.silver_dir)
    cohort = silver / args.cohort_name
    with open(args.dataset_names_file_path) as fh:
        names = [l.strip() for l in fh if l.strip()]

    missing = [n for n in names if not (silver / n).is_dir()
               or (silver / n).is_symlink()]
    wanted = [n for n in names if n not in missing]
    print(f'{cohort}: {len(names)} corpus datasets, {len(wanted)} with a '
          f'silver folder, {len(missing)} without')
    for n in missing:
        print(f'  MISSING silver folder (no link): {silver / n}')

    copies = []
    if cohort.exists() and not cohort.is_symlink():
        copies = [e.name for e in cohort.iterdir() if not e.is_symlink()]
    aside = silver / (f'.{args.cohort_name}_copies_'
                      + datetime.date.today().strftime('%Y%m%d'))
    if copies:
        print(f'The cohort folder holds {len(copies)} entries that are not '
              f'links (copies): it is renamed to {aside}')
        if aside.exists():
            sys.exit(f'{aside} exists already; move or delete it first.')

    if not args.run:
        print(f'Would link {len(wanted)} datasets as '
              f'{cohort}/<dataset> -> ../<dataset>.')
        print('Dry run; add --run to make the links.')
        return 1 if missing else 0

    if copies:
        os.rename(cohort, aside)
        print(f'renamed the copies to {aside}')
    cohort.mkdir(exist_ok=True)
    made = kept = removed = 0
    for e in cohort.iterdir():
        if e.is_symlink() and e.name not in wanted:
            e.unlink()
            removed += 1
    for n in wanted:
        link, target = cohort / n, Path('..') / n
        if link.is_symlink():
            if Path(os.readlink(link)) == target:
                kept += 1
                continue
            link.unlink()
        elif link.exists():
            sys.exit(f'{link} exists and is not a link; resolve it by hand.')
        os.symlink(target, link)
        made += 1
    bad = [n for n in wanted
           if not any((cohort / n).glob('*.h5ad'))]
    print(f'{made} links made, {kept} kept, {removed} removed '
          f'(datasets no longer in the list)')
    for n in bad:
        print(f'  FAIL {cohort / n} leads to no .h5ad file')
    if copies:
        print(f'Once evaluation works, delete the old copies: rm -rf {aside}'
              f'\nTo undo instead: rm -rf {cohort} && mv {aside} {cohort}')
    return 1 if (missing or bad) else 0


if __name__ == '__main__':
    sys.exit(main())
