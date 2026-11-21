#!/usr/bin/env python
"""
Find dataset folders in the NFS DATASETS root that TERRA does not use, and
move them out of the way (reversibly) once an identical copy exists on
Lustre.

A dataset folder is silver/<name>, bronze/<name> or gold/<tokenizer>/<name>.
It is kept (TERRA-relevant) if any of these holds:
- corpus: <name> is in a corpus list of this repository
  (analysis/data_preparation/*/*_dataset_names*.txt), or is a corpus folder
  (hst_corpus_*, mst_corpus_*);
- config: config/bronze_to_silver/<name>.yaml or a cell-graph /
  cell-neighborhood tokenizer config for <name> exists;
- naming: <name> follows TERRA's dataset naming (<letters><id>-..., e.g.
  xhs1000-3b_1p, mmb0-1b_smb1-20b_1p);
- extra: <name> is in --keep (default: the SpatialFusion and CAMEO downloads
  of the rebuttal, which exist only on NFS).
Everything else is a candidate for removal. --remove marks further folders
as candidates even if a rule keeps them. Top-level folders other than
bronze, silver, gold, tokenizer, protein_embeddings and genes, and gold
folders of other tools, are only listed, never moved.

For each candidate the dry run shows its size, whether Lustre holds an
identical copy at the same path (same files, folders, links and total size)
and which files of the TERRA and terra-reproducibility checkouts mention
its name.

--move moves every candidate with an identical Lustre copy (all candidates
with --allow_without_lustre_copy) to <root>/.to_delete_<date>/<same path>,
on the same file system, so it is instant and can be undone with
--restore <that folder>. A manifest of the moves is written there. Nothing
is deleted: once you are sure, delete that folder yourself.

Example:
    python prune_nfs_datasets.py                       # dry run
    python prune_nfs_datasets.py --remove xhs1000-3b_1p --keep yeh_hgsc
    python prune_nfs_datasets.py --move
    python prune_nfs_datasets.py --restore /nfs/team361/sb75/DATASETS/.to_delete_20261121
"""
import argparse
import csv
import datetime
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = '/nfs/team361/sb75/DATASETS'
LUSTRE_ROOT = '/lustre/scratch126/cellgen/lotfollahi/DATASETS'
REPOS = ['/nfs/team361/sb75/terra-reproducibility', '/nfs/team361/sb75/terra']
TERRA_TOP = {'bronze', 'silver', 'gold', 'tokenizer', 'protein_embeddings',
             'genes'}
TERRA_GOLD_TOOLS = {'cell-graph-tokenizer', 'cell-neighborhood-tokenizer'}
DEFAULT_KEEP = {'spatialfusion', 'cameo'}
TERRA_NAME = re.compile(r'^[a-z]+[0-9]+-')
CORPUS_NAME = re.compile(r'^(hst|mst)_corpus_')
# Files that only mention dataset names as examples
SELF = {'prune_nfs_datasets.py', 'copy_missing_silver_folders.py',
        'compare_silver_roots.py'}
TEXT_EXT = ('.py', '.lsf', '.sh', '.yaml', '.yml', '.txt', '.ipynb', '.md',
            '.json', '.cfg', '.toml')


def subdirs(path):
    try:
        return sorted(d for d in os.listdir(path)
                      if os.path.isdir(os.path.join(path, d))
                      and not d.startswith('.'))
    except FileNotFoundError:
        return []


def stats(path):
    """Files, folders, links and total file size below path (links are not
    followed); None if path does not exist."""
    if not os.path.lexists(path):
        return None
    if os.path.islink(path):
        return dict(link_to=os.path.realpath(path))
    n_files = n_dirs = n_links = size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for d in list(dirnames):
            if os.path.islink(os.path.join(dirpath, d)):
                n_links += 1
                dirnames.remove(d)
            else:
                n_dirs += 1
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.islink(fp):
                n_links += 1
            else:
                n_files += 1
                size += os.lstat(fp).st_size
    return dict(files=n_files, dirs=n_dirs, links=n_links, bytes=size)


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(n) < 1000 or unit == 'TB':
            return f'{n:.1f} {unit}' if unit != 'B' else f'{n} B'
        n /= 1000.0


def repo_names(repo):
    """Dataset names in corpus lists and configs of a terra-reproducibility
    checkout."""
    corpus, config = set(), set()
    dp = os.path.join(repo, 'analysis', 'data_preparation')
    for sub in subdirs(dp):
        for f in os.listdir(os.path.join(dp, sub)):
            if re.search(r'_dataset_names.*\.txt$', f):
                with open(os.path.join(dp, sub, f)) as fh:
                    corpus |= {l.strip() for l in fh if l.strip()}
    for sub, prefix in (('bronze_to_silver', ''),
                        ('cell-graph-tokenizer', 'cell-graph-tokenizer_'),
                        ('cell-neighborhood-tokenizer',
                         'cell-neighborhood-tokenizer_')):
        d = os.path.join(repo, 'config', sub)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith('.yaml'):
                    config.add(f[:-5][len(prefix):] if f.startswith(prefix)
                               else f[:-5])
    return corpus, config


def find_mentions(names, repos):
    """{name: [files mentioning it]} over the text files of the repos."""
    if not names:
        return {}
    pat = re.compile(r'(?<![A-Za-z0-9_-])(' + '|'.join(
        re.escape(n) for n in sorted(names, key=len, reverse=True))
        + r')(?![A-Za-z0-9_-])')
    hits = {n: [] for n in names}
    for repo in repos:
        for dirpath, dirnames, filenames in os.walk(repo):
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            for f in filenames:
                if f in SELF or not f.endswith(TEXT_EXT):
                    continue
                fp = os.path.join(dirpath, f)
                try:
                    if os.path.getsize(fp) > 50_000_000:
                        continue
                    with open(fp, errors='ignore') as fh:
                        found = set(pat.findall(fh.read()))
                except OSError:
                    continue
                for n in found:
                    hits[n].append(os.path.relpath(fp, os.path.dirname(repo)))
    return hits


def classify(name, corpus, config, keep, remove):
    if name in remove:
        return None
    if name in corpus or CORPUS_NAME.match(name):
        return 'corpus'
    if name in config:
        return 'config'
    if name in keep:
        return 'extra'
    if TERRA_NAME.match(name):
        return 'naming'
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--root', default=ROOT)
    p.add_argument('--lustre_root', default=LUSTRE_ROOT)
    p.add_argument('--repos', default=','.join(REPOS),
                   help='Comma-separated checkouts; the first one must be '
                        'terra-reproducibility (corpus lists, configs).')
    p.add_argument('--keep', default='',
                   help='Comma-separated folder names to keep as well.')
    p.add_argument('--remove', default='',
                   help='Comma-separated folder names to remove although a '
                        'rule keeps them.')
    p.add_argument('--move', action='store_true',
                   help='Move the candidates to <root>/.to_delete_<date>.')
    p.add_argument('--allow_without_lustre_copy', action='store_true',
                   help='With --move, also move candidates that have no '
                        'identical copy on Lustre.')
    p.add_argument('--restore', default=None,
                   help='Move everything in this .to_delete_* folder back.')
    p.add_argument('--n_workers', type=int, default=8)
    args = p.parse_args()

    if args.restore:
        return restore(args.restore)

    repos = [r for r in args.repos.split(',') if r]
    corpus, config = repo_names(repos[0])
    if not corpus:
        sys.exit(f'No corpus lists found in {repos[0]}; refusing to '
                 'classify without them.')
    keep = DEFAULT_KEEP | {x for x in args.keep.split(',') if x}
    remove = {x for x in args.remove.split(',') if x}

    folders, unknown = [], []
    for top in subdirs(args.root):
        if top not in TERRA_TOP:
            unknown.append(top)
    for tier in ('silver', 'bronze'):
        for name in subdirs(os.path.join(args.root, tier)):
            folders.append((f'{tier}/{name}', name))
    for tool in subdirs(os.path.join(args.root, 'gold')):
        if tool not in TERRA_GOLD_TOOLS:
            unknown.append(f'gold/{tool}')
            continue
        for name in subdirs(os.path.join(args.root, 'gold', tool)):
            folders.append((f'gold/{tool}/{name}', name))

    rows = []
    for rel, name in folders:
        reason = classify(name, corpus, config, keep, remove)
        rows.append(dict(path=rel, name=name, keep=reason))
    cands = [r for r in rows if r['keep'] is None]
    print(f'{len(rows)} dataset folders in {args.root}: '
          f'{len(rows) - len(cands)} kept, {len(cands)} candidates for '
          'removal', flush=True)

    def measure(r):
        r['nfs'] = stats(os.path.join(args.root, r['path']))
        r['lustre'] = stats(os.path.join(args.lustre_root, r['path']))
        return r

    print('measuring candidates and their Lustre copies ...', flush=True)
    with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        cands = list(ex.map(measure, cands))
    mentions = find_mentions({r['name'] for r in cands}, repos)

    print('\nKept (by rule):')
    for reason in ('corpus', 'config', 'naming', 'extra'):
        names = [r['path'] for r in rows if r['keep'] == reason]
        print(f'  {reason}: {len(names)}')
        if reason in ('naming', 'extra'):
            # the rules most likely to keep something by mistake: list them
            for n in names:
                print(f'    {n}')
    print('\nCandidates for removal:')
    movable, total = [], 0
    for r in sorted(cands, key=lambda r: r['path']):
        nfs, lus = r['nfs'], r['lustre']
        size = nfs.get('bytes', 0) if nfs else 0
        if lus is None:
            copy = 'NOT on Lustre'
        elif lus == nfs:
            copy = 'identical copy on Lustre'
        else:
            copy = 'Lustre copy DIFFERS'
        ok = copy.startswith('identical') or args.allow_without_lustre_copy
        if ok:
            movable.append(r)
            total += size
        refs = mentions.get(r['name'], [])
        ref = (f'; mentioned in {len(refs)} file(s): ' + ', '.join(refs[:3])
               + (' ...' if len(refs) > 3 else '')) if refs else ''
        what = (f'link to {nfs["link_to"]}' if nfs and 'link_to' in nfs
                else f'{human(size):>10s} {nfs["files"] if nfs else 0:7d} files')
        print(f'  {r["path"]:62s} {what}  {copy}{ref}')
    if unknown:
        print('\nOther folders (not TERRA tiers or tools; listed only, never '
              'moved):')
        for u in unknown:
            s = stats(os.path.join(args.root, u))
            print(f'  {u:62s} {human(s.get("bytes", 0)):>10s}')
    print(f'\n{len(movable)} of {len(cands)} candidates can be moved '
          f'({human(total)})' + ('' if args.allow_without_lustre_copy else
                                 '; the others have no identical Lustre copy'))
    if not args.move:
        print('Dry run; add --move to move them to '
              f'{args.root}/.to_delete_<date> (reversible).')
        return 0

    trash = os.path.join(
        args.root, '.to_delete_' + datetime.date.today().strftime('%Y%m%d'))
    os.makedirs(trash, exist_ok=True)
    manifest = os.path.join(trash, 'manifest.tsv')
    new = not os.path.exists(manifest)
    with open(manifest, 'a', newline='') as fh:
        w = csv.writer(fh, delimiter='\t')
        if new:
            w.writerow(['path', 'bytes', 'lustre_copy'])
        for r in movable:
            src = os.path.join(args.root, r['path'])
            dst = os.path.join(trash, r['path'])
            if os.path.lexists(dst):
                print(f'skip {r["path"]}: already in {trash}')
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
            w.writerow([r['path'], (r['nfs'] or {}).get('bytes', 0),
                        r['lustre'] == r['nfs']])
            fh.flush()
            print(f'moved {r["path"]}')
    print(f'\nMoved {len(movable)} folders to {trash} (manifest.tsv lists '
          'them). Nothing is deleted yet. To undo: --restore ' + trash)
    return 0


def restore(trash):
    manifest = os.path.join(trash, 'manifest.tsv')
    root = os.path.dirname(os.path.abspath(trash))
    with open(manifest) as fh:
        paths = [r['path'] for r in csv.DictReader(fh, delimiter='\t')]
    bad = 0
    for rel in paths:
        src, dst = os.path.join(trash, rel), os.path.join(root, rel)
        if not os.path.lexists(src):
            print(f'missing in trash: {rel}')
            bad += 1
            continue
        if os.path.lexists(dst):
            print(f'not restored, exists again: {rel}')
            bad += 1
            continue
        os.rename(src, dst)
        print(f'restored {rel}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
