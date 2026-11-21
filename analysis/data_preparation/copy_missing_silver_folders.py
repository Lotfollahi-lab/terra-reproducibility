#!/usr/bin/env python
"""
Copy dataset folders that exist in only one of two silver copies to the
other copy (e.g. the NFS and the Lustre DATASETS roots), so both hold every
dataset. Folders present in both copies are never touched.

Without --run it only lists what it would copy, with sizes, file counts,
symbolic links and the free space at each destination.

With --run each folder is copied with rsync into a hidden temporary folder
(<dest root>/.<name>.copying), checked (same number of files, folders and
links, same total file size as the source) and only then renamed to its
final name, so an interrupted copy never looks complete. Running it again
resumes interrupted copies and skips finished ones. Symbolic links are
copied as links (rsync -a), not followed.

Example:
    python copy_missing_silver_folders.py                  # dry run
    python copy_missing_silver_folders.py --run            # copy
    python copy_missing_silver_folders.py --run --direction a_to_b \\
        --exclude hest1k,spatch
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT_A = '/nfs/team361/sb75/DATASETS/silver'
ROOT_B = '/lustre/scratch126/cellgen/lotfollahi/DATASETS/silver'
TMP_SUFFIX = '.copying'


def dataset_folders(root):
    """Dataset folders of a root (hidden folders, incl. copies in progress,
    are left out)."""
    return {d for d in os.listdir(root)
            if not d.startswith('.') and os.path.isdir(os.path.join(root, d))}


def tree_stats(path):
    """Regular files, their total size, folders and symbolic links below
    path (links are not followed)."""
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


def plan(root_a, root_b, direction, excluded):
    a, b = dataset_folders(root_a), dataset_folders(root_b)
    todo = []
    if direction in ('both', 'a_to_b'):
        todo += [(d, root_a, root_b) for d in sorted(a - b) if d not in excluded]
    if direction in ('both', 'b_to_a'):
        todo += [(d, root_b, root_a) for d in sorted(b - a) if d not in excluded]
    return todo


def copy_one(name, src_root, dst_root):
    src = os.path.join(src_root, name)
    dst = os.path.join(dst_root, name)
    tmp = os.path.join(dst_root, '.' + name + TMP_SUFFIX)
    if os.path.exists(dst):
        print(f'  skip {name}: {dst} already exists', flush=True)
        return True
    if os.path.islink(src):
        # A dataset folder that is itself a link: make the same link (to the
        # absolute target), rather than copying the data it points to
        target = os.path.realpath(src)
        os.symlink(target, dst)
        print(f'  done {name}: link to {target}', flush=True)
        return True
    if os.path.exists(tmp):
        print(f'  resuming the interrupted copy in {tmp}', flush=True)
    t0 = time.time()
    cmd = ['rsync', '-a', '--partial', src + '/', tmp + '/']
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f'  FAILED {name}: rsync exit code {res.returncode}; '
              f'the partial copy stays in {tmp}', flush=True)
        return False
    s_src, s_tmp = tree_stats(src), tree_stats(tmp)
    if s_src != s_tmp:
        print(f'  FAILED {name}: copy does not match the source '
              f'(source {s_src}, copy {s_tmp}); it stays in {tmp}',
              flush=True)
        return False
    os.rename(tmp, dst)
    print(f'  done {name}: {s_src["files"]} files, {human(s_src["bytes"])} '
          f'in {time.time() - t0:.0f}s', flush=True)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--root_a', default=ROOT_A)
    p.add_argument('--root_b', default=ROOT_B)
    p.add_argument('--direction', choices=['both', 'a_to_b', 'b_to_a'],
                   default='both')
    p.add_argument('--exclude', default='',
                   help='Comma-separated dataset folders not to copy.')
    p.add_argument('--run', action='store_true',
                   help='Copy. Without it, only list what would be copied.')
    args = p.parse_args()
    excluded = {x for x in args.exclude.split(',') if x}

    todo = plan(args.root_a, args.root_b, args.direction, excluded)
    if not todo:
        print('Nothing to copy: no dataset folder is missing from either '
              'copy (apart from excluded ones).')
        return 0
    print(f'a = {args.root_a}\nb = {args.root_b}\n')
    totals = {}
    for name, src_root, dst_root in todo:
        src = os.path.join(src_root, name)
        arrow = 'a -> b' if src_root == args.root_a else 'b -> a'
        totals.setdefault(dst_root, 0)
        if os.path.islink(src):
            print(f'{arrow}  {name:60s} link to {os.path.realpath(src)} '
                  '(the link is recreated, nothing is copied)', flush=True)
            continue
        s = tree_stats(src)
        totals[dst_root] += s['bytes']
        print(f'{arrow}  {name:60s} {human(s["bytes"]):>10s} '
              f'{s["files"]:7d} files {s["links"]:5d} links', flush=True)
    print()
    for dst_root, n in totals.items():
        free = shutil.disk_usage(dst_root).free
        print(f'to {dst_root}: {human(n)} to copy, {human(free)} free on '
              f'that file system (quotas are not checked)')
    if excluded:
        print(f'excluded: {",".join(sorted(excluded))}')
    if not args.run:
        print('\nDry run; add --run to copy.')
        return 0

    print()
    ok = True
    for name, src_root, dst_root in todo:
        print(f'copying {name}: {src_root} -> {dst_root}', flush=True)
        ok &= copy_one(name, src_root, dst_root)
    print('\nAll copies finished and checked.' if ok else
          '\nSome copies failed (see above); run again to resume them.')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
