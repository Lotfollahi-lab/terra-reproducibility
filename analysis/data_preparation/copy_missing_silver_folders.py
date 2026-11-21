#!/usr/bin/env python
"""
Copy dataset folders that exist in only one of two silver copies to the
other copy (e.g. the NFS and the Lustre DATASETS roots), so both hold every
dataset. Folders present in both copies are never touched.

Without --run it only lists what it would copy, with sizes, file counts,
symbolic links and the free space at each destination.

With --run the files of all folders are copied by --streams rsync processes
running in parallel (large files one per process, small files in batches),
so several folders and several files of one large folder copy at the same
time. Each folder is copied into a hidden temporary folder
(<dest root>/.<name>.copying); a final rsync pass per folder adds folders,
links and file metadata, then the copy is checked (same number of files,
folders and links, same total file size as the source) and only then
renamed to its final name, so an interrupted copy never looks complete.
Running it again resumes interrupted copies (files already copied are
skipped) and skips finished folders. Symbolic links are copied as links,
not followed.

Example:
    python copy_missing_silver_folders.py                  # dry run
    python copy_missing_silver_folders.py --run --streams 8
    python copy_missing_silver_folders.py --run --direction a_to_b \\
        --exclude hest1k,spatch
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_A = '/nfs/team361/sb75/DATASETS/silver'
ROOT_B = '/lustre/scratch126/cellgen/lotfollahi/DATASETS/silver'
TMP_SUFFIX = '.copying'
# Files of at least this size get an rsync process of their own; smaller
# files are copied in batches of up to BATCH_BYTES / BATCH_FILES
BIG_FILE = 1 << 30
BATCH_BYTES = 4 << 30
BATCH_FILES = 5000


def dataset_folders(root):
    """Dataset folders of a root (hidden folders, incl. copies in progress,
    are left out)."""
    return {d for d in os.listdir(root)
            if not d.startswith('.') and os.path.isdir(os.path.join(root, d))}


def scan(path):
    """Regular files below path as (relative path, size), and counts of
    files, folders, symbolic links and bytes (links are not followed)."""
    files = []
    n_dirs = n_links = 0
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
                files.append((os.path.relpath(fp, path), os.lstat(fp).st_size))
    stats = dict(files=len(files), dirs=n_dirs, links=n_links,
                 bytes=sum(s for _, s in files))
    return files, stats


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


def batches(files):
    """Large files alone, small files grouped; largest first."""
    out, cur, cur_bytes = [], [], 0
    for rel, size in sorted(files, key=lambda x: -x[1]):
        if size >= BIG_FILE:
            out.append(([rel], size))
            continue
        cur.append(rel)
        cur_bytes += size
        if cur_bytes >= BATCH_BYTES or len(cur) >= BATCH_FILES:
            out.append((cur, cur_bytes))
            cur, cur_bytes = [], 0
    if cur:
        out.append((cur, cur_bytes))
    return out


def rsync(args):
    return subprocess.run(['rsync', '-a', '--partial', '--whole-file'] + args,
                          stdout=subprocess.DEVNULL).returncode


class Progress:
    def __init__(self, total):
        self.total, self.done, self.t0 = total, 0, time.time()
        self.lock = threading.Lock()

    def add(self, n, label):
        with self.lock:
            self.done += n
            dt = max(time.time() - self.t0, 1e-6)
            rate = self.done / dt
            eta = (self.total - self.done) / rate if rate > 0 else 0
            print(f'  [{human(self.done)} / {human(self.total)}, '
                  f'{human(rate)}/s, ~{eta / 60:.0f} min left] {label}',
                  flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--root_a', default=ROOT_A)
    p.add_argument('--root_b', default=ROOT_B)
    p.add_argument('--direction', choices=['both', 'a_to_b', 'b_to_a'],
                   default='both')
    p.add_argument('--exclude', default='',
                   help='Comma-separated dataset folders not to copy.')
    p.add_argument('--streams', type=int, default=8,
                   help='rsync processes running in parallel.')
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

    # Scan the sources once, in parallel (metadata reads over NFS are slow)
    def scan_one(item):
        name, src_root, _ = item
        src = os.path.join(src_root, name)
        return item, (None if os.path.islink(src) else scan(src))

    with ThreadPoolExecutor(max_workers=args.streams) as ex:
        scanned = list(ex.map(scan_one, todo))

    totals, links, folders = {}, [], []
    for (name, src_root, dst_root), res in scanned:
        src = os.path.join(src_root, name)
        arrow = 'a -> b' if src_root == args.root_a else 'b -> a'
        totals.setdefault(dst_root, 0)
        if res is None:
            links.append((name, src_root, dst_root))
            print(f'{arrow}  {name:60s} link to {os.path.realpath(src)} '
                  '(the link is recreated, nothing is copied)', flush=True)
            continue
        files, stats = res
        totals[dst_root] += stats['bytes']
        folders.append((name, src_root, dst_root, files, stats))
        print(f'{arrow}  {name:60s} {human(stats["bytes"]):>10s} '
              f'{stats["files"]:7d} files {stats["links"]:5d} links',
              flush=True)
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
    for name, src_root, dst_root in links:
        src, dst = os.path.join(src_root, name), os.path.join(dst_root, name)
        if os.path.lexists(dst):
            print(f'skip {name}: {dst} already exists', flush=True)
            continue
        os.symlink(os.path.realpath(src), dst)
        print(f'done {name}: link to {os.path.realpath(src)}', flush=True)

    # 1. Bulk copy: all files of all folders, --streams rsync processes
    jobs = []
    listdir = tempfile.mkdtemp(prefix='copy_silver_lists_')
    for name, src_root, dst_root, files, stats in folders:
        dst = os.path.join(dst_root, name)
        tmp = os.path.join(dst_root, '.' + name + TMP_SUFFIX)
        if os.path.exists(dst):
            print(f'skip {name}: {dst} already exists', flush=True)
            continue
        if os.path.exists(tmp):
            print(f'resuming the interrupted copy of {name} in {tmp}',
                  flush=True)
        os.makedirs(tmp, exist_ok=True)
        for i, (rels, nbytes) in enumerate(batches(files)):
            lst = os.path.join(listdir, f'{name}.{i}.txt')
            with open(lst, 'w') as fh:
                fh.write('\n'.join(rels) + '\n')
            jobs.append((name, src_root, tmp, lst, len(rels), nbytes))
    jobs.sort(key=lambda j: -j[5])
    progress = Progress(sum(j[5] for j in jobs))
    failed = set()
    print(f'copying {len(jobs)} batches with {args.streams} rsync processes',
          flush=True)

    def run_job(job):
        name, src_root, tmp, lst, n, nbytes = job
        rc = rsync([f'--files-from={lst}',
                    os.path.join(src_root, name) + '/', tmp + '/'])
        return job, rc

    with ThreadPoolExecutor(max_workers=args.streams) as ex:
        for fut in as_completed([ex.submit(run_job, j) for j in jobs]):
            (name, _, tmp, _, n, nbytes), rc = fut.result()
            if rc != 0:
                failed.add(name)
                print(f'  FAILED batch of {name} ({n} files): rsync exit '
                      f'code {rc}', flush=True)
            progress.add(nbytes, f'{name}: {n} file{"s" * (n > 1)}')
    shutil.rmtree(listdir, ignore_errors=True)

    # 2. Per folder: final pass (folders, links, metadata), check, rename
    def finish(item):
        name, src_root, dst_root, files, stats = item
        dst = os.path.join(dst_root, name)
        tmp = os.path.join(dst_root, '.' + name + TMP_SUFFIX)
        if os.path.exists(dst):
            return True
        if name in failed:
            return f'FAILED {name}: a batch failed; the partial copy stays ' \
                   f'in {tmp}; run again to resume'
        rc = rsync([os.path.join(src_root, name) + '/', tmp + '/'])
        if rc != 0:
            return f'FAILED {name}: final rsync pass exit code {rc}; the ' \
                   f'partial copy stays in {tmp}'
        _, copied = scan(tmp)
        if copied != stats:
            return f'FAILED {name}: copy does not match the source (source ' \
                   f'{stats}, copy {copied}); it stays in {tmp}'
        os.rename(tmp, dst)
        return f'done {name}: {stats["files"]} files, {human(stats["bytes"])}'

    print('\nfinal pass, check and rename per folder', flush=True)
    with ThreadPoolExecutor(max_workers=args.streams) as ex:
        for msg in ex.map(finish, folders):
            if msg is True:
                continue
            print('  ' + msg, flush=True)
            ok &= not msg.startswith('FAILED')
    print('\nAll copies finished and checked.' if ok else
          '\nSome copies failed (see above); run again to resume them.')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
