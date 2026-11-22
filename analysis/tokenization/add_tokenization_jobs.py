#!/usr/bin/env python
"""
Add tokenization jobs on another queue (e.g. long, when it is free) for the
hst_corpus_110m datasets that are not tokenized yet and that no job is
tokenizing, next to the jobs already waiting on other queues. All jobs run
run_tokenization_locked.sh, so each dataset is still tokenized by one job
only: whichever starts first tokenizes, the others exit when they start (or
take over if it failed).

What it does:
1. Waiting (PEND or PSUSP) tokenization jobs that run
   create_tokenized_dataset.sh directly (submitted before the wrapper used
   the lock) are switched to run_tokenization_locked.sh with bmod -Z. They
   keep their place in the queue. Otherwise such a job could start while a
   locked job tokenizes the same dataset.
2. Each dataset of --names (or --datasets) is skipped if
   - its part is finished (its settings record terra_tokenizer_params.json
     exists),
   - one of its jobs is running,
   - one of its jobs already waits on --queue (unless it asks for no more
     memory than an attempt that hit the memory limit),
   - its last attempt failed with an error (--retry_failed submits these),
     or ran out of time with at least --walltime,
   - it is listed in --exclude, has no memory estimate, or needs more
     memory than --queue allows;
   otherwise a locked job is submitted on --queue with
   submit_tokenization_twin.sh.
Without --submit it only prints what it would do. At the end it lists the
datasets that are not tokenized and have no job that can succeed, with the
command for the ones that need a queue with larger machines.

Memory per job: 1.5 x estimate + 32 GB, rounded up to 16 GB (the rule of the
first submissions), with the estimate from the table of
analysis/data_preparation/checks/tokenization_memory.py (--memory_estimates).
It is raised above the largest peak memory of the earlier attempts of the
same build and above the largest request that hit the memory limit (from
their LSF reports in logs/<dataset>/farm-job-outputs/cell-graph-tokenizer/).
If that is more than --max_ram_gb, the job gets --max_ram_gb ("tight") when
that is still at least --min_headroom x estimate + 16 GB and more than any
request that hit the memory limit (--no_tight: never). A tight job that
starts before a waiting hugemem job of the same dataset makes that job exit
when it starts during the run; if the tight job then hits the memory limit,
the dataset appears in the final list of datasets without a job.

Usage (farm head node, from the terra-reproducibility root, TERRA venv):
    python analysis/tokenization/add_tokenization_jobs.py --queue long --walltime 48:00             # preview
    python analysis/tokenization/add_tokenization_jobs.py --queue long --walltime 48:00 --submit    # do it
    # datasets that need larger machines, as listed at the end:
    python analysis/tokenization/add_tokenization_jobs.py --queue hugemem --walltime 72:00 --max_ram_gb 3800 --datasets <names> --submit
Running it again later is safe: datasets that are done, running or already
waiting on the queue are skipped. Wait a minute between --submit runs (LSF
may answer queries from a snapshot that is up to a minute old).
"""
import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

TOKENIZER_NAME = 'cell-graph-tokenizer'
LOCKED_SCRIPT = 'analysis/tokenization/run_tokenization_locked.sh'
TWIN_SCRIPT = 'analysis/tokenization/submit_tokenization_twin.sh'
DEFAULT_NAMES = ('analysis/data_preparation/hst_corpus_110m/'
                 'hst_corpus_110m_dataset_names.txt')
JOB_NAME_RE = re.compile(rf'^(?P<dataset>.+)_{TOKENIZER_NAME}_(?P<cores>\d+)$')
# States of a started job (UNKWN and ZOMBI follow host problems, when the
# job may still run)
ACTIVE_STATES = {'RUN', 'USUSP', 'SSUSP', 'UNKWN', 'ZOMBI', 'PROV', 'WAIT'}
# States of a job that has not started (PSUSP: held while pending)
WAITING_STATES = {'PEND', 'PSUSP'}
DELIM = '^'
MAX_WALLTIME_MINUTES = {'cpu-interactive': 12 * 60, 'long': 48 * 60}
# Most memory a job may have on these queues (683.5 GB per job)
QUEUE_MAX_GB = {'cpu-interactive': 683.5, 'long': 683.5}
UNIT_GB = {'K': 1 / 1024 ** 2, 'M': 1 / 1024, 'G': 1.0, 'T': 1024.0}


@dataclass
class Job:
    jobid: str
    stat: str
    queue: str
    name: str
    dataset: str
    cores: str
    command: str
    memlimit_gb: Optional[float] = None

    @property
    def locked(self) -> bool:
        return 'run_tokenization_locked.sh' in self.command

    @property
    def unlocked_tokenizer(self) -> bool:
        return (not self.locked) and 'create_tokenized_dataset.sh' in self.command


@dataclass
class Attempt:
    """One earlier job of this build, from its LSF output file."""
    jobid: int
    path: Path
    mtime: float
    outcome: str  # done / memlimit / runlimit / killed / failed / unfinished
    exit_code: Optional[int] = None
    requested_gb: Optional[float] = None
    max_memory_gb: Optional[float] = None
    run_seconds: Optional[float] = None


@dataclass
class Plan:
    dataset: str
    estimate_gb: Optional[int]
    action: str  # done / running / waiting / skip / submit
    ram_gb: Optional[int] = None
    notes: List[str] = field(default_factory=list)
    has_job: bool = False       # a running job, or a waiting job that can succeed
    needs_bigger: bool = False  # needs more memory than --queue allows
    warning: bool = False


def round_up_16(gb: float) -> int:
    return int(math.ceil(gb / 16.0)) * 16


def request_rule_gb(estimate_gb: int) -> int:
    """1.5 x estimate + 32 GB, rounded up to 16 GB (integer arithmetic, as
    in the shell commands of the first submissions)."""
    return (estimate_gb * 3 // 2 + 32 + 15) // 16 * 16


def walltime_minutes(walltime: str) -> int:
    if ':' in walltime:
        hours, minutes = walltime.split(':', 1)
        return int(hours) * 60 + int(minutes)
    return int(walltime)


def parse_memory_gb(text: str) -> Optional[float]:
    """'672 G', '688128 M', '672G', '1.5 T' -> GB; None without a unit (the
    unit of bare numbers depends on LSF_UNIT_FOR_LIMITS) or for '-'."""
    m = re.match(r'^\s*([\d.]+)\s*([KMGT])B?\s*$', text or '', re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1)) * UNIT_GB[m.group(2).upper()]


def load_config(config_file: Path) -> dict:
    with open(config_file) as f:
        return yaml.safe_load(f)


def record_path(config_file: Path) -> Path:
    """The settings record of the dataset's finished part, named as
    create_tokenized_dataset.py names the part (the same rule as
    run_tokenization_locked.sh)."""
    c = load_config(config_file)
    t, p, e = c['tokenization'], c['paths'], c['experiment']
    graph = ''
    if t.get('n_neighs'):
        graph += f"knn_{t['n_neighs']}_"
    if t.get('radius'):
        graph += f"radius_{t['radius']}_"
    if t.get('delaunay'):
        graph += 'delaunay_'
    name = (f"{e['experiment_name']}_{t['num_shards']}_{t.get('rank_cell_norm_method')}_"
            f"{t.get('rank_gene_norm_method')}_{t.get('rank_count_norm_method')}_"
            f"{t.get('count_cell_norm_method')}_{t.get('count_gene_norm_method')}_"
            f"{t.get('count_count_norm_method')}_{graph}{e['model_input_size']}"
            f"{'_incl_zero_expr' if t.get('include_zero_expr_genes') else ''}")
    return Path(f"{p['data_directory_path'].rstrip('/')}/gold/{e['tokenizer']}/"
                f"{e['experiment_name']}/{name}.dataset/terra_tokenizer_params.json")


def build_markers(config_file: Path) -> List[str]:
    """Lines create_tokenized_dataset.py prints for this config: an output
    file with all of them comes from a job of this build (older builds of the
    same dataset wrote to the same log folder; locked jobs that exit without
    tokenizing print none)."""
    c = load_config(config_file)
    markers = [f"Model input size: {c['experiment']['model_input_size']}."]
    storage_format = c['tokenization'].get('storage_format')
    if storage_format:
        markers.append(f'Storage format: {storage_format}.')
    return markers


def read_estimates(path: Path) -> Dict[str, Optional[int]]:
    """dataset -> estimate (GB) from the tokenization_memory.py table; None
    for MISSING rows (no silver files)."""
    estimates = {}
    with open(path) as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) != 10 or row[0] == 'dataset':
                continue
            if row[9] == 'MISSING' or not row[8].strip():
                estimates[row[0]] = None
                continue
            try:
                estimates[row[0]] = int(math.ceil(float(row[8])))
            except ValueError:
                continue
    return estimates


def run_bjobs(fields: List[str]):
    cmd = ['bjobs', '-noheader', '-o',
           ' '.join(fields) + f" delimiter='{DELIM}'"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode, res.stdout.strip(), res.stderr.strip()


def query_jobs() -> List[Job]:
    """The user's unfinished tokenization jobs. Exits if LSF does not
    answer (submitting without knowing the waiting jobs could start a second
    job without the lock)."""
    fields = ['jobid', 'stat', 'queue', 'job_name', 'memlimit', 'command']
    rc, out, err = run_bjobs(fields)
    if not out and 'No unfinished job found' not in err:
        # Without the memory limit field (if this LSF does not know it)
        fields.remove('memlimit')
        rc2, out, err2 = run_bjobs(fields)
        if not out and 'No unfinished job found' not in err2:
            sys.exit(f'bjobs failed (exit code {rc}): {err} {err2}')
        print('  (bjobs has no memlimit field; memory limits of waiting jobs '
              'are not checked)')
    jobs = []
    for line in out.splitlines():
        parts = line.split(DELIM, len(fields) - 1)
        if len(parts) != len(fields):
            continue
        values = dict(zip(fields, (x.strip() for x in parts)))
        m = JOB_NAME_RE.match(values['job_name'])
        if not m:
            continue
        jobs.append(Job(values['jobid'], values['stat'], values['queue'],
                        values['job_name'], m['dataset'], m['cores'],
                        values['command'],
                        parse_memory_gb(values.get('memlimit', ''))))
    return jobs


REPORT_START = 'Sender: LSF System'
REPORT_OUTPUT = 'The output (if any) follows:'


def lsf_verdict(text: str):
    """(outcome, exit code, requested GB, peak GB, run seconds) from the last
    LSF job report in an output file (a requeued job appends another
    report); the outcome is 'unfinished' without a report."""
    start = text.rfind(REPORT_START)
    if start < 0:
        return 'unfinished', None, None, None, None
    end = text.find(REPORT_OUTPUT, start)
    report = text[start:end if end >= 0 else len(text)]
    requested = re.search(r'Total Requested Memory :\s+([\d.]+) MB', report)
    requested_gb = float(requested.group(1)) / 1024.0 if requested else None
    peak = re.search(r'Max Memory :\s+([\d.]+) (MB|GB)', report)
    peak_gb = ((float(peak.group(1)) / (1024.0 if peak.group(2) == 'MB' else 1.0))
               if peak else None)
    run = re.search(r'Run time :\s+([\d.]+) sec', report)
    run_seconds = float(run.group(1)) if run else None
    exited = re.search(r'Exited with exit code (\d+)', report)
    exit_code = int(exited.group(1)) if exited else None
    if 'TERM_MEMLIMIT' in report:
        outcome = 'memlimit'
    elif 'TERM_RUNLIMIT' in report:
        outcome = 'runlimit'
    elif re.search(r'TERM_[A-Z_]+', report):
        outcome = 'killed'
    elif exited:
        if requested_gb and peak_gb and peak_gb >= 0.95 * requested_gb:
            # Out of memory without an LSF verdict (e.g. a worker killed by
            # the kernel, then BrokenProcessPool)
            outcome = 'memlimit'
        elif exit_code > 128:
            outcome = 'killed'  # by a signal
        else:
            outcome = 'failed'
    elif 'Successfully completed.' in report:
        outcome = 'done'
    else:
        outcome = 'unfinished'
    return outcome, exit_code, requested_gb, peak_gb, run_seconds


def parse_attempt(path: Path, markers: List[str]) -> Optional[Attempt]:
    """An earlier job of this build from its LSF output file, or None."""
    m = re.search(r'_(\d+)\.out$', path.name)
    if not m:
        return None
    try:
        text = path.read_text(errors='replace')
        mtime = path.stat().st_mtime
    except OSError:
        return None
    if not all(marker in text for marker in markers):
        return None
    return Attempt(int(m.group(1)), path, mtime, *lsf_verdict(text))


def all_attempts(dataset: str, markers: List[str]) -> List[Attempt]:
    """Attempts of this build, oldest first (LSF writes the report into the
    output file when the job ends, so the file time orders them; job ids do
    not: an old waiting job can run after a newer one)."""
    log_dir = Path('logs') / dataset / 'farm-job-outputs' / TOKENIZER_NAME
    attempts = [a for a in (parse_attempt(p, markers)
                            for p in log_dir.glob('*.out')) if a is not None]
    return sorted(attempts, key=lambda a: (a.mtime, a.jobid))


def memlimit_floor(attempts: List[Attempt]) -> Tuple[float, Optional[Attempt]]:
    """The largest request (GB) that hit the memory limit, and its attempt."""
    floor, which = 0.0, None
    for a in attempts:
        if a.outcome == 'memlimit':
            gb = a.requested_gb or a.max_memory_gb or 0.0
            if gb > floor:
                floor, which = gb, a
    return floor, which


def plan_memory(estimate_gb: int, attempts: List[Attempt], max_ram_gb: int,
                min_headroom: float, tight: bool):
    """(GB to request or None, notes, needs a queue with larger machines)."""
    notes = []
    need = request_rule_gb(estimate_gb)
    peak = max((a.max_memory_gb for a in attempts if a.max_memory_gb),
               default=0.0)
    if peak:
        need = max(need, round_up_16(peak * 1.2 + 16))
    floor, which = memlimit_floor(attempts)
    if which is not None:
        notes.append(f'job {which.jobid} hit the memory limit at {floor:.0f}G')
        need = max(need, round_up_16(floor * 1.25))
    if need <= max_ram_gb:
        return need, notes, False
    if max_ram_gb <= floor:
        return None, notes + [f'needs more than {max_ram_gb}G'], True
    if tight and max_ram_gb >= min_headroom * max(estimate_gb, peak) + 16:
        return max_ram_gb, notes + [f'tight: {need}G by the rule, '
                                    f'{max_ram_gb}G is the most allowed'], False
    return None, notes + [f'needs {need}G, more than {max_ram_gb}G'], True


def err_tail(attempt: Attempt) -> str:
    """Last non-empty line of the job's .err (the Python traceback goes
    there), for the note."""
    err = attempt.path.with_suffix('.err')
    try:
        lines = [l.strip() for l in err.read_text(errors='replace').splitlines()
                 if l.strip()]
    except OSError:
        return ''
    return lines[-1][:160] if lines else ''


def stuck_lock_marker(dataset: str, minutes: float = 5) -> Optional[Path]:
    """The takeover marker of the dataset's current lock if it is older than
    `minutes`: the job that made it died before moving the lock (it does so
    within moments), and every later job exits until it is removed."""
    lock = Path(os.environ.get('TOKENIZATION_LOCK_DIR',
                               'logs/tokenization_locks')) / f'{dataset}.lock'
    try:
        marker = lock.parent / f'{dataset}.lock.takeover_i{lock.stat().st_ino}'
        age = time.time() - marker.stat().st_mtime
    except OSError:
        return None
    return marker if age > 60 * minutes else None


def can_succeed(job: Job, floor: float) -> bool:
    """False if the waiting job asks for no more memory than a request that
    already hit the memory limit (known from its memory limit, or from its
    queue's maximum)."""
    if not floor:
        return True
    limit = job.memlimit_gb
    if limit is None:
        limit = QUEUE_MAX_GB.get(job.queue)
        if limit is None:
            return True
    return limit > floor * 1.02


def make_plans(names, jobs, estimates, excluded, args) -> List[Plan]:
    by_dataset: Dict[str, List[Job]] = {}
    for job in jobs:
        by_dataset.setdefault(job.dataset, []).append(job)
    plans = []
    for dataset in names:
        config = Path('config') / TOKENIZER_NAME / f'{TOKENIZER_NAME}_{dataset}.yaml'
        plan = Plan(dataset, estimates.get(dataset), 'skip')
        plans.append(plan)
        if not config.is_file():
            plan.notes.append(f'no config {config}')
            continue
        if record_path(config).is_file():
            plan.action = 'done'
            continue
        own = by_dataset.get(dataset, [])
        active = [j for j in own if j.stat in ACTIVE_STATES]
        if active:
            plan.action, plan.has_job = 'running', True
            plan.notes.append(', '.join(
                f'{j.jobid} on {j.queue} ({j.stat}'
                f"{'' if j.locked else ', without the lock'})" for j in active))
            started = [j for j in active if j.stat != 'UNKWN']
            if len(started) > 1 and any(not j.locked for j in started):
                plan.warning = True
                plan.notes.append(
                    'WARNING: two jobs may be writing this part: compare '
                    f"bjobs -o 'jobid stat start_time' {' '.join(j.jobid for j in started)} "
                    'and bkill the newer one')
            continue
        marker = stuck_lock_marker(dataset)
        if marker is not None:
            plan.warning = True
            plan.notes.append('the lock is stuck (a job died while taking it '
                              f'over): rm -r {marker}   (not the lock itself)')
            continue
        attempts = all_attempts(dataset, build_markers(config))
        floor, _ = memlimit_floor(attempts)
        waiting = [j for j in own if j.stat in WAITING_STATES]
        good = [j for j in waiting if can_succeed(j, floor)]
        bad = [j for j in waiting if not can_succeed(j, floor)]
        for j in bad:
            plan.warning = True
            plan.notes.append(f'waiting job {j.jobid} on {j.queue} will hit the '
                              f'memory limit again ({floor:.0f}G was too '
                              f'little): bkill {j.jobid}')
        here = [j for j in good if j.queue == args.queue]
        if here:
            plan.action, plan.has_job = 'waiting', True
            plan.notes.append(f'already waiting on {args.queue}: '
                              + ', '.join(j.jobid for j in here))
            continue
        if good:
            plan.has_job = True
            plan.notes.append('also waiting: ' + ', '.join(
                f'{j.jobid} on {j.queue}' for j in good))
        last = attempts[-1] if attempts else None
        if last is not None and last.outcome == 'failed' and not args.retry_failed:
            tail = err_tail(last)
            plan.notes.append(
                f'job {last.jobid} failed (exit code {last.exit_code}): '
                f"{last.path.with_suffix('.err')}"
                + (f' ({tail})' if tail else '') + '; --retry_failed resubmits'
                + (' (the waiting jobs will retry it when they start)'
                   if good else ''))
            continue
        # The longest run that ran out of time (a later, shorter one does not
        # show that --walltime is enough)
        timed_out = max((a for a in attempts
                         if a.outcome == 'runlimit' and a.run_seconds),
                        key=lambda a: a.run_seconds, default=None)
        if (timed_out is not None and timed_out.run_seconds
                >= 0.95 * 60 * walltime_minutes(args.walltime)):
            plan.notes.append(f'job {timed_out.jobid} ran out of time after '
                              f'{timed_out.run_seconds / 3600:.1f} h: needs '
                              f'more than {args.walltime}')
            continue
        if last is not None and last.outcome == 'runlimit':
            plan.notes.append(f'job {last.jobid} ran out of time')
        elif last is not None and last.outcome == 'killed':
            plan.notes.append(f'job {last.jobid} was killed')
        if dataset in excluded:
            plan.notes.append(f'excluded ({args.exclude})')
            continue
        if plan.estimate_gb is None:
            plan.notes.append('no memory estimate (not in '
                              f'{args.memory_estimates}, or no silver files)')
            continue
        ram, notes, needs_bigger = plan_memory(
            plan.estimate_gb, attempts, args.max_ram_gb, args.min_headroom,
            not args.no_tight)
        plan.notes.extend(notes)
        plan.needs_bigger = needs_bigger
        if ram is None:
            continue
        plan.action, plan.ram_gb = 'submit', ram
    return plans


def locked_command(dataset: str) -> str:
    return (f'{LOCKED_SCRIPT} config/{TOKENIZER_NAME}/'
            f'{TOKENIZER_NAME}_{dataset}.yaml')


def convert_unlocked(jobs: List[Job], args) -> Tuple[List[str], List[str]]:
    """bmod -Z the waiting jobs that run create_tokenized_dataset.sh without
    the lock; returns the job ids switched and those that could not be."""
    switched, failed = [], []
    for job in jobs:
        if job.stat not in WAITING_STATES or not job.unlocked_tokenizer:
            continue
        expected = f'config/{TOKENIZER_NAME}/{TOKENIZER_NAME}_{job.dataset}.yaml'
        if expected not in job.command:
            print(f'  {job.jobid} {job.name}: command does not use {expected} '
                  f'({job.command}); left unchanged')
            failed.append(job.jobid)
            continue
        if not args.submit:
            print(f'  would switch {job.jobid} {job.name} ({job.queue}, '
                  f'{job.stat}) to the lock')
            switched.append(job.jobid)
            continue
        cmd = ['bmod', '-G', args.group, '-Z', locked_command(job.dataset),
               job.jobid]
        res = subprocess.run(cmd, capture_output=True, text=True)
        ok = res.returncode == 0
        print(f"  {'switched' if ok else 'COULD NOT switch'} {job.jobid} "
              f"{job.name} ({job.queue}) {(res.stdout + res.stderr).strip()}")
        (switched if ok else failed).append(job.jobid)
    return switched, failed


def submit(plan: Plan, args) -> bool:
    cmd = ['bash', TWIN_SCRIPT, plan.dataset, args.queue, f'{plan.ram_gb}G',
           args.walltime, str(args.cores)]
    env = dict(os.environ, CWD=os.getcwd())
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = (res.stdout + res.stderr).strip().replace('\n', ' | ')
    ok = res.returncode == 0 and 'is submitted' in res.stdout
    print(f"  {'submitted' if ok else 'FAILED to submit'} {plan.dataset}: {out}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--queue', required=True)
    ap.add_argument('--walltime', required=True, help='e.g. 48:00')
    ap.add_argument('--max_ram_gb', type=int, default=672,
                    help='Most memory a job may request on --queue '
                         '(long and cpu-interactive: 683.5 GB per job).')
    ap.add_argument('--min_headroom', type=float, default=1.1,
                    help='Request --max_ram_gb for a dataset above it only '
                         'if --max_ram_gb >= this x estimate + 16 GB.')
    ap.add_argument('--no_tight', action='store_true',
                    help='Never request --max_ram_gb for datasets whose rule '
                         'asks for more.')
    ap.add_argument('--cores', type=int, default=12)
    ap.add_argument('--group', default='team361')
    ap.add_argument('--names', default=DEFAULT_NAMES)
    ap.add_argument('--datasets', nargs='+',
                    help='Only these datasets (default: all of --names).')
    ap.add_argument('--exclude',
                    help='File with dataset names (one per line) to leave out.')
    ap.add_argument('--memory_estimates',
                    default=os.path.expanduser('~/tok_mem.tsv'),
                    help='Output of tokenization_memory.py.')
    ap.add_argument('--retry_failed', action='store_true',
                    help='Also submit datasets whose last attempt failed '
                         'with an error.')
    ap.add_argument('--quiet', action='store_true',
                    help='Do not list datasets that are running or waiting '
                         'without a warning.')
    ap.add_argument('--submit', action='store_true',
                    help='Switch waiting jobs to the lock and submit (default: '
                         'only print what would be done).')
    args = ap.parse_args()

    if 'gpu' in args.queue or args.queue == 'inference':
        sys.exit(f"Queue '{args.queue}' is a GPU queue; tokenization runs on "
                 'CPU queues.')
    limit = MAX_WALLTIME_MINUTES.get(args.queue)
    if limit is not None and walltime_minutes(args.walltime) > limit:
        sys.exit(f'Queue {args.queue} allows at most {limit // 60}:00, not '
                 f'{args.walltime}.')
    if args.max_ram_gb > QUEUE_MAX_GB.get(args.queue, float('inf')):
        sys.exit(f'Queue {args.queue} allows at most '
                 f'{QUEUE_MAX_GB[args.queue]} GB per job.')
    if not (Path('config') / TOKENIZER_NAME).is_dir() or not Path(LOCKED_SCRIPT).is_file():
        sys.exit('Run from the terra-reproducibility root.')

    names = [line.split(',')[0].strip() for line in open(args.names)
             if line.strip()]
    if args.datasets:
        unknown = sorted(set(args.datasets) - set(names))
        if unknown:
            sys.exit(f'Not in {args.names}: {unknown}')
        names = [n for n in names if n in set(args.datasets)]
    selected = set(names)
    excluded = set()
    if args.exclude:
        excluded = {l.split(',')[0].strip() for l in open(args.exclude)
                    if l.strip()}
    estimates = read_estimates(Path(args.memory_estimates))

    print('1. Waiting jobs without the lock:')
    jobs = query_jobs()
    scoped = [j for j in jobs if j.dataset in selected]
    switched, failed_ids = convert_unlocked(scoped, args)
    if not switched and not failed_ids:
        print('  none')
    if args.submit and switched:
        # Again for the states: a job may have started meanwhile
        jobs = query_jobs()
    # bmod -Z returned 0 for these, so they run the lock whenever they start
    # (a fresh query can still show the old command for a while)
    for job in jobs:
        if job.jobid in switched and job.stat in WAITING_STATES:
            job.command = locked_command(job.dataset)
    blocked = {j.dataset for j in jobs
               if j.stat in WAITING_STATES and not j.locked}

    plans = make_plans(names, jobs, estimates, excluded, args)
    for plan in plans:
        if plan.action == 'submit' and plan.dataset in blocked:
            plan.action, plan.ram_gb = 'skip', None
            plan.notes.append('a waiting job of it does not use the lock '
                              '(could not be switched)')

    print(f'\n2. Datasets ({len(plans)}), queue {args.queue}, at most '
          f'{args.max_ram_gb}G, {args.walltime}, {args.cores} cores:')
    width = max(len(p.dataset) for p in plans) if plans else 10
    order = {'submit': 0, 'skip': 1, 'running': 2, 'waiting': 3, 'done': 4}
    for plan in sorted(plans, key=lambda p: (order[p.action], p.dataset)):
        if plan.action == 'done' or (plan.action in ('running', 'waiting')
                                     and not plan.warning and args.quiet):
            continue
        est = '-' if plan.estimate_gb is None else f'{plan.estimate_gb}G'
        ram = f'{plan.ram_gb}G' if plan.ram_gb else ''
        verb = ('would submit' if plan.action == 'submit' and not args.submit
                else plan.action)
        print(f'  {plan.dataset:{width}s}  est {est:>6s}  {verb:12s} {ram:>6s}  '
              + '; '.join(plan.notes))
    counts = {}
    for plan in plans:
        counts[plan.action] = counts.get(plan.action, 0) + 1
    print('\n' + ', '.join(f'{v} {k}' for k, v in sorted(counts.items())))

    no_job = [p for p in plans if p.action == 'skip' and not p.has_job]
    if no_job:
        print(f'\nNot tokenized and no job that can succeed ({len(no_job)}):')
        for plan in no_job:
            print(f'  {plan.dataset}: ' + '; '.join(plan.notes))
        bigger = [p.dataset for p in no_job if p.needs_bigger]
        if bigger:
            print(f'Queue with larger machines for {len(bigger)} of them:\n'
                  '  python analysis/tokenization/add_tokenization_jobs.py '
                  '--queue hugemem --walltime 72:00 --max_ram_gb 3800 '
                  f"--datasets {' '.join(bigger)}")
    warnings = [p.dataset for p in plans if p.warning]
    if warnings:
        print(f"\nWARNINGS for {len(warnings)} dataset(s): {' '.join(warnings)} "
              '(see their lines above)')

    if not args.submit:
        print('\nNothing changed. Add --submit to switch the waiting jobs and '
              'submit.')
        return
    print('\n3. Submitting:')
    n_ok = sum(submit(plan, args) for plan in plans if plan.action == 'submit')
    n_planned = sum(plan.action == 'submit' for plan in plans)
    print(f'\n{n_ok} of {n_planned} jobs submitted on {args.queue}.')
    if n_ok < n_planned or failed_ids:
        sys.exit(1)


if __name__ == '__main__':
    main()
