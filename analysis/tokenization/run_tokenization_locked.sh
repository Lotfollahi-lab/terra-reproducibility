#!/bin/bash
# Run create_tokenized_dataset.sh <config> under a per-dataset lock, so that
# several jobs for the same dataset (e.g. one on hugemem, one on
# cpu-interactive and one on long) never tokenize it at the same time:
# - the dataset is already tokenized (the settings record of its part
#   exists): exit 0;
# - a job of the dataset that runs without the lock (submitted with
#   wrapper_create_cell_graph_tokenized_dataset.sh before it used this
#   script) is running: exit 0;
# - another job holds the lock and is still running (or LSF cannot tell,
#   e.g. UNKWN / ZOMBI after a host problem): exit 0;
# - the job holding the lock has ended without finishing (TERM_MEMLIMIT,
#   TERM_RUNLIMIT, killed, requeued): take the lock over and tokenize;
# - LSF does not answer (after retries), the lock folder cannot be made
#   (e.g. quota), or a lock is stuck (a job died while taking it over):
#   exit 3 without tokenizing, with the reason in the .err file.
# Locks: <terra-reproducibility root>/logs/tokenization_locks/<dataset>.lock
# with the owner's job id in 'owner' (mkdir is atomic, so only one job wins).
# A lock taken over is moved to <dataset>.lock.ended_i<inode>; the marker
# <dataset>.lock.takeover_i<inode> lets only one job take over each lock.
# Only the record says a dataset is finished: a 'done' file in the lock is
# written for information.
#
# Usage (as the command of an LSF job started in the terra-reproducibility
# root, e.g. by wrapper_create_cell_graph_tokenized_dataset.sh,
# submit_tokenization_twin.sh or add_tokenization_jobs.py, or set on a
# pending job with bmod -Z):
#   analysis/tokenization/run_tokenization_locked.sh config/cell-graph-tokenizer/cell-graph-tokenizer_<dataset>.yaml
CONFIG_FILE="${1:?usage: run_tokenization_locked.sh <config file>}"
DATASET=$(basename "$CONFIG_FILE" .yaml)
DATASET=${DATASET#cell-graph-tokenizer_}
# Job names of this dataset's tokenization jobs: <dataset>_cell-graph-tokenizer_<cores>
JOB_NAME_PREFIX="${DATASET}_cell-graph-tokenizer_"
LOCK_DIR="${TOKENIZATION_LOCK_DIR:-logs/tokenization_locks}"
LOCK="${LOCK_DIR}/${DATASET}.lock"
JOB="${LSB_JOBID:-$$}"
# LSF lists an array element as <job id>[<index>]
SELF="$JOB"
if [ -n "$LSB_JOBID" ] && [ -n "$LSB_JOBINDEX" ] && [ "$LSB_JOBINDEX" != 0 ]; then
    SELF="${LSB_JOBID}[${LSB_JOBINDEX}]"
fi
RETRY_SLEEP="${TOKENIZATION_LSF_RETRY_SLEEP:-30}"
# How long to wait for the owner file of a lock (NFS can show a new file
# only after its directory cache expires, up to a minute)
OWNER_WAIT_TRIES="${TOKENIZATION_OWNER_WAIT_TRIES:-24}"
OWNER_WAIT_SLEEP="${TOKENIZATION_OWNER_WAIT_SLEEP:-5}"
# How often to ask LSF again while its listing still shows this job as
# pending (its query snapshots can lag; 11 x 30 s is beyond LSF's 300 s cap)
SELF_WAIT_TRIES="${TOKENIZATION_SELF_WAIT_TRIES:-11}"
# A takeover marker older than this (minutes) on a lock that was not moved
# means the job that made it died before moving the lock
STUCK_MINUTES="${TOKENIZATION_STUCK_MINUTES:-5}"
mkdir -p "$LOCK_DIR"

# Python environment (with PyYAML): keep in step with the block in
# create_tokenized_dataset.sh
if [ "$USER" == "am84" ]; then
    source /etc/profile.d/modules.sh
    module load cellgen/conda
    conda activate terra-reproducibility
elif [ "$USER" == "sb75" ]; then
    source /nfs/team361/sb75/.venvs/terra/bin/activate
fi

# Settings record of the finished part, from the config (as
# create_tokenized_dataset.py names it)
RECORD=$(python - "$CONFIG_FILE" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
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
print(f"{p['data_directory_path'].rstrip('/')}/gold/{e['tokenizer']}/"
      f"{e['experiment_name']}/{name}.dataset/terra_tokenizer_params.json")
EOF
)
if [ -z "$RECORD" ]; then
    echo "Could not read ${CONFIG_FILE}." >&2
    exit 1
fi
if [ -f "$RECORD" ]; then
    echo "${DATASET} is already tokenized (${RECORD}); job ${JOB} exits."
    exit 0
fi

# job_state <job id>: the job's LSF state (RUN, PEND, DONE, EXIT, ...), or
# ENDED when LSF no longer knows the job (finished jobs are forgotten after
# a while). Returns 1 without output when LSF does not answer.
job_state() {
    local out state try
    for try in 1 2 3 4 5; do
        out=$(bjobs -noheader -o stat "$1" 2>&1)
        # LSF may print notices first ("LSF is processing your request.
        # Please wait ..."): take the first line that is a state
        state=$(printf '%s\n' "$out" | tr -d ' \t\r' \
            | grep -m1 -xE 'PEND|PSUSP|RUN|USUSP|SSUSP|DONE|EXIT|UNKWN|WAIT|ZOMBI|PROV')
        if [ -n "$state" ]; then
            echo "$state"
            return 0
        fi
        case "$out" in
            *"is not found"*)
                echo ENDED
                return 0
                ;;
        esac
        [ "$try" -lt 5 ] && sleep "$RETRY_SLEEP"
    done
    return 1
}

# unlocked_siblings: job ids of this dataset's running jobs that do not use
# this lock (old jobs that run create_tokenized_dataset.sh directly), one
# per line. Inside an LSF job, LSF must list this job as started, so that
# the listing is not older than this job. Returns 1 when LSF does not
# answer, 2 when it keeps listing this job as pending.
unlocked_siblings() {
    local out try pending=""
    for try in $(seq 1 "$SELF_WAIT_TRIES"); do
        out=$(bjobs -noheader -o "jobid stat job_name command delimiter='^'" 2>/dev/null)
        if [ -z "$LSB_JOBID" ] || printf '%s\n' "$out" \
                | awk -F'^' -v me="$SELF" '$1 == me && $2 != "PEND" {found = 1} END {exit !found}'; then
            printf '%s\n' "$out" | awk -F'^' -v me="$SELF" -v p="$JOB_NAME_PREFIX" '
                $1 != me && index($3, p) == 1 && substr($3, length(p) + 1) ~ /^[0-9]+$/ &&
                $2 ~ /^(RUN|PSUSP|USUSP|SSUSP|UNKWN|ZOMBI|PROV|WAIT)$/ &&
                $4 !~ /run_tokenization_locked\.sh/ {print $1}'
            return 0
        fi
        if printf '%s\n' "$out" | awk -F'^' -v me="$SELF" '$1 == me {found = 1} END {exit !found}'; then
            pending=1
        fi
        [ "$try" -lt "$SELF_WAIT_TRIES" ] && sleep "$RETRY_SLEEP"
    done
    [ -n "$pending" ] && return 2
    return 1
}

# inode_of <path>: identifies one lock (each mkdir makes a new folder, and
# replaced locks are moved aside, so their numbers stay taken)
inode_of() {
    ls -di "$1" 2>/dev/null | awk '{print $1}'
}

SIBLINGS=$(unlocked_siblings)
case $? in
    1)
        echo "LSF did not answer; job ${JOB} exits without tokenizing ${DATASET}." >&2
        exit 3
        ;;
    2)
        echo "LSF still lists job ${SELF} as pending; job ${JOB} exits without tokenizing ${DATASET}." >&2
        exit 3
        ;;
esac
if [ -n "$SIBLINGS" ]; then
    echo "${DATASET} is being tokenized by job(s) $(echo $SIBLINGS) without the lock; job ${JOB} exits."
    exit 0
fi

ACQUIRED=""
MKDIR_ERR=""
for round in 1 2 3; do
    if MKDIR_ERR=$(mkdir "$LOCK" 2>&1); then
        echo "$JOB" > "${LOCK}/owner"
        ACQUIRED=1
        break
    fi
    LOCK_ID=$(inode_of "$LOCK")
    if [ -z "$LOCK_ID" ]; then
        # Moved aside by a job taking it over just now: try again
        sleep 1
        continue
    fi
    OWNER=""
    for attempt in $(seq 1 "$OWNER_WAIT_TRIES"); do
        if [ -f "$RECORD" ]; then
            echo "${DATASET} was tokenized by job $(cat "${LOCK}/owner" 2>/dev/null); job ${JOB} exits."
            exit 0
        fi
        OWNER=$(cat "${LOCK}/owner" 2>/dev/null)
        [ -n "$OWNER" ] && break
        # The lock was just made and its owner not yet written (or not yet
        # visible here)
        sleep "$OWNER_WAIT_SLEEP"
    done
    if [ -z "$OWNER" ]; then
        # The job that made the lock ended before writing its id
        STATE="no owner"
    elif [ "$OWNER" == "$JOB" ]; then
        # This job held the lock before LSF requeued it, so its earlier
        # run has ended
        STATE="requeued"
    elif ! STATE=$(job_state "$OWNER"); then
        echo "LSF did not answer about job ${OWNER}, which holds the lock of ${DATASET}; job ${JOB} exits without tokenizing." >&2
        exit 3
    fi
    case "$STATE" in
        RUN|PEND|PSUSP|USUSP|SSUSP|WAIT|PROV|UNKWN|ZOMBI)
            echo "${DATASET} is being tokenized by job ${OWNER} (${STATE}); job ${JOB} exits."
            exit 0
            ;;
    esac
    # It may have finished meanwhile
    if [ -f "$RECORD" ]; then
        echo "${DATASET} was tokenized by job ${OWNER}; job ${JOB} exits."
        exit 0
    fi
    # The owner has ended without finishing: take this lock over. Only the
    # job that creates its takeover marker may (mkdir is atomic); a job
    # arriving meanwhile can still win the lock itself, and then this job
    # exits.
    if ! mkdir "${LOCK}.takeover_i${LOCK_ID}" 2>/dev/null; then
        # The job holding the marker moves the lock within moments: an old
        # marker on a lock that is still in place means it died first
        if [ "$(inode_of "$LOCK")" = "$LOCK_ID" ] && [ -n "$(find "${LOCK}.takeover_i${LOCK_ID}" -maxdepth 0 -mmin +"$STUCK_MINUTES" 2>/dev/null)" ]; then
            echo "The lock of ${DATASET} is stuck: remove ${LOCK}.takeover_i${LOCK_ID} (not the lock itself) to free it; job ${JOB} exits without tokenizing." >&2
            exit 3
        fi
        echo "Another job took over the lock of ${DATASET}; job ${JOB} exits."
        exit 0
    fi
    if [ "$(inode_of "$LOCK")" != "$LOCK_ID" ]; then
        echo "The lock of ${DATASET} was replaced meanwhile; job ${JOB} exits."
        exit 0
    fi
    mv "$LOCK" "${LOCK}.ended_i${LOCK_ID}" 2>/dev/null
    if MKDIR_ERR=$(mkdir "$LOCK" 2>&1); then
        echo "$JOB" > "${LOCK}/owner"
        echo "Job ${OWNER:-unknown} (${STATE}) did not finish ${DATASET}; job ${JOB} takes over."
        ACQUIRED=1
        break
    fi
    if [ ! -d "$LOCK" ]; then
        echo "Could not make the lock of ${DATASET} (${MKDIR_ERR}); job ${JOB} exits without tokenizing." >&2
        exit 3
    fi
    echo "Another job took the lock of ${DATASET}; job ${JOB} exits."
    exit 0
done
if [ -z "$ACQUIRED" ]; then
    echo "Could not take the lock of ${DATASET} (${MKDIR_ERR}); job ${JOB} exits without tokenizing." >&2
    exit 3
fi
# A job without the lock may have finished it after the first check
if [ -f "$RECORD" ]; then
    echo "${DATASET} was tokenized meanwhile (${RECORD}); job ${JOB} exits."
    exit 0
fi

analysis/tokenization/create_tokenized_dataset.sh "$CONFIG_FILE"
RC=$?
if [ $RC -eq 0 ]; then
    touch "${LOCK}/done"
fi
exit $RC
