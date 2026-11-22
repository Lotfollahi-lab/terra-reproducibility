#!/bin/bash
# Run create_tokenized_dataset.sh <config> under a per-dataset lock, so that
# several jobs for the same dataset (e.g. one on hugemem and one on
# cpu-interactive) never tokenize it at the same time:
# - the dataset is already tokenized (its settings record exists, or a
#   job finished it under the lock): exit 0;
# - another job holds the lock and is still running: exit 0;
# - the job holding the lock has ended without finishing (TERM_MEMLIMIT,
#   TERM_RUNLIMIT, killed): take the lock over and tokenize.
# Locks: <terra-reproducibility root>/logs/tokenization_locks/<dataset>.lock
# (mkdir is atomic, so only one job wins).
#
# Usage (as the command of an LSF job started in the terra-reproducibility
# root, e.g. by submit_tokenization_twin.sh, or set on a pending job with
# bmod -Z):
#   analysis/tokenization/run_tokenization_locked.sh config/cell-graph-tokenizer/cell-graph-tokenizer_<dataset>.yaml
CONFIG_FILE="${1:?usage: run_tokenization_locked.sh <config file>}"
DATASET=$(basename "$CONFIG_FILE" .yaml)
DATASET=${DATASET#cell-graph-tokenizer_}
LOCK_DIR="${TOKENIZATION_LOCK_DIR:-logs/tokenization_locks}"
LOCK="${LOCK_DIR}/${DATASET}.lock"
JOB="${LSB_JOBID:-$$}"
mkdir -p "$LOCK_DIR"

if [ "$USER" == "sb75" ]; then
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

if mkdir "$LOCK" 2>/dev/null; then
    echo "$JOB" > "${LOCK}/owner"
else
    for attempt in 1 2 3; do
        if [ -f "${LOCK}/done" ] || [ -f "$RECORD" ]; then
            echo "${DATASET} was tokenized by job $(cat "${LOCK}/owner" 2>/dev/null); job ${JOB} exits."
            exit 0
        fi
        OWNER=$(cat "${LOCK}/owner" 2>/dev/null)
        [ -n "$OWNER" ] && break
        # the lock was just made and its owner not yet written
        sleep 5
    done
    STATE=$(bjobs -noheader -o stat "$OWNER" 2>/dev/null | head -1)
    case "$STATE" in
        RUN|PEND|PSUSP|USUSP|SSUSP|WAIT|PROV)
            echo "${DATASET} is being tokenized by job ${OWNER} (${STATE}); job ${JOB} exits."
            exit 0
            ;;
    esac
    # It may have finished meanwhile
    if [ -f "${LOCK}/done" ] || [ -f "$RECORD" ]; then
        echo "${DATASET} was tokenized by job ${OWNER}; job ${JOB} exits."
        exit 0
    fi
    # The owner has ended without finishing: take the lock over. Only the
    # job that creates the takeover marker for this owner may (mkdir is
    # atomic); a job arriving meanwhile can still win the lock itself, and
    # then this job exits.
    if ! mkdir "${LOCK}.takeover_${OWNER:-unknown}" 2>/dev/null; then
        echo "Another job took over the lock of ${DATASET}; job ${JOB} exits."
        exit 0
    fi
    mv "$LOCK" "${LOCK}.ended_${OWNER:-unknown}" 2>/dev/null
    if ! mkdir "$LOCK" 2>/dev/null; then
        echo "Another job took the lock of ${DATASET}; job ${JOB} exits."
        exit 0
    fi
    echo "$JOB" > "${LOCK}/owner"
    echo "Job ${OWNER:-unknown} (${STATE:-ended}) did not finish ${DATASET}; job ${JOB} takes over."
fi

analysis/tokenization/create_tokenized_dataset.sh "$CONFIG_FILE"
RC=$?
if [ $RC -eq 0 ]; then
    touch "${LOCK}/done"
fi
exit $RC
