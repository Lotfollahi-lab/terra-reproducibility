#!/bin/bash
# Run one benchmark step in a job-local lustre working folder, score what it
# produced, then copy the results to the permanent NFS folder and delete the
# lustre folder.
#
#   job_wrapper.sh <python> <script.py> [args...]
#
# - Inputs are read from lustre (SILVER_FOLDER_PATH in bench_utils.py).
# - Results and caches are written to $WORK on lustre (fast I/O).
# - Method scripts only write embeddings (*_embedded.h5ad); every embedding is
#   then clustered and scored by score.py --pending in the reference (banksy)
#   environment, in the same job. Scoring never runs in a method environment.
# - On success: rsync $WORK/results -> $FINAL_OUTPUT_DIR (NFS, backed up),
#   verify the copy with a checksum dry-run, then remove $WORK.
# - On failure: nothing is copied and $WORK is kept for debugging.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

PYTHON=$1; shift
SCRIPT=$1; shift

WORK=$LUSTRE_WORK_ROOT/jobs/${LSB_JOBID:-local}_${LSB_JOBINDEX:-0}
mkdir -p "$WORK/results" "$WORK/cache" "$FINAL_OUTPUT_DIR"
export BENCH_OUTPUT_DIR=$WORK/results
export BENCH_CACHE_DIR=$WORK/cache
echo "Working folder (lustre): $WORK"

cd "$BENCH_DIR"
if ! /usr/bin/time -v "$PYTHON" -u "$SCRIPT" "$@"; then
    echo "FAILED: results NOT copied; working folder kept at $WORK" >&2
    exit 1
fi

# Score every embedding the step wrote (reference environment, one thread per
# resolution-scan worker).
if [ -n "$(find "$WORK/results" -name '*_embedded.h5ad' -print -quit)" ]; then
    echo "Scoring embeddings with score.py ($PYTHON_BANKSY)"
    if ! OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            /usr/bin/time -v "$PYTHON_BANKSY" -u score.py --pending; then
        echo "SCORING FAILED: results NOT copied; working folder kept at $WORK" >&2
        exit 1
    fi
fi
if [ -n "$(find "$WORK/results" -name '*_embedded.h5ad' -print -quit)" ]; then
    echo "Unscored embeddings left in $WORK/results; results NOT copied" >&2
    exit 1
fi

echo "Copying results to $FINAL_OUTPUT_DIR ..."
rsync -a "$WORK/results/" "$FINAL_OUTPUT_DIR/"
# Verify: a checksum dry-run must report no files left to transfer.
if [ -n "$(rsync -a --checksum --dry-run --out-format='%n' "$WORK/results/" "$FINAL_OUTPUT_DIR/" | grep -v '/$' || true)" ]; then
    echo "COPY VERIFICATION FAILED: working folder kept at $WORK" >&2
    exit 1
fi
echo "Copy verified. Removing $WORK"
rm -rf "$WORK"
