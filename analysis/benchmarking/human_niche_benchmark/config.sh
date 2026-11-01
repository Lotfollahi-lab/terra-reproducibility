#!/bin/bash
# Job-level configuration for the human niche benchmark, sourced by every .lsf
# script and by job_wrapper.sh. Python-side paths live in bench_utils.py.

# This folder and the repository's result folder (<repo>/artifacts/benchmarking/human,
# same as bench_utils.FINAL_OUTPUT_FOLDER_PATH).
BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FINAL_OUTPUT_DIR=$(cd "$BENCH_DIR/../../.." && pwd)/artifacts/benchmarking/human

# Job-local working folders on lustre (fast I/O; deleted after a successful copy).
LUSTRE_WORK_ROOT=${BENCH_LUSTRE_ROOT:-/lustre/scratch126/cellgen/lotfollahi/$USER/human_niche_benchmark}

# Python environments (existing environments, used read-only).
PYTHON_BANKSY=/nfs/team361/sb75/.venvs/banksy/bin/python             # BANKSY, neighbourhood PCA, re-scoring
PYTHON_CELLCHARTER=/nfs/team361/sb75/.venvs/cellcharter/bin/python
PYTHON_NICHECOMPASS=/nfs/team361/sb75/.venvs/nichecompass/bin/python
PYTHON_NICHEFORMER=/nfs/team361/sb75/.venvs/nicheformer/bin/python
PYTHON_TERRA=/nfs/team361/sb75/.venvs/terra/bin/python

export NUMBA_CPU_NAME=generic   # reproducible UMAP/Leiden (as in the repo notebooks)
