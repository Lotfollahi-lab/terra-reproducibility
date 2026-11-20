#!/bin/bash
# Job-level configuration for the human niche benchmark, sourced by every .lsf
# script and by job_wrapper.sh. Python-side paths live in bench_utils.py.

# This folder and the repository's result folder (<repo>/artifacts/benchmarking/human,
# same as bench_utils.FINAL_OUTPUT_FOLDER_PATH).
BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FINAL_OUTPUT_DIR=${BENCH_FINAL_OUTPUT_DIR:-$(cd "$BENCH_DIR/../../.." && pwd)/artifacts/benchmarking/human}
# BENCH_FINAL_OUTPUT_DIR (optional) sends results elsewhere, e.g. a separate
# results tree; bench_utils.FINAL_OUTPUT_FOLDER_PATH reads the same variable. It
# must be absolute: job_wrapper.sh and the Python scripts run in different folders.
case $FINAL_OUTPUT_DIR in
    /*) ;;
    *) echo "BENCH_FINAL_OUTPUT_DIR must be an absolute path: $FINAL_OUTPUT_DIR" >&2
       return 1 2>/dev/null || exit 1 ;;
esac

# Job-local working folders on lustre (fast I/O; deleted after a successful copy).
LUSTRE_WORK_ROOT=${BENCH_LUSTRE_ROOT:-/lustre/scratch126/cellgen/lotfollahi/$USER/human_niche_benchmark}

# Python environments. PYTHON_BANKSY is the reference environment: all clustering
# and scoring runs in it (scoring.py checks its package versions). Add one line
# per method environment and pass its name as PYTHON_ENV to run_method.lsf.
PYTHON_BANKSY=/nfs/team361/sb75/.venvs/banksy/bin/python             # scoring; neighbourhood PCA example
PYTHON_CIFM=/nfs/team361/jr43/.venvs/cifm/bin/python                 # CIFM (envs/cifm.sh)
PYTHON_HEIST=/nfs/team361/jr43/.venvs/heist/bin/python               # HEIST (envs/heist.sh)
PYTHON_STOFM=/nfs/team361/jr43/.venvs/stofm/bin/python               # SToFM (envs/stofm.sh)
PYTHON_AIDO_TISSUE=/nfs/team361/jr43/.venvs/aido_tissue/bin/python   # AIDO.Tissue (envs/aido_tissue.sh)

export NUMBA_CPU_NAME=generic   # reproducible UMAP/Leiden (as in the repo notebooks)
