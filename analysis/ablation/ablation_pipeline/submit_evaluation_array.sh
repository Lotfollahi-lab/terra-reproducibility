#!/bin/bash
# Submit the evaluation job array (evaluate_ablation_array.lsf) of a dataset: packages and evaluates, on 1 GPU per
# element, the runs of INDICES that have finished training and whose evaluation is missing or stale (e.g. after a
# change of the evaluation code, the labels or EVAL_FP32), at most MAX_CONCURRENT at a time (default 4, i.e. 4 GPUs).
# Runs that have not finished training are left out (submit_ablation_array.sh trains them); FORCE_EVAL=1 evaluates
# every trained run of INDICES again. The checks, the options and the one-array-at-a-time rule are those of
# submit_ablation_array.sh, which this script runs in its evaluation-only mode.
#
# Usage: bash submit_evaluation_array.sh ENV_FILE [INDICES (default: all rows)] [MAX_CONCURRENT (default: 4)]
# Example: bash submit_evaluation_array.sh datasets/mouse.env 1-4
#          FORCE_EVAL=1 bash submit_evaluation_array.sh datasets/mouse.env 1-51 2

set -euo pipefail

SUBMIT_EVALUATION_ONLY=1 exec bash "$(dirname "${BASH_SOURCE[0]}")/submit_ablation_array.sh" "$@"
