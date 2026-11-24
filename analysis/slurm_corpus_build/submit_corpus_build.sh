#!/bin/bash
# Submit the Slurm corpus build of a set of multi-sample .h5ad files:
#
#   prep      one job per dataset: per-sample silver files (prepare_silver.py)
#   norm      one job after all prep jobs: norm factors of these datasets
#             only (compute_cohort_norm_factors.py; tie_break_mean)
#   tokenize  one job per dataset after norm: tokenized part with the
#             hst_corpus_110m settings (16 x 512, compact) and these norm
#             factors, checked with check_tokenized_part.py
#
# Jobs are chained with afterok dependencies; a failed job cancels the jobs
# that depend on it (--kill-on-invalid-dep=yes).
#
# Usage (from the terra-reproducibility checkout, on a login node):
#   bash analysis/slurm_corpus_build/submit_corpus_build.sh \
#       --config <site_config.sh> --datasets <datasets.tsv> \
#       [--stages prep,norm,tokenize | --stages inspect] [--only <name>[,<name>]] \
#       [--overwrite] [--dry_run]
#
#   --config    site settings (see site_config.example.sh)
#   --datasets  table, one dataset per line, columns separated by tabs or spaces,
#               '#' comments (see datasets.example.tsv):
#               name  dataset_id  h5ad  expected_n_samples|-  exclude_samples|-  sample_key|auto  [prepare_silver.py args]
#   --stages    comma-separated subset of prep,norm,tokenize (default: all), or
#               inspect (structure and detection only, nothing written)
#   --only      prep / tokenize / inspect only these datasets (norm always
#               uses every dataset of the table)
#   --overwrite replace existing silver folders, norm factors and parts
#               (without it, finished steps with the same inputs are kept)
#   --dry_run   print the sbatch commands without submitting
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG="" DATASETS="" STAGES="prep,norm,tokenize" ONLY="" OVERWRITE=0 DRY_RUN=0

usage() { sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
    case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument $1"; usage; exit 1 ;;
    esac
done
[ -n "$CONFIG" ] && [ -n "$DATASETS" ] || { usage; exit 1; }
CONFIG=$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")
DATASETS=$(cd "$(dirname "$DATASETS")" && pwd)/$(basename "$DATASETS")
# shellcheck disable=SC1090
source "$CONFIG"

fail() { echo "ERROR: $*" >&2; exit 1; }
for VAR in CORPUS_NAME TERRA_DATA_DIR TERRA_ENV_ACTIVATE TOKEN_DICTIONARY SLURM_PARTITION; do
    [ -n "${!VAR:-}" ] || fail "$VAR is not set in $CONFIG"
done
[ -f "$TOKEN_DICTIONARY" ] || fail "TOKEN_DICTIONARY $TOKEN_DICTIONARY does not exist"
if [ -n "${GENE_NAME_MAP:-}" ]; then
    [ -f "$GENE_NAME_MAP" ] || fail "GENE_NAME_MAP $GENE_NAME_MAP does not exist"
fi
if [ -n "${HELDOUT_SAMPLES:-}" ]; then
    [ -f "$HELDOUT_SAMPLES" ] || fail "HELDOUT_SAMPLES $HELDOUT_SAMPLES does not exist"
fi
LOG_DIR="${LOG_DIR:-$TERRA_DATA_DIR/logs/$CORPUS_NAME}"

for STAGE in $(echo "$STAGES" | tr ',' ' '); do
    case "$STAGE" in
    prep | norm | tokenize) ;;
    inspect) [ "$STAGES" = inspect ] || fail "--stages inspect runs alone" ;;
    *) fail "unknown stage $STAGE" ;;
    esac
done
has_stage() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
selected() { [ -z "$ONLY" ] && return 0; case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

# Read the dataset table (bash 3 compatible: parallel indexed arrays)
NAMES=() IDS=() H5ADS=() EXPECTED=() EXCLUDES=() KEYS=() EXTRAS=()
# Columns are separated by tabs or spaces (pasting can turn tabs into spaces),
# so paths must not contain spaces; the 7th column takes the rest of the line
while read -r NAME ID H5AD EXP EXCL KEY EXTRA || [ -n "${NAME:-}" ]; do
    case "${NAME:-}" in '' | '#'*) continue ;; esac
    [ -n "${KEY:-}" ] || fail "dataset $NAME: expected 6 columns (name, dataset_id, h5ad, expected_n_samples, exclude_samples, sample_key)"
    echo "$NAME" | grep -Eq '^[A-Za-z0-9._-]+$' || fail "dataset name '$NAME': use letters, digits, '.', '_', '-'"
    echo "$ID" | grep -Eq '^[0-9]+$' || fail "dataset $NAME: dataset_id '$ID' is not an integer"
    [ -r "$H5AD" ] || fail "dataset $NAME: cannot read $H5AD"
    for OTHER in ${NAMES[@]+"${NAMES[@]}"}; do [ "$OTHER" != "$NAME" ] || fail "dataset name $NAME twice"; done
    for OTHER in ${IDS[@]+"${IDS[@]}"}; do [ "$OTHER" != "$ID" ] || fail "dataset_id $ID twice"; done
    NAMES+=("$NAME") IDS+=("$ID") H5ADS+=("$H5AD") EXPECTED+=("$EXP")
    EXCLUDES+=("$EXCL") KEYS+=("$KEY") EXTRAS+=("${EXTRA:-}")
done <"$DATASETS"
[ ${#NAMES[@]} -gt 0 ] || fail "no dataset in $DATASETS"
if [ -n "$ONLY" ]; then
    for WANTED in $(echo "$ONLY" | tr ',' ' '); do
        FOUND=0
        for NAME in "${NAMES[@]}"; do [ "$NAME" = "$WANTED" ] && FOUND=1; done
        [ $FOUND = 1 ] || fail "--only $WANTED is not in $DATASETS"
    done
fi

echo "Corpus ${CORPUS_NAME}: ${#NAMES[@]} datasets (${NAMES[*]}); stages ${STAGES}"
echo "Data root ${TERRA_DATA_DIR}; logs ${LOG_DIR}"
echo "terra-reproducibility $(git -C "$REPO_DIR" log --oneline -1 2>/dev/null || echo '(no git)')"

if [ $DRY_RUN = 0 ]; then
    mkdir -p "$LOG_DIR" "$TERRA_DATA_DIR/tokenizer/$CORPUS_NAME" "$TERRA_DATA_DIR/silver"
    NAMES_FILE="$TERRA_DATA_DIR/tokenizer/$CORPUS_NAME/${CORPUS_NAME}_dataset_names.txt"
    printf '%s\n' "${NAMES[@]}" >"$NAMES_FILE.tmp"
    if [ -f "$NAMES_FILE" ] && ! cmp -s "$NAMES_FILE" "$NAMES_FILE.tmp"; then
        echo "NOTE: the datasets of the corpus changed:"
        diff "$NAMES_FILE" "$NAMES_FILE.tmp" || true
    fi
    mv "$NAMES_FILE.tmp" "$NAMES_FILE"
fi

read -r -a EXTRA_SBATCH <<<"${SLURM_EXTRA_ARGS:-}"
DRY_ID=0
# submit <job name> <cpus> <mem> <time> <dependency or ''> <corpus_job.sh args...>
submit() {
    local JOB_NAME="$1" CPUS="$2" MEM="$3" TIME="$4" DEP="$5"
    shift 5
    local CMD=(sbatch --parsable --partition "$SLURM_PARTITION"
        --job-name "$JOB_NAME" --cpus-per-task "$CPUS" --mem "$MEM"
        --time "$TIME" --output "$LOG_DIR/%x_%j.out"
        --export "ALL,SITE_CONFIG=$CONFIG,REPO_DIR=$REPO_DIR,OVERWRITE=$OVERWRITE")
    if [ -n "$DEP" ]; then
        CMD+=(--dependency "afterok:$DEP" --kill-on-invalid-dep=yes)
    fi
    CMD+=(${EXTRA_SBATCH[@]+"${EXTRA_SBATCH[@]}"} "$SCRIPT_DIR/corpus_job.sh" "$@")
    if [ $DRY_RUN = 1 ]; then
        DRY_ID=$((DRY_ID + 1))
        echo "  [dry run] ${CMD[*]}" >&2
        JOB_ID="dry$DRY_ID"
    else
        JOB_ID=$("${CMD[@]}")
        JOB_ID="${JOB_ID%%;*}"
    fi
    echo "  submitted $JOB_NAME: job $JOB_ID${DEP:+ (after $DEP)}" >&2
}

if [ "$STAGES" = inspect ]; then
    for i in "${!NAMES[@]}"; do
        selected "${NAMES[$i]}" || continue
        read -r -a EXTRA_ARGS <<<"${EXTRAS[$i]}"
        submit "corpus_inspect_${NAMES[$i]}" "${INSPECT_CPUS:-2}" "${INSPECT_MEM:-64G}" "${INSPECT_TIME:-02:00:00}" "" \
            inspect "${NAMES[$i]}" "${IDS[$i]}" "${H5ADS[$i]}" "${EXPECTED[$i]}" "${EXCLUDES[$i]}" "${KEYS[$i]}" \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    done
    echo "Logs: $LOG_DIR/corpus_inspect_<name>_<job id>.out"
    exit 0
fi

PREP_IDS=""
if has_stage prep; then
    for i in "${!NAMES[@]}"; do
        selected "${NAMES[$i]}" || continue
        read -r -a EXTRA_ARGS <<<"${EXTRAS[$i]}"
        submit "corpus_prep_${NAMES[$i]}" "${PREP_CPUS:-4}" "${PREP_MEM:-256G}" "${PREP_TIME:-12:00:00}" "" \
            prep "${NAMES[$i]}" "${IDS[$i]}" "${H5ADS[$i]}" "${EXPECTED[$i]}" "${EXCLUDES[$i]}" "${KEYS[$i]}" \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
        PREP_IDS="${PREP_IDS:+$PREP_IDS:}$JOB_ID"
    done
fi

NORM_ID=""
if has_stage norm; then
    submit "corpus_norm_${CORPUS_NAME}" "${NORM_CPUS:-2}" "${NORM_MEM:-128G}" "${NORM_TIME:-24:00:00}" "$PREP_IDS" norm
    NORM_ID="$JOB_ID"
fi

if has_stage tokenize; then
    for i in "${!NAMES[@]}"; do
        selected "${NAMES[$i]}" || continue
        DEP="$NORM_ID"
        if [ -z "$DEP" ] && has_stage prep; then
            DEP="$PREP_IDS"
        fi
        submit "corpus_tokenize_${NAMES[$i]}" "${TOKENIZE_CPUS:-16}" "${TOKENIZE_MEM:-256G}" "${TOKENIZE_TIME:-48:00:00}" "$DEP" \
            tokenize "${NAMES[$i]}" "${IDS[$i]}"
    done
fi
echo "Logs: $LOG_DIR/<job name>_<job id>.out; follow with: squeue -u \$USER"
