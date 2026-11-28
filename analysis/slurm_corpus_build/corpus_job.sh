#!/bin/bash
# One stage of the Slurm corpus build. submit_corpus_build.sh submits it with
# SITE_CONFIG and REPO_DIR in the environment; do not sbatch it by hand.
#
#   corpus_job.sh inspect|prep <name> <dataset_id> <h5ad> <expected_n_samples|-> <exclude_samples|-> <sample_key> [prepare_silver.py args...]
#   corpus_job.sh norm
#   corpus_job.sh tokenize <name> <dataset_id>
#   corpus_job.sh concat
#   corpus_job.sh split
#
# Stages:
#   inspect   prepare_silver.py --inspect: file structure and detection only
#   prep      prepare_silver.py: per-sample silver files of one dataset
#   norm      compute_cohort_norm_factors.py over all datasets of the corpus
#             (no hst_corpus_110m held-out ids; HELDOUT_SAMPLES, if set,
#             are left out), without PFlog1pPF targets (shifted_log)
#   tokenize  create_tokenized_dataset.py with the hst_corpus_110m settings
#             and the corpus' own norm factors, then check_tokenized_part.py
#   concat    concatenate_datasets.py: the parts of every dataset into one
#             corpus (kept when it fits the parts)
#   split     make_splits.py: training / validation / test splits of the
#             corpus per variant (SPLITS), held-out samples from SPLIT_SAMPLES
#             (none if unset)
set -euo pipefail

STAGE="${1:?stage}"
shift
: "${SITE_CONFIG:?SITE_CONFIG is not set (submit with submit_corpus_build.sh)}"
: "${REPO_DIR:?REPO_DIR is not set (submit with submit_corpus_build.sh)}"
# shellcheck disable=SC1090
source "$SITE_CONFIG"
TOOLS="$REPO_DIR/analysis/slurm_corpus_build"
OVERWRITE="${OVERWRITE:-0}"

echo "== $(date '+%F %T') stage ${STAGE} on $(hostname); Slurm job ${SLURM_JOB_ID:-none}, ${SLURM_CPUS_PER_TASK:-?} CPUs, ${SLURM_MEM_PER_NODE:-?} MB"
echo "== terra-reproducibility: $(git -C "$REPO_DIR" log --oneline -1 2>/dev/null || echo 'not a git checkout') ($(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?'))"

# Activation scripts (conda in particular) may read unset variables
set +u
eval "$TERRA_ENV_ACTIVATE"
set -u
export HDF5_USE_FILE_LOCKING=FALSE
# One thread per process: the tokenizer runs one process per input file, and
# torch / BLAS threads per process oversubscribe many-core nodes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

python - "$STAGE" <<'EOF'
import importlib
import subprocess
import sys
from pathlib import Path

stage = sys.argv[1]
problems = []
try:
    import terra
    src = Path(terra.__file__).resolve().parent
    def git(*args):
        try:
            return subprocess.run(['git', '-C', str(src), *args],
                                  capture_output=True, text=True).stdout.strip()
        except OSError:
            return ''
    print(f"== TERRA {getattr(terra, '__version__', '?')} {terra.__file__} "
          f"{git('rev-parse', '--short', 'HEAD') or '(no git)'} "
          f"{git('rev-parse', '--abbrev-ref', 'HEAD')}")
except Exception as error:
    message = f'TERRA cannot be imported ({type(error).__name__}: {error})'
    if stage in ('tokenize', 'concat', 'split'):
        problems.append(message)
    else:
        print(f'== WARNING: {message}; this stage does not need it')
versions = []
for name in ('python', 'anndata', 'scanpy', 'squidpy', 'datasets', 'numpy',
             'scipy', 'pandas', 'h5py', 'pyarrow'):
    if name == 'python':
        versions.append(f"python {sys.version.split()[0]}")
        continue
    try:
        versions.append(f"{name} {importlib.import_module(name).__version__}")
    except Exception as error:
        versions.append(f"{name} MISSING ({type(error).__name__})")
print('== ' + ', '.join(versions))
if stage in ('tokenize', 'concat', 'split') and not problems:
    try:
        importlib.import_module('terra.datasets.compact')
    except ImportError:
        problems.append('this TERRA cannot write compact rows: install the '
                        'TERRA branch feat/compact-tokenized-data')
if stage == 'tokenize' and not problems:
    try:
        importlib.import_module('wandb')
    except ImportError:
        problems.append("wandb is missing (create_tokenized_dataset.py "
                        "imports it): pip install 'wandb>=0.16'")
    try:
        importlib.import_module('squidpy')
    except ImportError:
        problems.append('squidpy is missing (k-NN graph)')
if problems:
    sys.exit('ERROR: ' + '; '.join(problems))
EOF

NAMES_FILE="$TERRA_DATA_DIR/tokenizer/$CORPUS_NAME/${CORPUS_NAME}_dataset_names.txt"
if [ -n "${HELDOUT_SAMPLES:-}" ]; then
    NF_PREFIX="${CORPUS_NAME}_train"
else
    NF_PREFIX="$CORPUS_NAME"
fi
NF="$TERRA_DATA_DIR/tokenizer/$CORPUS_NAME/${NF_PREFIX}_norm_factors.csv"

case "$STAGE" in
inspect | prep)
    NAME="$1" DATASET_ID="$2" H5AD="$3" EXPECTED="$4" EXCLUDE="$5" SAMPLE_KEY="$6"
    shift 6
    ARGS=(--h5ad "$H5AD" --dataset_name "$NAME" --dataset_id "$DATASET_ID"
        --data_directory_path "$TERRA_DATA_DIR"
        --token_dictionary_file_path "$TOKEN_DICTIONARY"
        --sample_key "$SAMPLE_KEY")
    if [ -n "${GENE_NAME_MAP:-}" ]; then
        ARGS+=(--gene_name_map "$GENE_NAME_MAP")
    fi
    if [ "$EXPECTED" != "-" ]; then
        ARGS+=(--expected_n_samples "$EXPECTED")
    fi
    if [ "$EXCLUDE" != "-" ]; then
        IFS=',' read -r -a EXCLUDED <<<"$EXCLUDE"
        ARGS+=(--exclude_samples "${EXCLUDED[@]}")
    fi
    if [ "$STAGE" = inspect ]; then
        ARGS+=(--inspect)
    fi
    if [ "$OVERWRITE" = 1 ]; then
        ARGS+=(--overwrite)
    fi
    python "$TOOLS/prepare_silver.py" "${ARGS[@]}" ${1+"$@"}
    ;;

norm)
    [ -f "$NAMES_FILE" ] || { echo "ERROR: no dataset names file $NAMES_FILE"; exit 1; }
    echo "== Datasets of corpus ${CORPUS_NAME}:"
    cat "$NAMES_FILE"
    while read -r NAME; do
        [ -f "$TERRA_DATA_DIR/silver/$NAME/silver_manifest.tsv" ] || {
            echo "ERROR: silver/$NAME is not prepared (no silver_manifest.tsv)"
            exit 1
        }
    done <"$NAMES_FILE"
    ARGS=(--data_directory_path "$TERRA_DATA_DIR"
        --cohort_name "$CORPUS_NAME"
        --artifacts_base_path "$TERRA_DATA_DIR/tokenizer"
        --output_prefix "$NF_PREFIX"
        --cohort_dataset_names_file_path "$NAMES_FILE"
        --token_dictionary_file_path "$TOKEN_DICTIONARY"
        --no-exclude_heldout_batches
        --skip_pf_targets)
    HELDOUT_ARRAY=()
    if [ -n "${HELDOUT_SAMPLES:-}" ]; then
        HELDOUT_IDS=$(python "$TOOLS/corpus_tools.py" heldout-ids \
            --heldout_samples "$HELDOUT_SAMPLES" \
            --data_directory_path "$TERRA_DATA_DIR")
        read -r -a HELDOUT_ARRAY <<<"$HELDOUT_IDS"
        echo "== Left out of the norm factors: ${HELDOUT_ARRAY[*]}"
        ARGS+=(--exclude_batch_ids "${HELDOUT_ARRAY[@]}")
    fi
    STATUS=$(python "$TOOLS/corpus_tools.py" norm-status \
        --norm_factor_csv "$NF" --names_file "$NAMES_FILE" \
        --data_directory_path "$TERRA_DATA_DIR" \
        --token_dictionary_file_path "$TOKEN_DICTIONARY" \
        --excluded_batch_ids ${HELDOUT_ARRAY[@]+"${HELDOUT_ARRAY[@]}"})
    echo "== Existing norm factors: $STATUS"
    if [ "$STATUS" = current ] && [ "$OVERWRITE" != 1 ]; then
        echo "== Keeping $NF (same silver files; --overwrite recomputes it)"
    else
        if [ "$STATUS" != absent ] && [ "$OVERWRITE" != 1 ]; then
            echo "ERROR: $NF exists but does not fit the silver files ($STATUS). Resubmit with --overwrite (the tokenized parts are then redone too)."
            exit 1
        fi
        if [ "$STATUS" != absent ]; then
            ARGS+=(--overwrite)
        fi
        python "$REPO_DIR/analysis/data_preparation/compute_cohort_norm_factors.py" "${ARGS[@]}"
    fi
    echo "== Norm factors: $NF"
    cat "${NF%.csv}.json"
    echo
    ;;

tokenize)
    NAME="$1" DATASET_ID="$2"
    [ -f "$NF" ] || { echo "ERROR: no norm factors $NF (run the norm stage first)"; exit 1; }
    [ -f "$TERRA_DATA_DIR/silver/$NAME/silver_manifest.tsv" ] || {
        echo "ERROR: silver/$NAME is not prepared"
        exit 1
    }
    WORK="$TERRA_DATA_DIR/work/$CORPUS_NAME"
    CONFIG="$WORK/configs/cell-graph-tokenizer_${NAME}.yaml"
    PART=$(python "$TOOLS/corpus_tools.py" tokenizer-config \
        --dataset_name "$NAME" --dataset_id "$DATASET_ID" \
        --corpus_name "$CORPUS_NAME" \
        --data_directory_path "$TERRA_DATA_DIR" \
        --token_dictionary_file_path "$TOKEN_DICTIONARY" \
        --norm_factor_csv "$NF" --out "$CONFIG")
    echo "== Tokenizer config $CONFIG:"
    cat "$CONFIG"
    STATUS=$(python "$TOOLS/corpus_tools.py" part-status --part "$PART" \
        --norm_factor_csv "$NF" --token_dictionary_file_path "$TOKEN_DICTIONARY")
    echo "== Part $PART: $STATUS"
    if [ "$STATUS" = done ] && [ "$OVERWRITE" = 1 ]; then
        STATUS=stale
    fi
    if [ "$STATUS" = incomplete ] || [ "$STATUS" = stale ]; then
        echo "== Removing the $STATUS part"
        rm -rf "$PART"
    fi
    if [ "$STATUS" != done ]; then
        NPROC="${TOKENIZE_NPROC:-}"
        if [ -z "$NPROC" ]; then
            NPROC=$(python "$TOOLS/corpus_tools.py" nproc --dataset_name "$NAME" \
                --data_directory_path "$TERRA_DATA_DIR" \
                --cpus "${SLURM_CPUS_PER_TASK:-1}")
        fi
        # create_tokenized_dataset.py takes its process count from this LSF
        # variable (the farm's scheduler)
        export LSB_DJOB_NUMPROC="$NPROC"
        echo "== Tokenizing with $NPROC processes"
        mkdir -p "$WORK"
        cd "$WORK" # create_tokenized_dataset.py writes ./logs/<name>/cProfiler
        python "$REPO_DIR/analysis/tokenization/create_tokenized_dataset.py" \
            --config_file "$CONFIG"
    fi
    echo "== Checking $PART"
    python "$REPO_DIR/analysis/data_preparation/checks/check_tokenized_part.py" \
        --dataset_path "$PART" \
        --token_dictionary_file_path "$TOKEN_DICTIONARY" \
        --norm_factor_csv "$NF" \
        --silver_dir "$TERRA_DATA_DIR/silver/$NAME"
    ;;

concat | split)
    [ -f "$NF" ] || { echo "ERROR: no norm factors $NF"; exit 1; }
    # Fails unless every part is tokenized with the current norm factors
    corpus_status() {
        python "$TOOLS/corpus_tools.py" corpus-status \
            --names_file "$NAMES_FILE" --data_directory_path "$TERRA_DATA_DIR" \
            --corpus_name "$CORPUS_NAME" --norm_factor_csv "$NF" \
            --token_dictionary_file_path "$TOKEN_DICTIONARY"
    }
    # (an assignment, so that set -e stops the job when the check fails)
    STATUS_LINE=$(corpus_status)
    read -r STATUS CORPUS <<<"$STATUS_LINE"
    echo "== Corpus $CORPUS: $STATUS"
    if [ "$STAGE" = concat ]; then
        if [ "$STATUS" = current ] && [ "$OVERWRITE" != 1 ]; then
            echo "== Keeping the corpus (it fits the parts; --overwrite rebuilds it)"
        else
            if [ "$STATUS" = outdated ] && [ "$OVERWRITE" != 1 ]; then
                echo "ERROR: the corpus does not fit the parts. Resubmit with --overwrite (the splits are then redone too)."
                exit 1
            fi
            ARGS=(--data_directory_path "$TERRA_DATA_DIR"
                --cohort_name "$CORPUS_NAME"
                --cohort_dataset_names_file_path "$NAMES_FILE"
                --tokenizer cell-graph-tokenizer --num_shards 32
                --count_count_norm_method shifted_log
                --n_neighs 16 --seq_len 8704 --expect_storage_format compact
                --num_proc "${SLURM_CPUS_PER_TASK:-1}")
            if [ "$STATUS" != absent ]; then
                ARGS+=(--overwrite)
            fi
            # Names a leftover <corpus>.partial-<id> folder by the Slurm job
            export LSB_JOBID="${SLURM_JOB_ID:-$$}"
            python "$REPO_DIR/analysis/tokenization/concatenate_datasets.py" "${ARGS[@]}"
            STATUS_LINE=$(corpus_status)
            read -r STATUS CORPUS <<<"$STATUS_LINE"
            echo "== Corpus $CORPUS: $STATUS"
            [ "$STATUS" = current ] || { echo "ERROR: the new corpus does not fit the parts"; exit 1; }
        fi
    else
        [ "$STATUS" = current ] || { echo "ERROR: the corpus is $STATUS: run the concat stage first"; exit 1; }
        ARGS=(--corpus_path "$CORPUS" --data_directory_path "$TERRA_DATA_DIR"
            --names_file "$NAMES_FILE"
            --splits "${SPLITS:-}"
            --out_dir "$TERRA_DATA_DIR/tokenizer/$CORPUS_NAME/splits"
            --norm_factor_csv "$NF" --token_dictionary_file_path "$TOKEN_DICTIONARY")
        if [ -n "${SPLIT_SAMPLES:-}" ]; then
            ARGS+=(--split_samples "$SPLIT_SAMPLES")
        fi
        if [ "$OVERWRITE" = 1 ]; then
            ARGS+=(--overwrite)
        fi
        python "$TOOLS/make_splits.py" "${ARGS[@]}"
    fi
    ;;

*)
    echo "ERROR: unknown stage $STAGE"
    exit 1
    ;;
esac
echo "== $(date '+%F %T') stage ${STAGE} finished"
