#!/bin/bash
# Benchmark of the inference-efficiency prototypes on one GPU node (see
# README.md). Needs setup_variants.sh to have run (source trees + bundles).
#
#   H5AD=/path/to/sample.h5ad bash run_efficiency.sh
#
# Optional: WORK, MODELS, N_CELLS (20000), SAMPLE_KEY + SAMPLE, BATCH_SIZE
# (128), NUM_WORKERS (12), REPS (2), TOK_NPROCS ("1 4"), SKIP_TOKENIZATION=1,
# EMB_VARIANTS and TOK_VARIANTS (the variants to run, base first; default:
# all), ALLOW_CPU=1 (only for a CPU smoke test of the scripts).
#
# Results go to WORK/results/<LSF job ID>. A failing variant is recorded in
# steps.tsv and reported, and the other variants still run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$HERE/work}"
WORK="$(cd "$WORK" && pwd)"
H5AD="${H5AD:?set H5AD to a raw-count spatial AnnData (obsm['spatial'])}"
MODELS="${MODELS:-TERRA-112M TERRA-96M}"
N_CELLS="${N_CELLS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-12}"
REPS="${REPS:-2}"
TOK_NPROCS="${TOK_NPROCS:-1 4}"
EMB_VARIANTS="${EMB_VARIANTS:-base base_rerun p2 p1 p12 base_pin p12_pin base_nw0 final}"
TOK_VARIANTS="${TOK_VARIANTS:-base tok tok_gen tok_gen_fp final}"
RESULTS="$WORK/results/${LSB_JOBID:-local-$(date +%Y%m%d-%H%M%S)}"
if [ -n "${SAMPLE_KEY:-}" ] && [ -z "${SAMPLE:-}" ]; then
    echo "SAMPLE_KEY is set: set SAMPLE to the section to use as well." >&2
    exit 1
fi

# The source tree of each variant
emb_tree() {
    case "$1" in
        base|base_rerun|base_pin|base_nw0) echo base ;;
        p12_pin) echo p12 ;;
        p1|p2|p12|final) echo "$1" ;;
        *) return 1 ;;
    esac
}
tok_tree() {
    case "$1" in
        base|final) echo "$1" ;;
        tok|tok_gen|tok_gen_fp) echo tok ;;
        *) return 1 ;;
    esac
}
# Stop before any work if a variant is unknown or its tree is missing
for label in $EMB_VARIANTS; do
    tree="$(emb_tree "$label")" \
        || { echo "Unknown embedding variant: $label" >&2; exit 1; }
    [ -d "$WORK/src/$tree/src" ] \
        || { echo "No tree $WORK/src/$tree: run setup_variants.sh." >&2; exit 1; }
done
for label in $TOK_VARIANTS; do
    tree="$(tok_tree "$label")" \
        || { echo "Unknown tokenization variant: $label" >&2; exit 1; }
    [ -d "$WORK/src/$tree/src" ] \
        || { echo "No tree $WORK/src/$tree: run setup_variants.sh." >&2; exit 1; }
done
# The same number of CPU threads for every variant (the timed loop includes
# host-side work): the job's slots
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${LSB_DJOB_NUMPROC:-8}}"
mkdir -p "$RESULTS"
echo "Results: $RESULTS"

# Run python with the TERRA source tree of the given variant
py() {
    local variant="$1"
    shift
    PYTHONPATH="$WORK/src/$variant/src" TERRA_TREE="$WORK/src/$variant/src" \
        python "$@"
}

# Run one step; a failure is recorded in steps.tsv and the job goes on
n_failed=0
last_status=0
step() {
    local name="$1"
    shift
    last_status=0
    "$@" || last_status=$?
    if [ "$last_status" = 0 ]; then
        printf '%s\tok\n' "$name" >> "$RESULTS/steps.tsv"
    else
        printf '%s\tFAILED (exit %s)\n' "$name" "$last_status" \
            >> "$RESULTS/steps.tsv"
        echo "FAILED: $name (exit $last_status)" >&2
        n_failed=$((n_failed + 1))
    fi
}

# Embedding of $data/dataset with one variant
emb() {
    local variant="$1" label="$2"
    shift 2
    echo "== $model embedding: $label"
    py "$variant" "$HERE/bench_embed.py" --bundle "$bundle" \
        --dataset "$data/dataset" --out "$RESULTS/emb__${model}__${label}" \
        --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
        --reps "$REPS" ${ALLOW_CPU:+--allow-cpu} "$@"
}

# Tokenization of $data/harmonized.h5ad with one variant
tok() {
    local variant="$1" label="$2" nproc="$3"
    shift 3
    echo "== $model tokenization, nproc $nproc: $label"
    py "$variant" "$HERE/bench_tokenize.py" run \
        --harmonized "$data/harmonized.h5ad" --bundle "$bundle" \
        --out "$RESULTS/tok__${model}__np${nproc}__${label}" \
        --nproc "$nproc" --reps "$REPS" "$@"
}

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
if [ -z "${ALLOW_CPU:-}" ]; then
    python -c 'import sys, torch; print("torch", torch.__version__, "CUDA", torch.version.cuda, "available:", torch.cuda.is_available()); sys.exit(not torch.cuda.is_available())' \
        || { echo "torch sees no CUDA device: stopping." >&2; exit 1; }
fi
# Random dataset fingerprints (tok_gen_fp) need datasets >= 4.3
if python -c 'import inspect, sys, datasets; sys.exit("fingerprint" not in inspect.signature(datasets.Dataset.from_generator).parameters)'; then
    fingerprint_ok=1
else
    fingerprint_ok=0
    echo "datasets has no Dataset.from_generator(fingerprint=...):" \
         "tok_gen_fp is skipped."
fi

for model in $MODELS; do
    bundle="$WORK/bundles/$model"
    data="$WORK/data/$model"
    if [ ! -f "$bundle/model_checkpoint.pt" ]; then
        echo "Missing bundle $bundle: run setup_variants.sh on a login node." >&2
        exit 1
    fi

    # Common input: harmonized AnnData + dataset tokenized with the base tree
    # (prepared anew whenever H5AD, SAMPLE_KEY, SAMPLE or N_CELLS changed)
    step "prepare $model" py base "$HERE/prepare_data.py" --h5ad "$H5AD" \
        --bundle "$bundle" --out "$data" --n-cells "$N_CELLS" \
        ${SAMPLE_KEY:+--sample-key "$SAMPLE_KEY" --sample "$SAMPLE"}
    if [ "$last_status" != 0 ]; then
        continue
    fi
    cp "$data/prepare.json" "$RESULTS/prepare__${model}.json"
    # Read the dataset once, so that no variant pays for a cold page cache
    cat "$data"/dataset/*.arrow > /dev/null || true

    # Embedding: every variant embeds the same tokenized dataset.
    # base_rerun is the determinism control.
    for label in $EMB_VARIANTS; do
        case "$label" in
            base_pin|p12_pin) extra=(--pin-memory) ;;
            base_nw0) extra=(--num-workers 0) ;;
            *) extra=() ;;
        esac
        step "emb $model $label" emb "$(emb_tree "$label")" "$label" \
            ${extra[@]+"${extra[@]}"}
    done

    # Tokenization: base; prototype; prototype formatting the examples in
    # the generator; the same with a random dataset fingerprint; the final
    # change. Each dataset is compared with the base dataset of the same
    # nproc.
    if [ "${SKIP_TOKENIZATION:-0}" != 1 ]; then
        for nproc in $TOK_NPROCS; do
            base_dataset="$RESULTS/tok__${model}__np${nproc}__base/dataset"
            for label in $TOK_VARIANTS; do
                name="tok $model np$nproc $label"
                case "$label" in
                    base)
                        step "$name" tok base base "$nproc" --save ;;
                    tok|final)
                        step "$name" tok "$label" "$label" "$nproc" \
                            --compare-with "$base_dataset" ;;
                    tok_gen)
                        EFFREAD_FORMAT_IN_GEN=1 step "$name" \
                            tok tok tok_gen "$nproc" \
                            --compare-with "$base_dataset" ;;
                    tok_gen_fp)
                        if [ "$fingerprint_ok" = 1 ]; then
                            EFFREAD_FORMAT_IN_GEN=1 EFFREAD_FINGERPRINT=1 \
                                step "$name" tok tok tok_gen_fp "$nproc" \
                                --compare-with "$base_dataset"
                        fi ;;
                esac
            done
        done
    fi
done

python "$HERE/compare.py" "$RESULTS"
if [ "$n_failed" != 0 ]; then
    echo "$n_failed step(s) failed: see $RESULTS/steps.tsv and the log." >&2
    exit 1
fi
