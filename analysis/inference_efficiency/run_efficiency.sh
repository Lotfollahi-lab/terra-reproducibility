#!/bin/bash
# Benchmark of the inference-efficiency prototypes on one GPU node (see
# README.md). Needs setup_variants.sh to have run (source trees + bundles).
#
#   H5AD=/path/to/sample.h5ad bash run_efficiency.sh
#
# Optional: WORK, MODELS, N_CELLS (20000), SAMPLE_KEY + SAMPLE, BATCH_SIZE
# (128), NUM_WORKERS (12), REPS (2), TOK_NPROCS ("1 4"), SKIP_TOKENIZATION=1,
# ALLOW_CPU=1 (only for a CPU smoke test of the scripts).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$HERE/work}"
H5AD="${H5AD:?set H5AD to a raw-count spatial AnnData (obsm['spatial'])}"
MODELS="${MODELS:-TERRA-112M TERRA-96M}"
N_CELLS="${N_CELLS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-12}"
REPS="${REPS:-2}"
TOK_NPROCS="${TOK_NPROCS:-1 4}"
RESULTS="$WORK/results"
if [ -n "${SAMPLE_KEY:-}" ] && [ -z "${SAMPLE:-}" ]; then
    echo "SAMPLE_KEY is set: set SAMPLE to the section to use as well." >&2
    exit 1
fi
mkdir -p "$RESULTS"

# Run python with the TERRA source tree of the given variant
py() {
    local variant="$1"
    shift
    PYTHONPATH="$WORK/src/$variant/src" TERRA_TREE="$WORK/src/$variant/src" \
        python "$@"
}

# Tokenization of $data/harmonized.h5ad with one variant
tok() {
    local variant="$1" label="$2" nproc="$3"
    echo "== $model tokenization, nproc $nproc: $label"
    py "$variant" "$HERE/bench_tokenize.py" run \
        --harmonized "$data/harmonized.h5ad" --bundle "$bundle" \
        --out "$RESULTS/tok__${model}__np${nproc}__${label}" \
        --nproc "$nproc" --reps "$REPS"
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

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -c 'import torch; print("torch", torch.__version__)'

for model in $MODELS; do
    bundle="$WORK/bundles/$model"
    data="$WORK/data/$model"
    if [ ! -f "$bundle/model_checkpoint.pt" ]; then
        echo "Missing bundle $bundle: run setup_variants.sh on a login node." >&2
        exit 1
    fi

    # Common input: harmonized AnnData + dataset tokenized with the base tree
    if [ ! -d "$data/dataset" ]; then
        py base "$HERE/prepare_data.py" --h5ad "$H5AD" --bundle "$bundle" \
            --out "$data" --n-cells "$N_CELLS" \
            ${SAMPLE_KEY:+--sample-key "$SAMPLE_KEY" --sample "$SAMPLE"}
    fi

    # Tokenization: base; prototype; prototype formatting the examples in
    # the generator; the same with a random dataset fingerprint
    if [ "${SKIP_TOKENIZATION:-0}" != 1 ]; then
        for nproc in $TOK_NPROCS; do
            tok base base "$nproc"
            tok tok tok "$nproc"
            EFFREAD_FORMAT_IN_GEN=1 tok tok tok_gen "$nproc"
            EFFREAD_FORMAT_IN_GEN=1 EFFREAD_FINGERPRINT=1 \
                tok tok tok_gen_fp "$nproc"
        done
    fi

    # Embedding: every variant embeds the same tokenized dataset
    emb base base
    emb base base_rerun                  # determinism control
    emb p2 p2
    emb p1 p1
    emb p12 p12
    emb base base_pin --pin-memory       # pinned host memory
    emb p12 p12_pin --pin-memory
    emb base base_nw0 --num-workers 0    # no DataLoader workers
done

python "$HERE/compare.py" "$RESULTS"
