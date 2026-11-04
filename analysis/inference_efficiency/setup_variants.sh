#!/bin/bash
# Prepare the TERRA source trees of the benchmark variants and download the
# model bundles. Run on a login node (needs git and internet access), from
# analysis/inference_efficiency/. Safe to run again: every run resets the
# variant trees to TERRA_REF and applies their patches anew.
#
#   bash setup_variants.sh
#
# TERRA is cloned from the same host, and in the same way (HTTPS or SSH), as
# this terra-reproducibility checkout. Set TERRA_REPO to clone it from
# elsewhere (another URL or an existing TERRA clone). WORK (default ./work)
# holds the trees, bundles, data and results.
#
# Variants (one git worktree each, all at TERRA_REF):
#   base  TERRA as is
#   p1    cell-only context computed on the special-token + index-cell block only
#   p2    full context without materializing the [B, 1, L, L] attention mask
#   p12   p1 + p2
#   tok   tokenization prototype (CSR row access, one ranking per cell,
#         neighbors from the CSR arrays; EFFREAD_FORMAT_IN_GEN=1 also formats
#         the examples inside the dataset generator)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$HERE/work}"
if [ -z "${TERRA_REPO:-}" ]; then
    origin="$(git -C "$HERE" remote get-url origin)"
    origin="${origin%/}"
    TERRA_REPO="${origin%/*}/terra.git"
fi
TERRA_REF="${TERRA_REF:-d4811cf}"
MODELS="${MODELS:-TERRA-112M TERRA-96M}"

mkdir -p "$WORK/src" "$WORK/bundles"
if [ ! -d "$WORK/src/repo/.git" ]; then
    # (without any credentials in the URL)
    echo "Cloning TERRA from $(sed -E 's#://[^/@]*@#://#' <<< "$TERRA_REPO")"
    git clone -q "$TERRA_REPO" "$WORK/src/repo"
fi
git -C "$WORK/src/repo" fetch -q origin

for variant in base p1 p2 p12 tok; do
    tree="$WORK/src/$variant"
    if [ ! -d "$tree" ]; then
        git -C "$WORK/src/repo" worktree add -q --detach "$tree" "$TERRA_REF"
    fi
    # Reset on every run, so that no variant keeps the code of an
    # interrupted or older setup
    git -C "$tree" reset -q --hard "$TERRA_REF"
    if [ "$variant" != base ]; then
        git -C "$tree" apply "$HERE/patches/$variant.patch"
    fi
    echo "$variant: $(git -C "$tree" rev-parse --short HEAD)" \
         "$(git -C "$tree" diff --shortstat)"
done

# Model bundles (Hugging Face 'main'). download_pretrained only fetches
# missing or changed files, so it also completes an interrupted download.
for model in $MODELS; do
    bundle="$WORK/bundles/$model"
    PYTHONPATH="$WORK/src/base/src" python -c \
        'import sys; from terra import download_pretrained; download_pretrained(sys.argv[1], local_dir=sys.argv[2])' \
        "Lotfollahi-lab/$model" "$bundle"
    for file in model_checkpoint.pt model_config.yaml token_dictionary.pkl \
                gene_count_dictionary.pkl; do
        if [ ! -f "$bundle/$file" ]; then
            echo "Bundle $model has no $file." >&2
            exit 1
        fi
    done
    if [ ! -f "$bundle/ensembl_dictionary.pkl" ] \
            && [ ! -f "$bundle/ensembl_dic.pkl" ]; then
        echo "Bundle $model has no gene mapping (ensembl_dictionary.pkl)." >&2
        exit 1
    fi
    echo "bundle $model: $(ls "$bundle" | tr '\n' ' ')"
done
