#!/bin/bash
# Prepare the TERRA source trees of the benchmark variants and download the
# model bundles. Run on a login node (needs git and internet access), from
# analysis/inference_efficiency/.
#
#   WORK=/lustre/.../terra_inference_efficiency bash setup_variants.sh
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
TERRA_REPO="${TERRA_REPO:-git@github.com:Lotfollahi-lab/terra.git}"
TERRA_REF="${TERRA_REF:-d4811cf}"
MODELS="${MODELS:-TERRA-112M TERRA-96M}"

mkdir -p "$WORK/src" "$WORK/bundles"
if [ ! -d "$WORK/src/repo/.git" ]; then
    git clone -q "$TERRA_REPO" "$WORK/src/repo"
fi
git -C "$WORK/src/repo" fetch -q origin

for variant in base p1 p2 p12 tok; do
    tree="$WORK/src/$variant"
    if [ ! -d "$tree" ]; then
        git -C "$WORK/src/repo" worktree add -q --detach "$tree" "$TERRA_REF"
        if [ "$variant" != base ]; then
            git -C "$tree" apply "$HERE/patches/$variant.patch"
        fi
    fi
    echo "$variant: $(git -C "$tree" rev-parse --short HEAD)" \
         "$(git -C "$tree" diff --shortstat)"
done

# Model bundles (Hugging Face 'main': the complete bundle with
# ensembl_dictionary.pkl and gene_count_dictionary.pkl)
for model in $MODELS; do
    if [ ! -f "$WORK/bundles/$model/model_checkpoint.pt" ]; then
        PYTHONPATH="$WORK/src/base/src" python -c \
            "from terra import download_pretrained; download_pretrained('Lotfollahi-lab/$model', local_dir='$WORK/bundles/$model')"
    fi
    echo "bundle $model: $(ls "$WORK/bundles/$model" | tr '\n' ' ')"
done
