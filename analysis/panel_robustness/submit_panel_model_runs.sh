#!/bin/bash
# Package the trained models of a panel-training run and embed and evaluate each with gene_panel_subsampling.py on the
# reference run's cells and panels: a model trained on the full panel on every panel of the reference run (the
# zero-shot arm with the same training data), a model trained on a smaller panel on that panel only. Each model is a
# gene_panel_subsampling run NAME_<model> (submit_gene_panel_subsampling.sh). Run it again when more models have
# finished training: packaged models and finished conditions are skipped.
#
# Usage: bash submit_panel_model_runs.sh NAME [options of `gene_panel_subsampling.py configure`]
# Environment: as submit_gene_panel_subsampling.sh, plus ALLOW_UNFINISHED=1 (package models whose training has not
# reached the last epoch, e.g. to look at a checkpoint early).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

NAME=${1:?Usage: bash submit_panel_model_runs.sh NAME [configure options]}
shift
TRAINING_ROOT=${TRAINING_ROOT:-/nfs/team361/sb75/terra-reproducibility/artifacts/panel_training}
TRAINING_FOLDER_PATH=${TRAINING_ROOT}/${NAME}

source /nfs/team361/sb75/.venvs/terra/bin/activate
python panel_training.py package --training_folder_path "${TRAINING_FOLDER_PATH}" ${ALLOW_UNFINISHED:+--allow_unfinished}

N_SUBMITTED=0
while IFS=$'\t' read -r MODEL PANEL MODEL_FOLDER REFERENCE_RUN; do
    OPTIONS=(--model_folder_path "${MODEL_FOLDER}" --panels_from "${REFERENCE_RUN}")
    if [ "${PANEL}" != "full" ]; then
        OPTIONS+=(--conditions "${PANEL}")
    fi
    echo "== ${MODEL} (trained on ${PANEL})"
    bash submit_gene_panel_subsampling.sh "${NAME}_${MODEL}" "${OPTIONS[@]}" "$@"
    N_SUBMITTED=$((N_SUBMITTED + 1))
done < <(python panel_training.py list_packaged --training_folder_path "${TRAINING_FOLDER_PATH}")

if [ "${N_SUBMITTED}" -eq 0 ]; then
    echo "No packaged model yet (see: python panel_training.py status --training_folder_path ${TRAINING_FOLDER_PATH})."
else
    echo "Submitted the embedding runs of ${N_SUBMITTED} models. When they have finished:"
    echo "  bash submit_panel_comparison.sh ${NAME}"
fi
