#!/bin/bash
# One TERRA training run on Slurm. submit_training.sh submits it with
# SITE_CONFIG, RUN_DIR, RUN_CONFIG, RUN_NAME, EXPERIMENT_NAME and TRAIN_GPUS
# in the environment; do not sbatch it by hand.
# One node: torchrun --standalone with TRAIN_GPUS processes. Several nodes:
# srun starts one torchrun per node, which meet at the first node (c10d).
set -euo pipefail
: "${SITE_CONFIG:?SITE_CONFIG is not set (submit with submit_training.sh)}"
: "${RUN_DIR:?}" "${RUN_CONFIG:?}" "${RUN_NAME:?}"
# shellcheck disable=SC1090
source "$SITE_CONFIG"
set +u
eval "$TERRA_ENV_ACTIVATE"
set -u
# The configs give their paths as ${TERRA_DATA_DIR}/...; EXPERIMENT_NAME and
# RUN_NAME make the wandb run id
export TERRA_DATA_DIR EXPERIMENT_NAME="${EXPERIMENT_NAME:-terra}" RUN_NAME
export WANDB_MODE="${WANDB_MODE:-offline}" WANDB_DIR="$RUN_DIR"
export PYTHONUNBUFFERED=1
GPUS="${TRAIN_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
NODES="${SLURM_NNODES:-1}"
MAIN="$TERRA_REPO_DIR/src/terra/training/main.py"

echo "== $(date '+%F %T') run $RUN_NAME, Slurm job ${SLURM_JOB_ID:-none}: $NODES node(s) x $GPUS GPU(s) on $(hostname)"
echo "== config $RUN_CONFIG"
echo "== folder $RUN_DIR; wandb $WANDB_MODE"
echo "== TERRA $(git -C "$TERRA_REPO_DIR" log --oneline -1 2>/dev/null || echo '?') ($(git -C "$TERRA_REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?'))"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "== (nvidia-smi not available)"
cd "$RUN_DIR"
if [ "$NODES" = 1 ]; then
    torchrun --standalone --nnodes=1 --nproc_per_node="$GPUS" \
        "$MAIN" --backend nccl --fname "$RUN_CONFIG"
else
    HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    PORT=$((29400 + ${SLURM_JOB_ID:-0} % 1000))
    srun --ntasks-per-node=1 torchrun --nnodes="$NODES" --nproc_per_node="$GPUS" \
        --rdzv_id="${SLURM_JOB_ID:-0}" --rdzv_backend=c10d --rdzv_endpoint="$HEAD:$PORT" \
        "$MAIN" --backend nccl --fname "$RUN_CONFIG"
fi
echo "== $(date '+%F %T') run $RUN_NAME finished"
