#!/bin/bash
# Submit one TERRA training run to Slurm (torchrun on one or more GPU nodes)
# with a config of make_training_configs.py and the site settings of the
# corpus build.
#
# Usage (on a login node):
#   bash analysis/slurm_corpus_build/submit_training.sh --config <site_config.sh> \
#       --training_config <config.yaml> [--run_name <name>] [--dry_run]
#   bash analysis/slurm_corpus_build/submit_training.sh --config <site_config.sh> \
#       --resume <run folder> [--dry_run]
#
# A new run gets the folder $TRAIN_RUNS_DIR/<config name>/<run name> (default
# run name: <config name>_<date>_<time>). The job reads a copy of the config
# there with state.folder_path set to the folder, and writes its checkpoints,
# logs and offline wandb files into it.
# --resume continues a run in its folder from its last end-of-epoch
# checkpoint (<write_tag>-ep<N>.pth.tar), with its run name and config.
#
# Site settings (site_config.sh), besides TERRA_DATA_DIR and TERRA_ENV_ACTIVATE:
#   TERRA_REPO_DIR     TERRA checkout (src/terra/training/main.py); required
#   TRAIN_PARTITION    GPU partition; required
#   TRAIN_NODES        nodes (default 1)
#   TRAIN_GPUS         GPUs per node (default 4). The config must be made for
#                      TRAIN_NODES x TRAIN_GPUS GPUs (its training_gpus line)
#   TRAIN_GPU_REQUEST  Slurm GPU request (default --gpus-per-node=$TRAIN_GPUS,
#                      e.g. --gres=gpu:a100:4)
#   TRAIN_CPUS         CPUs per node (default 16 per GPU)
#   TRAIN_MEM          memory per node (default 64G per GPU)
#   TRAIN_TIME         time limit (default 72:00:00)
#   TRAIN_RUNS_DIR     run folders (default <parent of TERRA_DATA_DIR>/runs)
#   WANDB_MODE         default offline (wandb files stay in the run folder)
#   SLURM_EXTRA_ARGS   e.g. "--account=... --qos=..."
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG="" TRAINING_CONFIG="" RESUME="" RUN_NAME="" DRY_RUN=0
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
fail() { echo "ERROR: $*" >&2; exit 1; }
while [ $# -gt 0 ]; do
    case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --training_config) TRAINING_CONFIG="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    --dry_run) DRY_RUN=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument $1"; usage; exit 1 ;;
    esac
done
[ -n "$CONFIG" ] || { usage; exit 1; }
{ [ -n "$TRAINING_CONFIG" ] && [ -z "$RESUME" ]; } || { [ -z "$TRAINING_CONFIG" ] && [ -n "$RESUME" ]; } \
    || fail "give either --training_config or --resume"
[ -f "$CONFIG" ] || fail "--config $CONFIG does not exist"
CONFIG=$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")
# shellcheck disable=SC1090
source "$CONFIG"
for VAR in TERRA_DATA_DIR TERRA_ENV_ACTIVATE TERRA_REPO_DIR TRAIN_PARTITION; do
    [ -n "${!VAR:-}" ] || fail "$VAR is not set in $CONFIG"
done
[ -f "$TERRA_REPO_DIR/src/terra/training/main.py" ] || fail "TERRA_REPO_DIR $TERRA_REPO_DIR has no src/terra/training/main.py"
TRAIN_NODES=${TRAIN_NODES:-1}
TRAIN_GPUS=${TRAIN_GPUS:-4}
TRAIN_GPU_REQUEST=${TRAIN_GPU_REQUEST:---gpus-per-node=$TRAIN_GPUS}
TRAIN_CPUS=${TRAIN_CPUS:-$((16 * TRAIN_GPUS))}
TRAIN_MEM=${TRAIN_MEM:-$((64 * TRAIN_GPUS))G}
TRAIN_TIME=${TRAIN_TIME:-72:00:00}
TRAIN_RUNS_DIR=${TRAIN_RUNS_DIR:-$(dirname "$TERRA_DATA_DIR")/runs}

if [ -n "$TRAINING_CONFIG" ]; then
    [ -f "$TRAINING_CONFIG" ] || fail "--training_config $TRAINING_CONFIG does not exist"
    TRAINING_CONFIG=$(cd "$(dirname "$TRAINING_CONFIG")" && pwd)/$(basename "$TRAINING_CONFIG")
    NAME=$(basename "$TRAINING_CONFIG" .yaml)
    RUN_NAME=${RUN_NAME:-${NAME}_$(date +%Y%m%d_%H%M%S)}
    echo "$RUN_NAME" | grep -Eq '^[A-Za-z0-9._-]+$' || fail "--run_name '$RUN_NAME': use letters, digits, '.', '_', '-'"
    RUN_DIR="$TRAIN_RUNS_DIR/$NAME/$RUN_NAME"
    [ ! -e "$RUN_DIR" ] || fail "$RUN_DIR exists: give another --run_name, or --resume it"
    SOURCE="$TRAINING_CONFIG"
    RUN_CONFIG="$RUN_DIR/training_config.yaml"
    LOAD=false READ=null
else
    [ -d "$RESUME" ] || fail "--resume $RESUME is not a folder"
    RUN_DIR=$(cd "$RESUME" && pwd)
    for f in training_config.yaml run_name experiment_name; do
        [ -f "$RUN_DIR/$f" ] || fail "$RUN_DIR has no $f: not a folder of submit_training.sh"
    done
    [ -z "$RUN_NAME" ] || fail "--run_name does not go with --resume (the run keeps its name)"
    RUN_NAME=$(cat "$RUN_DIR/run_name")
    NAME=$(cat "$RUN_DIR/experiment_name")
    SOURCE="$RUN_DIR/training_config.yaml"
    # Last end-of-epoch checkpoint: a mid-epoch save can overwrite
    # <tag>-latest.pth.tar (TERRA TR-1), so the resume names the epoch file
    TAG=$(sed -n 's/^  write_tag: *//p' "$SOURCE" | head -n 1 | tr -d "'\"")
    LAST=$(ls "$RUN_DIR" | sed -n "s/^${TAG}-ep\([0-9][0-9]*\)\.pth\.tar\$/\1/p" | sort -n | tail -n 1)
    [ -n "$LAST" ] || fail "$RUN_DIR has no end-of-epoch checkpoint ${TAG}-ep<N>.pth.tar: start a new run"
    RUN_CONFIG="$RUN_DIR/training_config_resume_ep${LAST}.yaml"
    LOAD=true READ="'${TAG}-ep${LAST}.pth.tar'"
fi

# The config's learning rate, EMA and checkpoint spacing are made for its GPU count
CONFIG_GPUS=$(sed -n 's/^# training_gpus: *\([0-9][0-9]*\).*/\1/p' "$SOURCE" | head -n 1)
JOB_GPUS=$((TRAIN_NODES * TRAIN_GPUS))
if [ -n "$CONFIG_GPUS" ] && [ "$CONFIG_GPUS" != "$JOB_GPUS" ]; then
    fail "the config is made for $CONFIG_GPUS GPU(s), the job has $TRAIN_NODES x $TRAIN_GPUS = $JOB_GPUS: make the configs for $JOB_GPUS GPUs (make_training_configs.py --gpus $JOB_GPUS) or change TRAIN_NODES / TRAIN_GPUS"
fi
for KEY in folder_path load_checkpoint read_checkpoint; do
    [ "$(grep -c "^  $KEY:" "$SOURCE")" = 1 ] || fail "$SOURCE needs one state.$KEY line"
done

read -r -a GPU_ARGS <<<"$TRAIN_GPU_REQUEST"
read -r -a EXTRA_SBATCH <<<"${SLURM_EXTRA_ARGS:-}"
CMD=(sbatch --parsable --job-name "terra_${NAME}" --partition "$TRAIN_PARTITION"
    --nodes "$TRAIN_NODES" --ntasks-per-node 1 --cpus-per-task "$TRAIN_CPUS"
    --mem "$TRAIN_MEM" --time "$TRAIN_TIME" "${GPU_ARGS[@]}"
    --output "$RUN_DIR/slurm_%j.out"
    --export "ALL,SITE_CONFIG=$CONFIG,RUN_DIR=$RUN_DIR,RUN_CONFIG=$RUN_CONFIG,RUN_NAME=$RUN_NAME,EXPERIMENT_NAME=$NAME,TRAIN_GPUS=$TRAIN_GPUS"
    ${EXTRA_SBATCH[@]+"${EXTRA_SBATCH[@]}"} "$SCRIPT_DIR/train_job.sh")

echo "Run $RUN_NAME: $TRAIN_NODES node(s) x $TRAIN_GPUS GPU(s) on $TRAIN_PARTITION"
echo "Folder $RUN_DIR"
echo "Config $RUN_CONFIG ($([ "$LOAD" = true ] && echo "resume from $READ" || echo "copy of $SOURCE"))"
if [ $DRY_RUN = 1 ]; then
    echo "[dry run] ${CMD[*]}"
    exit 0
fi
mkdir -p "$RUN_DIR"
sed -e "s#^  folder_path:.*#  folder_path: '$RUN_DIR'#" \
    -e "s#^  load_checkpoint:.*#  load_checkpoint: $LOAD#" \
    -e "s#^  read_checkpoint:.*#  read_checkpoint: $READ#" "$SOURCE" >"$RUN_CONFIG.tmp"
grep -q "^  folder_path: '$RUN_DIR'\$" "$RUN_CONFIG.tmp" || fail "could not set state.folder_path in $RUN_CONFIG"
mv "$RUN_CONFIG.tmp" "$RUN_CONFIG"
echo "$RUN_NAME" >"$RUN_DIR/run_name"
echo "$NAME" >"$RUN_DIR/experiment_name"
JOB_ID=$("${CMD[@]}")
JOB_ID=${JOB_ID%%;*}
echo "Submitted job $JOB_ID; log $RUN_DIR/slurm_${JOB_ID}.out"
