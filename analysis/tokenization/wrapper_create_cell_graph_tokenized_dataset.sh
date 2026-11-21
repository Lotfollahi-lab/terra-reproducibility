#!/bin/bash
# Submit the tokenization of one dataset (create_tokenized_dataset.sh with
# config/cell-graph-tokenizer/cell-graph-tokenizer_<dataset>.yaml) as an LSF job.
#
# Usage, from the terra-reproducibility root (the job runs there):
#   bash analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh <dataset> [cores] [queue] [ram] [walltime]
# Arguments 2-5 override the row of the resource table below (e.g. 4 hugemem 1536G 48:00).
# All 158 hst_corpus_110m datasets:
#   grep . analysis/data_preparation/hst_corpus_110m/hst_corpus_110m_dataset_names.txt | \
#       while read -r ds; do bash analysis/tokenization/wrapper_create_cell_graph_tokenized_dataset.sh "$ds"; done
#
# Memory. The resource table was sized for earlier builds (10 x 256 genes, model_input_size
# 2816, the whole dataset assembled in one dict). The hst_corpus_110m configs tokenize
# 16 x 512 genes (8704) with stream_per_file: true and storage_format: compact. The
# tokenizer's per-file dict then takes up to ~0.21 MB per cell (cells with >= 512 genes;
# about 24 bytes x 17 x mean min(genes, 512) per cell), and a parallel job holds up to
# about 3 x cores files at once (2 x cores finished, cores in progress). So a dataset
# with few large files needs about the same RAM with fewer cores, and one with many files
# less RAM with fewer cores. analysis/data_preparation/checks/tokenization_memory.py
# estimates this per dataset from the silver files and compares it with this table.
# Unverified at farm scale: watch the first large datasets (bjobs -l / the .out file's
# "Max Memory") and resubmit with arguments 2-5 when a job hits TERM_MEMLIMIT.
GROUP="team361"

# Set the experiment parameters
EXPERIMENT_NAME="${1:?usage: wrapper_create_cell_graph_tokenized_dataset.sh <dataset> [cores] [queue] [ram] [walltime]}"
TOKENIZER_NAME="cell-graph-tokenizer"
LOG_DIR="logs/${EXPERIMENT_NAME}/farm-job-outputs/${TOKENIZER_NAME}"

SCRIPT="analysis/tokenization/create_tokenized_dataset.sh"

case $EXPERIMENT_NAME in
    mmb0-1b_smb1-1b_1p)
        RAM="200G"
        TIME="12:00"
        CORES=12
        QUEUE="normal"
        ;;
    mmb0-1b_smb1-1b_1p_coord_aligned)
        RAM="200G"
        TIME="12:00"
        CORES=12
        QUEUE="normal"
        ;;
    mmb0-1b_smb1-1b_2p)
        RAM="200G"
        TIME="12:00"
        CORES=12
        QUEUE="normal"
        ;;
    mmb0-1b_smb1-1b_2p_coord_aligned)
        RAM="200G"
        TIME="12:00"
        CORES=12
        QUEUE="normal"
        ;;
    xhb1002-AT6-1-1b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhe1008-8-20b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xho47-S1-1b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhb1002-AT9-6b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhp1016-2-18b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhb1002-AT5-1b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhe1008-7-20b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhb1002-AT15-10b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhe1008-6-20b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhb1002-AT6-4-1b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhb1002-AT10-7b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhe1008-9-20b_1p)
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xho47-S2-1b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1015-R6-28b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1015-R1-28b_1p)    
        RAM="1536G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1015-R3-28b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1015-R4-28b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1015-R5-28b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1015-R7-28b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhk1018-5-23b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhk1018-6-23b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xht43-2b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1020-8-16b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mhp20-2b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1013-4b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mhu21-3b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhb1002-AT16-2-15b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1020-6-16b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1021-1-2b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mhl63-10b_1p)    
        RAM="512G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xho1012-AU16-7b_1p)    
        RAM="512G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhl1020-3-16b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mhc15-2b_1p)    
        RAM="600G"
        TIME="24:00"
        CORES=12
        QUEUE="long"
        ;;
    xhc38-4b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    xhp1017-4b_1p)    
        RAM="1024G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mmb0-P1-239b_1p)    
        RAM="1536G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mmb0-P2-239b_1p)    
        RAM="1536G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mmb0-P3-239b_1p)    
        RAM="1536G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mmb0-P4-239b_1p)    
        RAM="1536G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    mmb0-P5-239b_1p)    
        RAM="1536G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    smb1-20b_1p)    
        RAM="1536G"
        TIME="24:00"
        CORES=12
        QUEUE="hugemem"
        ;;
    *)
        #RAM="2200G" # "600G", "1000G", "1500G"
        #TIME="240:00" # "48:00", "120:00"
        #CORES=12
        #QUEUE="teramem" # "long", "hugemem", "teramem"
        RAM="512G" # "600G", "1000G", "1500G"
        TIME="24:00" # "48:00", "120:00"
        CORES=12
        QUEUE="long" # "long", "hugemem", "teramem", "basement", "parallel", "week"
        ;;
esac

# If not provided, use the default value
CORES="${2:-$CORES}"
QUEUE="${3:-$QUEUE}"
RAM="${4:-$RAM}"
TIME="${5:-$TIME}"

# Tokenization needs no GPU; GPU queues (inference, gpu-*) also need another group
if [[ "$QUEUE" == *gpu* || "$QUEUE" == inference ]]; then
    echo "Queue '${QUEUE}' is a GPU queue; tokenization runs on CPU queues (long, hugemem, teramem)." >&2
    exit 1
fi

# The job runs in CWD (default: the current directory, which must be the
# terra-reproducibility root: the config and script paths are relative to it)
CWD="${CWD:-$(pwd)}"

# Set the configuration file
CONFIG_FILE="config/${TOKENIZER_NAME}/${TOKENIZER_NAME}_${EXPERIMENT_NAME}.yaml"
if [[ ! -f "${CWD}/${CONFIG_FILE}" || ! -f "${CWD}/${SCRIPT}" ]]; then
    echo "${CWD}/${CONFIG_FILE} or ${CWD}/${SCRIPT} not found; run from the terra-reproducibility root (or set CWD)." >&2
    exit 1
fi

# Set the output and error files
mkdir -p "${CWD}/${LOG_DIR}"
OUTPUT_FILE="${LOG_DIR}/${TOKENIZER_NAME}_${CORES}_%J.out"
ERROR_FILE="${LOG_DIR}/${TOKENIZER_NAME}_${CORES}_%J.err"

JOB_NAME="${EXPERIMENT_NAME}_${TOKENIZER_NAME}_${CORES}"

echo "Log directory: ${CWD}/${LOG_DIR}"
echo "Running ${TOKENIZER_NAME} on ${EXPERIMENT_NAME}: ${CORES} cores, queue ${QUEUE}, ${RAM} RAM, ${TIME} walltime, config ${CONFIG_FILE}"

bsub \
    -G "${GROUP}" \
    -n "${CORES}" \
    -q "${QUEUE}" \
    -M "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}] span[hosts=1]" \
    -W "${TIME}" \
    -cwd "${CWD}" \
    -o "${OUTPUT_FILE}" \
    -e "${ERROR_FILE}" \
    -J "${JOB_NAME}" \
    "${SCRIPT}" "${CONFIG_FILE}"
