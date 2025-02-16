#!/bin/bash
CONFIG_FILE=$1 # path to the config file
DEVICES=$2 # GPU devices

. /usr/share/modules/init/bash
module load cuda-12.1.1
module load cellgen/conda
conda activate nichejepa_new

python ../nichejepa/main.py \
    --fname ${CONFIG_FILE} \
    --devices ${DEVICES}

echo "Finished script." 