#!/bin/bash

CONFIG_FILE=$1  # path to the config file

# Modify to match your python environment activation command(s)
if [ "$USER" == "am84" ]; then
    source /etc/profile.d/modules.sh
    module load cellgen/conda
    conda activate nichejepa-reproducibility
elif [ "$USER" == "sb75" ]; then
    source ~/.venvs/nichejepa/bin/activate
fi

python analysis/tokenization/create_tokenized_dataset.py \
    --config_file ${CONFIG_FILE}