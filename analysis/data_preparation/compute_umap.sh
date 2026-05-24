#!/bin/bash

# # Add settings for different users
if [ "$USER" == "am84" ]; then
    source /etc/profile.d/modules.sh
    module load cellgen/conda
    conda activate nichejepa-reproducibility
elif [ "$USER" == "sb75" ]; then
    source /etc/profile.d/modules.sh
    module load cellgen/conda
    conda activate nichejepa_new
fi

python analysis/data_preparation/compute_umap.py