#!/bin/bash

CONFIG_FILE=$1  # path to the config file

# Modify to match your python environment activation command(s)
if [ "$USER" == "am84" ]; then
    source /etc/profile.d/modules.sh
    module load cellgen/conda
    conda activate terra-reproducibility
elif [ "$USER" == "sb75" ]; then
    # The venv with the editable TERRA checkout /nfs/team361/sb75/terra
    source /nfs/team361/sb75/.venvs/terra/bin/activate
fi
# Print which TERRA the job uses (the dataset record stores only its version)
python -c "import os, subprocess, terra; print('TERRA', terra.__version__, terra.__file__, subprocess.run(['git', '-C', os.path.dirname(terra.__file__), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip())"

python analysis/tokenization/create_tokenized_dataset.py \
    --config_file ${CONFIG_FILE}