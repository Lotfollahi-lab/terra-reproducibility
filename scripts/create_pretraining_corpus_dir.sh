#!/bin/bash

# cohort_name="human_cohort2_2m"
cohort_name="human_cohort2_10m"
# cohort_name="human_cohort1_50m"

# File containing the list of DATASETS names
DATASET_NAMES_FILE="/lustre/scratch126/cellgen/team361/am84/NicheJEPA/reproducibility/scripts/${cohort_name}_dataset_names.txt"

# Check if the file exists
if [[ ! -f $DATASET_NAMES_FILE ]]; then
    echo "Error: File $DATASET_NAMES_FILE does not exist."
    exit 1
fi

mkdir /lustre/scratch126/cellgen/team361/DATASETS/silver/$cohort_name

# Loop through each line in the file
while IFS= read -r dataset_name; do
    # Skip empty lines
    if [[ -z "$dataset_name" ]]; then
        continue
    fi
    
    echo "Copying Dataset: $dataset_name"

    cp -r /lustre/scratch126/cellgen/team361/DATASETS/silver/$dataset_name /lustre/scratch126/cellgen/team361/DATASETS/silver/$cohort_name

done < "$DATASET_NAMES_FILE"

echo "All datasets copied."