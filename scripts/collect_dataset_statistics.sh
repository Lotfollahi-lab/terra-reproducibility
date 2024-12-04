#!/bin/bash

cohort_name="human_cohort2_2m"
# cohort_name="human_cohort2_10m"
# cohort_name="human_cohort1_50m"

# File containing the list of DATASETS names
DATASET_NAMES_FILE="/lustre/scratch126/cellgen/team361/am84/NicheJEPA/reproducibility/scripts/${cohort_name}_dataset_names.txt"

# Check if the file exists
if [[ ! -f $DATASET_NAMES_FILE ]]; then
    echo "Error: File $DATASET_NAMES_FILE does not exist."
    exit 1
fi

# Loop through each line in the file
while IFS= read -r dataset_name; do
    # Skip empty lines
    if [[ -z "$dataset_name" ]]; then
        continue
    fi
    
    echo "Collecting Statistics for Dataset: $dataset_name"


    python /lustre/scratch126/cellgen/team361/am84/NicheJEPA/reproducibility/analysis/statistics/compute_anndata_statistics.py \
        --dataset_name "$dataset_name"

done < "$DATASET_NAMES_FILE"

echo "All datasets processed."