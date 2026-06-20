#!/bin/bash

# cohort_name="human_cohort2_2m"
cohort_name="human_cohort2_10m"
# cohort_name="human_cohort1_50m"

# File containing the list of DATASETS names
DATASET_NAMES_FILE="/lustre/scratch126/cellgen/team361/am84/TERRA/reproducibility/analysis/data_preparation/${cohort_name}/${cohort_name}_dataset_names.txt"

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
    
    echo "Harmonizing Dataset: $dataset_name"

    config_file="/lustre/scratch126/cellgen/team361/am84/TERRA/reproducibility/config/bronze_to_silver/${dataset_name}.yaml"
    echo "Config file: $config_file"

    /lustre/scratch126/cellgen/team361/am84/TERRA/reproducibility/scripts/create_harmonized_data.sh "$config_file" >> /lustre/scratch126/cellgen/team361/am84/TERRA/reproducibility/logs/harmonize_datasets_${cohort_name}.log 2>&1

done < "$DATASET_NAMES_FILE"

echo "All datasets harmonized."