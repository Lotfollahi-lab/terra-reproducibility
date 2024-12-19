#!/bin/bash

while read line; do
    dataset_name=$(echo $line | cut -d ' ' -f1)
    source /lustre/scratch126/cellgen/team361/am84/NicheJEPA/reproducibility/scripts/wrapper_create_cell_graph_tokenized_dataset.sh ${dataset_name}
done < <(cat /lustre/scratch126/cellgen/team361/am84/NicheJEPA/reproducibility/analysis/tokenization/human_cohort1_50m_dataset_names.txt)