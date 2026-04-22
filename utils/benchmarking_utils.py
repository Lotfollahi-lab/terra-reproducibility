from typing import List, Literal, Optional

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment


def max_overlap_plot(df, col1, col2):
    """
    Plots the maximum overlap between two label columns in a DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data.
    - col1: str, name of the first label column.
    - col2: str, name of the second label column.
    """
    # Create the contingency table
    contingency_table = pd.crosstab(df[col1], df[col2])

    # Convert to cost matrix
    cost_matrix = -contingency_table.values

    # Apply Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = list(zip(contingency_table.index[row_ind], contingency_table.columns[col_ind]))
    total_overlap = -cost_matrix[row_ind, col_ind].sum()

    print("Optimal Mappings:")
    for col1_label, col2_label in mapping:
        overlap = contingency_table.loc[col1_label, col2_label]
        print(f"{col1_label} -> {col2_label} (Overlap: {overlap})")
    print(f"\nTotal Maximum Overlap: {total_overlap}")

    # Plot Heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues')

    # Highlight optimal mappings
    for (label1, label2) in mapping:
        row_idx = contingency_table.index.get_loc(label1)
        col_idx = contingency_table.columns.get_loc(label2)
        ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='red', lw=3))

    plt.title(f'Contingency Table with Optimal Mappings between {col1} and {col2}')
    plt.xlabel(f'{col2} Labels')
    plt.ylabel(f'{col1} Labels')
    plt.show()