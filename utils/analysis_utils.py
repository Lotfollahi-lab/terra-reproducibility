import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_stats(adata,
                             embedding_key='jepa_emb',
                             type_key='niche_type',
                             n_samples=10000,
                             random_state=42):
    """
    Function to compute mean and variance of cosine similarity between pairs of types (e.g., cell types).
    
    Parameters:
    - adata: AnnData object containing the data.
    - embedding_key: The key to use in .obsm for the embeddings (default: 'jepa_emb').
    - type_key: The key in .obs used to stratify and differentiate types (default: 'niche_type').
    - n_samples: Number of samples to draw from the dataset (default: 10000).
    - random_state: Seed for reproducibility (default: 42).
    
    Returns:
    - A DataFrame with mean and variance of cosine similarities between each pair of types.

    Example usage:
    - similarity_stats = compute_similarity_stats(adata)
    """
    
    # Stratified sampling based on the specified types
    types = adata.obs[type_key]
    if n_samples is not None:
      sampled_indices, _ = train_test_split(
        np.arange(adata.n_obs), 
        train_size=n_samples, 
        stratify=types, 
        random_state=random_state
      )
    
      # Subset the AnnData object to keep only the sampled cells
      adata_sampled = adata[sampled_indices].copy()
    else:
      adata_sampled = adata  
    # Extract the embeddings or the data matrix (from .obsm)
    data = adata_sampled.obsm[embedding_key]  # Replace with appropriate layer if necessary

    # Get the unique types
    unique_types = adata_sampled.obs[type_key].unique()

    # List to store the results for each class pair
    similarity_stats_list = []

    # Compute similarity between each pair of types
    for i, type_1 in enumerate(unique_types):
        for j, type_2 in enumerate(unique_types):
            # Select cells of the respective types
            cells_1 = data[adata_sampled.obs[type_key] == type_1]
            cells_2 = data[adata_sampled.obs[type_key] == type_2]

            # Compute pairwise cosine similarity between cells from type 1 and type 2
            similarity_matrix = cosine_similarity(cells_1, cells_2)
            # Compute mean and variance of the similarity scores
            mean_similarity = np.mean(similarity_matrix)
            variance_similarity = np.var(similarity_matrix)

            # Append results to the list
            similarity_stats_list.append({
                'Class 1': type_1,
                'Class 2': type_2,
                'Mean Similarity': mean_similarity,
                'Variance Similarity': variance_similarity
            })

    # Convert the list to a DataFrame
    similarity_stats = pd.DataFrame(similarity_stats_list)

    # Optionally print the resulting DataFrame
    print(similarity_stats)

    # Return the similarity stats DataFrame
    return similarity_stats


def plot_similarity_heatmaps(similarity_stats, title_mean='Mean Similarity Between Types', title_variance='Variance of Similarity Between Types'):
    """
    Function to create and display two heatmaps for Mean Similarity and Variance Similarity side by side.
    
    Parameters:
    - similarity_stats: DataFrame containing the similarity statistics with 'Class 1', 'Class 2', 'Mean Similarity', and 'Variance Similarity'.
    - title_mean: Title for the mean similarity heatmap (default: 'Mean Similarity Between Types').
    - title_variance: Title for the variance similarity heatmap (default: 'Variance of Similarity Between Types').
    
    Returns:
    - Displays two heatmaps (Mean Similarity and Variance Similarity).
    """
    # Create pivot tables for mean similarity and variance similarity
    similarity_pivot_mean = similarity_stats.pivot(index='Class 1', columns='Class 2', values='Mean Similarity')
    similarity_pivot_variance = similarity_stats.pivot(index='Class 1', columns='Class 2', values='Variance Similarity')
    
    # Create the figure and two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Mean similarity heatmap
    sns.heatmap(similarity_pivot_mean, annot=True, cmap="coolwarm", cbar=True, fmt=".2f", ax=axes[0])
    axes[0].set_title(title_mean)
    axes[0].set_xlabel('Class 2')
    axes[0].set_ylabel('Class 1')
    
    # Variance similarity heatmap
    sns.heatmap(similarity_pivot_variance, annot=True, cmap="coolwarm", cbar=True, fmt=".2f", ax=axes[1])
    axes[1].set_title(title_variance)
    axes[1].set_xlabel('Class 2')
    axes[1].set_ylabel('Class 1')
    
    # Adjust the layout
    plt.tight_layout()
    
    # Display the heatmaps
    plt.show()

# Example usage:
# plot_similarity_heatmaps(similarity_stats)