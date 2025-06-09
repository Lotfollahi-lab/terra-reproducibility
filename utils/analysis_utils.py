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


def plot_similarity_heatmaps(similarity_stats, 
                             title_mean='Mean Similarity Between Types', 
                             title_variance='Variance of Similarity Between Types',
                             save_file_path=None):
    """
    Function to create and display two heatmaps for Mean Similarity and Variance Similarity side by side.
    
    Parameters:
    - similarity_stats: DataFrame containing the similarity statistics with 'Class 1', 'Class 2', 
                        'Mean Similarity', and 'Variance Similarity'.
    - title_mean: Title for the mean similarity heatmap (default: 'Mean Similarity Between Types').
    - title_variance: Title for the variance similarity heatmap (default: 'Variance of Similarity Between Types').
    - save_file_path: File path to save the figure. If None, the figure is not saved (default: None).
    
    Returns:
    - Displays two heatmaps (Mean Similarity and Variance Similarity). If a file path is provided, the plot is also saved.
    """
    # Create pivot tables for mean similarity and variance similarity
    similarity_pivot_mean = similarity_stats.pivot(index='Class 1', columns='Class 2', values='Mean Similarity')
    similarity_pivot_variance = similarity_stats.pivot(index='Class 1', columns='Class 2', values='Variance Similarity')
    
    # Create the figure and two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Mean similarity heatmap without gridlines
    ax0 = axes[0]
    sns.heatmap(similarity_pivot_mean, annot=True, cmap="coolwarm", cbar=True, fmt=".2f",
                ax=ax0, linewidths=0, linecolor='none')
    ax0.set_title(title_mean)
    ax0.set_xlabel('Class 2')
    ax0.set_ylabel('Class 1')
    
    # Remove axes spines for a cleaner look
    for spine in ax0.spines.values():
        spine.set_visible(False)
    
    # Variance similarity heatmap without gridlines
    ax1 = axes[1]
    sns.heatmap(similarity_pivot_variance, annot=True, cmap="coolwarm", cbar=True, fmt=".2f",
                ax=ax1, linewidths=0, linecolor='none')
    ax1.set_title(title_variance)
    ax1.set_xlabel('Class 2')
    ax1.set_ylabel('Class 1')
    
    # Remove axes spines for a cleaner look
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the plot if a file path is provided
    if save_file_path:
        plt.savefig(save_file_path,
                    dpi=300,
                    bbox_inches='tight')
    
    # Display the heatmaps
    plt.show()

# Example usage:
# plot_similarity_heatmaps(similarity_stats)

custom_palette = {
    "0": "#66C5CC",
    "1": "#F6CF71",
    "2": "#F89C74",
    "3": "#DCB0F2",
    "4": "#87C55F",
    "5": "#9EB9F3",
    "6": "#FE88B1",
    "7": "#C9DB74",
    "8": "#8BE0A4",
    "9": "#B497E7",
    "10": "#D3B484",
    "11": "#B3B3B3",
    "12": "#276A8C", # Royal Blue
    "13": "#DAB6C4", # Pink
    "14": "#C38D9E", # Mauve-Pink
    "15": "#9D88A2", # Mauve
    "16": "#FF4D4D", # Light Red
    "17": "#9B4DCA", # Lavender-Purple
    "18": "#8B0000", # Dark Red
    "19": "#228B22", # Forest Green
    "20": "#8A2BE2", # Blue Violet
    "21": "#DA70D6", # Orchid
    "22": "#BA55D3", # Medium Orchid
    "23": "#88A0A8", # Gray-Blue-Green
    "24": "#9370DB", # Medium Purple
    "25": "#7B68EE", # Medium Slate Blue
    "26": "#4169E1", # Royal Blue
    "27": "#FF8C8C", # Salmon Pink
    "28": "#FFAA80", # Light Coral
    "29": "#48D1CC", # Medium Turquoise
    "30": "#40E0D0", # Turquoise
    "31": "#00FF00", # Lime
    "32": "#7FFF00", # Chartreuse
    "33": "#ADFF2F", # Green Yellow
    "34": "#32CD32", # Lime Green
    "35": "#FF69B4", # Hot Pink
    "36": "#FFD8B8", # Peach
    "37": "#008080", # Teal
    "38": "#20B2AA", # Light Sea Green
    "39": "#00FFFF", # Cyan
    "40": "#FF00FF", # Magenta
    "41": "#4169E1", # Royal Blue
    "42": "#0000CD", # Medium Blue
    "43": "#00008B", # Dark Blue
    "44": "#8B008B", # Dark Magenta
    "45": "#FF1493", # Deep Pink
    "46": "#FF4500", # Orange Red
    "47": "#006400", # Dark Green
    "48": "#FF6347", # Tomato
    "49": "#FF7F50", # Coral
    "50": "#CD5C5C", # Indian Red
    "51": "#B22222", # Fire Brick
    "52": "#FF9CDA", # Bright Pink
    "53": "#FFB83F", # Light Orange
    "54": "#D2691E", # Chocolate
    "55": "#A0522D", # Sienna
    "56": "#800000", # Maroon
    "57": "#808080", # Gray
    "58": "#A9A9A9", # Dark Gray
    "59": "#C0C0C0", # Silver
    "60": "#9DD84A",
    "61": "#F5F5F5", # White Smoke
    "62": "#F17171", # Light Red
    "63": "#000000", # Black
    "64": "#FF8C42", # Tangerine
    "65": "#FACC15", # Golden Yellow
    "66": "#00BFFF", # Deep Sky Blue
    "67": "#E2E062", # Pale Lime
    "68": "#BADE92", # Soft Lime
    "69": "#70C1B3", # Greenish-Blue
    "70": "#41B3A3", # Turquoise
    "71": "#5EAAA8", # Gray-Green
    "72": "#72B01D", # Chartreuse
    "73": "#9CD08F", # Light Green
    "74": "#8EBA43", # Olive Green
    "75": "#FAC8C3", # Light Pink
    "76": "#E27D60", # Dark Salmon
    "77": "#C38D9E", # Mauve-Pink
    "78": "#937D64", # Light Brown
    "79": "#B1C1CC", # Light Blue-Gray
    "80": "#F9A11F", # Bright Orange-Yellow
    "81": "#4E598C", # Dark Blue-Purple
    "82": "#4B4E6D", # Dark Gray-Blue
    "83": "#8E9AAF", # Light Blue-Grey
    "84": "#C0D6DF", # Pale Blue-Grey
    "85": "#97C1A9", # Blue-Green
    "86": "#4C6E5D", # Dark Green
    "87": "#95B9C7", # Pale Blue-Green
    "88": "#C1D5E0", # Pale Gray-Blue
    "89": "#ECDB54", # Bright Yellow
    "90": "#E89B3B", # Bright Orange
    "91": "#CE5A57", # Deep Red
    "92": "#C3525A", # Dark Red
    "93": "#B85D8E", # Berry
    "94": "#7D5295", # Deep Purple
    "-1" : "#E1D9D1",
    "None" : "#E1D9D1"
}
