import seaborn as sns
import matplotlib.pyplot as plt


def plot_closeness_centrality_heatmap(closeness_matrix, output_main):
    """
    Plots a heatmap for the closeness centrality evolution over time.

    Args:
        closeness_matrix (DataFrame or 2D numpy array): The matrix representing closeness centrality values.
        output_main (str): Directory path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    # Capture the return of sns.heatmap which is an AxesSubplot object
    sns.heatmap(closeness_matrix, cmap="viridis", cbar_kws={'label': 'Closeness Centrality'})
    plt.title("Closeness Centrality Evolution Over Time", fontsize=14)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Node Index", fontsize=12)
    plt.xticks(fontsize=10)  # Adjust depending on the density of labels desired
    plt.yticks(fontsize=10)  # Adjust depending on the density of labels desired

    plt.tight_layout()  # Adjust layout to prevent overlap of titles/labels
    plt.savefig(f"{output_main}/closeness_centrality_over_time.png")
    plt.close()


def plot_eigenvector_centrality_heatmap(eigenvector_matrix, output_main):
    """
    Plots a heatmap for the eigenvector centrality evolution over time.

    Args:
        eigenvector_matrix (DataFrame or 2D numpy array): The matrix representing eigenvector centrality values.
        output_main (str): Directory path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(eigenvector_matrix, cmap="viridis", cbar_kws={'label': 'Eigenvector Centrality'})
    plt.title("Eigenvector Centrality Evolution Over Time", fontsize=14)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Node Index", fontsize=12)
    plt.xticks(fontsize=10)  # Adjust depending on the density of labels desired
    plt.yticks(fontsize=10)  # Adjust depending on the density of labels desired
    plt.tight_layout()  # Adjust layout to prevent overlap of titles/labels
    plt.savefig(f"{output_main}/eigenvector_centrality_over_time.png")
    plt.close()
