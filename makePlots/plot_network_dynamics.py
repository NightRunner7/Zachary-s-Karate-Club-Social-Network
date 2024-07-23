import matplotlib.pyplot as plt

def plot_network_dynamics(network_dynamics, output_main):
    """
    Creates a 2x2 subplot layout for various network metrics.

    Args:
        network_dynamics (DataFrame): DataFrame containing all the network dynamics data.
        output_main (str): Directory path to save the plot.
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Adjust size as needed

    # Average Shortest Path Length
    axs[0, 0].plot(network_dynamics['time_dyn'], network_dynamics['average_path_length'],
                   marker='o', linestyle='-', color='blue')
    axs[0, 0].set_title('Average Shortest Path Length')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Path Length')

    # Diameter
    axs[0, 1].plot(network_dynamics['time_dyn'], network_dynamics['diameter'],
                   marker='o', linestyle='-', color='red')
    axs[0, 1].set_title('Diameter')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Diameter')

    # Average Clustering Coefficient
    axs[1, 0].plot(network_dynamics['time_dyn'], network_dynamics['average_clustering'],
                   marker='o', linestyle='-', color='green')
    axs[1, 0].set_title('Average Clustering Coefficient')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Clustering Coefficient')

    # Adjust layout
    fig.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f"{output_main}/network_dynamics.png")
    plt.close()
