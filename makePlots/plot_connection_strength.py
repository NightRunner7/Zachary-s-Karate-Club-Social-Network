import matplotlib.pyplot as plt


def plot_connection_strength(network_dynamics, output_main, log_scale=False):
    """
    Plots the connection strength of a network over time.

    Args:
        network_dynamics (DataFrame): DataFrame containing 'time_str' and 'connection_str' columns.
        output_main (str): Directory path to save the plot.
        log_scale (bool): Apply logarithmic scale to the y-axis.
    """
    plt.figure(figsize=(10, 6))

    # Prepare data
    time = network_dynamics['time_arr']
    strength = network_dynamics['connection_str']

    # Plotting
    plt.plot(time, strength, marker='o', linestyle='-', color='royalblue', label='Connection Strength')

    # Enhancing plot aesthetics
    plt.title('Connection Strength of Division Over Time', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Connection Strength', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Apply logarithmic scale if required
    if log_scale:
        plt.yscale('log')
        plt.ylabel('Connection Strength (log scale)', fontsize=12)

    # Save and close the plot
    plt.savefig(f"{output_main}/connection_strength_over_time.png")
    plt.close()
