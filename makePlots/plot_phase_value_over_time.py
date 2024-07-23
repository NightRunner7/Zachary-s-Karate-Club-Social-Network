import matplotlib.pyplot as plt


def plot_phase_value_over_time(network_dynamics, output_main):
    """
    Plots the phase value of a network over time.

    Args:
        network_dynamics (DataFrame): DataFrame containing 'time_step_arr' and 'phase_arr' columns.
        output_main (str): Directory path to save the plot.
    """
    plt.figure(figsize=(10, 6))

    # Prepare data
    time = network_dynamics['time_arr']
    phase_values = network_dynamics['phase_arr']

    # Plotting
    plt.plot(time, phase_values, marker='o', linestyle='-', color='blue', label='Phase Value')

    # Enhancing plot aesthetics
    plt.title('Phase Value Over Time', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Phase Value', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Save and close the plot
    plt.savefig(f"{output_main}/phase_value_over_time.png")
    plt.close()
