"""
auxiliaryFunctions.py

This module provides auxiliary functions for managing directories, creating GIFs, and plotting graphs and node state
evolutions.

Functions:
    - make_directory(path_name): Creates a directory if it doesn't already exist.
    - make_gif(frame_folder, name): Creates a GIF from PNG images in a specified folder.
    - draw_graph(graph, output_path, step): Draws a graph with node states and edge weights, saving it as an image.
    - plot_node_evolution(network_state, node, output_path): Plots the evolution of the state of a selected node over
        time.

Author:
    Krzysztof Szafrański

Last Modified:
    June 23, 2024
"""
import os
import glob
import pickle
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm

# ------------------------------------------- DEAL WITH MATRICES ----------------------------------------------------- #
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

# ------------------------------------------- MAKE STH --------------------------------------------------------------- #
def make_directory(path_name):
    """
    Create a directory with the given name if it doesn't already exist.

    :param path_name: The name of the directory to create
    """
    try:
        os.mkdir(path_name)
        print(f"Directory {path_name} created.")
    except FileExistsError:
        print(f"Directory {path_name} already exists.")

def make_gif(frame_folder, name):
    """
    Create a GIF from all PNG images in the specified folder.

    :param frame_folder: The folder containing the image frames
    :param name: The name of the resulting GIF file
    """
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.PNG")]
    if frames:
        frame_one = frames[0]
        frame_one.save(f"{frame_folder}/{name}.gif", format="GIF", append_images=frames,
                       save_all=True, duration=60, loop=1)
    else:
        print(f"No PNG images found in {frame_folder}")

# ------------------------------------------- DEAL WITH NETWORKS ----------------------------------------------------- #
def save_network(graph: nx.Graph,
                 output_path: str, step: str,
                 file_name: str = 'network_at_step') -> None:
    """
    Draw the graph with node states and edge weights, and save it as an image.

    :param graph: The NetworkX graph to draw
    :param output_path: The directory where the network will be saved
    :param step: The current step of the evolution (used in the filename)
    :param file_name: Base name for the output file
    """
    full_name = f'{output_path}/{file_name}_{step}'
    # Save the graph to a file
    with open(f'{full_name}.pkl', 'wb') as f:
        pickle.dump(graph, f)

# ------------------------------------------- DEAL WITH PHASES ----------------------------------------------------- #
def count_affirmations(network):
    """
    Count nodes with 'far-left' and 'far-right' affirmations in a network.

    :param network: A NetworkX graph with nodes that may have 'affirmation' attributes.
    :return: A tuple containing counts (count_far_left, count_far_right)
    """
    count_far_left = 0
    count_far_right = 0

    # Iterate over all nodes and their data in the network
    for node, data in network.nodes(data=True):
        affirmation = data.get('affirmation')  # Safely get the 'affirmation' attribute

        # Check and update counts based on the affirmation
        if affirmation == 'far-left':
            count_far_left += 1
        elif affirmation == 'far-right':
            count_far_right += 1

    return count_far_left, count_far_right

def find_the_phase(network, epsilon=0.05, neutral_width=0.4, division_threshold=0.2, wall_threshold=0.2):
    """
    Determines the phase of the system based on the distribution of state values at the end of an evolutionary
    process.

    Args:
        network (NetworkX graph): TA NetworkX graph with nodes that may have 'affirmation' attributes.
        epsilon (float): The tolerance used to classify left and right members as radicals.
        neutral_width (float): The width of the neutral zone centered at 0.5, within which members are considered
            neutral.
        division_threshold (float): The threshold ratio of radical change needed to declare a division phase.
        wall_threshold (float): The threshold ratio of neutral members needed to declare a wall phase.

    Returns:
        float: A numeric code representing different phases of the system:
            4.0 - 'nonrecognition'
            3.0 - 'domination'
            1.0 to 2.0 - 'division' (variable based on the degree of radical change)
            0.0 to 1.0 - 'wall' (variable, inversely related to the degree of neutrality)
    """
    # --- SETTING PARAMETERS OF NETWORK
    # Choose between internal state vector or an external one based on `out_of_class`
    s_vec = np.array([state for _, state in network.nodes(data='state', default='Not Available')])
    # Determine initial counts of radical members, considering external or internal data source
    N_left_init, N_right_init = count_affirmations(network)

    # Total number of members in the network
    N = len(network.nodes)
    # Total initial radical members
    N_rad = N_left_init + N_right_init

    # --- COUNTING NECESSARY MEMBERS IN DIFFERENT GROUPS
    # Count members within specific state ranges at the end of the evolution
    N_left_end = np.sum(s_vec <= epsilon)
    N_right_end = np.sum(s_vec >= 1.0 - epsilon)
    N_neutral_end = np.sum((s_vec > 0.5 - neutral_width / 2) & (s_vec < 0.5 + neutral_width / 2))

    # Calculate the change in the number of radical members from initial to final
    delta_N_rad = (N_left_end + N_right_end) - (N_left_init + N_right_init)

    # Population not initially identified as radical
    remaining_non_radical = N - N_rad

    print("N_left_end:", N_left_end, "N_right_end:", N_right_end, "N_neutral_end:", N_neutral_end)
    print("N_left_init:", N_left_init, "N_right_init:", N_right_init)
    print("remaining_non_radical:", remaining_non_radical, "N:", N, "N_rad:", N_rad)

    # --- TAKE CARE ABOUT PHASES
    # Determine the phase based on conditions involving changes in radical and neutral members
    if N_left_end == N - N_right_init or N_right_end == N - N_left_init:
        phase = 3.0  # Domination phase indicates complete shift to one radical side
    elif (delta_N_rad / remaining_non_radical) >= division_threshold:
        phase = 1.0 + (delta_N_rad / remaining_non_radical)  # Division phase indicates significant radicalization
    elif (N_neutral_end / remaining_non_radical) >= wall_threshold:
        phase = 1.0 - (N_neutral_end / remaining_non_radical)  # Wall phase indicates a significant neutral buffer
    else:
        phase = 4.0  # Nonrecognition phase indicates no significant change or pattern

    return phase

# ------------------------------------------- PLOTS ------------------------------------------------------------------ #
# --- DRAW SPRING
def draw_graph(graph: nx.Graph,
               output_path: str, step: str,
               file_name: str = 'Zahary-evolution-in-step') -> None:
    """
    Draw the graph with node states and edge weights, and save it as an image.

    :param graph: The NetworkX graph to draw
    :param output_path: The directory where the image will be saved
    :param step: The current step of the evolution (used in the filename)
    :param file_name: Base name for the output file
    """
    node_colors = [graph.nodes[i]['state'] for i in graph.nodes]
    edge_weights = [graph.edges[i, j]['weight'] for i, j in graph.edges]

    nx.draw_spring(graph,
                   cmap=mpl.colormaps['cool'], vmin=0, vmax=1, with_labels=True,
                   node_color=node_colors,
                   edge_cmap=mpl.colormaps['binary'], edge_vmin=0, edge_vmax=1,
                   edge_color=edge_weights)

    plt.savefig(f"{output_path}/{file_name}-{step}.png")
    plt.close()

# --- SPRING GRAPH
def draw_graph_spring(graph: nx.Graph, positions,
                      output_path: str, step: str,
                      file_name: str = 'Twitter-network-in-step') -> None:
    """
    Draw the graph with node states and edge weights, and save it as an image.

    :param graph: The NetworkX graph to draw
    :param output_path: The directory where the image will be saved
    :param step: The current step of the evolution (used in the filename)
    :param file_name: Base name for the output file
    """
    node_colors = [graph.nodes[i]['state'] for i in graph.nodes]
    edge_weights = [graph.edges[i, j]['weight'] for i, j in graph.edges]

    plt.figure(figsize=(12, 12))  # Increase the figure size
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    # nx.draw_spring(graph,
    #                cmap=mpl.colormaps['cool'], vmin=0, vmax=1,
    #                node_color=node_colors, node_size=10,
    #                edge_cmap=mpl.colormaps['binary'], edge_vmin=0, edge_vmax=1,
    #                edge_color=edge_weights)

    nx.draw(graph, pos=positions, cmap=plt.get_cmap('cool'), vmin=0, vmax=1,
            node_color=node_colors, node_size=10,
            edge_cmap=plt.get_cmap('binary'), edge_vmin=0, edge_vmax=1,
            edge_color=edge_weights, width=0.4)

    plt.savefig(f"{output_path}/{file_name}_{step}.png")
    plt.close()

# --- SPECTRAL LAYOUT
def draw_graph_spectral(graph: nx.Graph,
                        output_path: str, step: str,
                        file_name: str = 'Twitter-network-in-step') -> None:
    """
    Draw the graph with node states and edge weights, and save it as an image. Here we'll use
    a specific layout: `spectral_layout`. Use it for large graph.

    :param graph: The NetworkX graph to draw
    :param output_path: The directory where the image will be saved
    :param step: The current step of the evolution (used in the filename)
    :param file_name: Base name for the output file
    """
    pos = nx.spectral_layout(graph)
    node_colors = [graph.nodes[i]['state'] for i in graph.nodes]
    edge_weights = [graph.edges[i, j]['weight'] for i, j in graph.edges]

    plt.figure(figsize=(12, 12))  # Increase the figure size

    nx.draw(graph, pos,
            cmap=plt.get_cmap('cool'), vmin=0, vmax=1, with_labels=False,  # Remove labels
            node_color=node_colors, node_size=50,  # Adjust node size
            edge_color=edge_weights, edge_cmap=plt.get_cmap('binary'),
            edge_vmin=0, edge_vmax=1, alpha=0.5)  # Adjust edge transparency

    plt.savefig(f"{output_path}/{file_name}-{step}.png")
    plt.close()

# --- KAMADA KAWAI LAYOUT
def draw_graph_kamada_kawai(graph: nx.Graph,
                            output_path: str, step: str,
                            file_name: str = 'Twitter-network-in-step') -> None:
    """
    Draw the graph with node states and edge weights, and save it as an image. Here we'll use
    a specific layout: `kamada_kawai_layout`.

    :param graph: The NetworkX graph to draw
    :param output_path: The directory where the image will be saved
    :param step: The current step of the evolution (used in the filename)
    :param file_name: Base name for the output file
    """
    pos = nx.kamada_kawai_layout(graph)
    node_colors = [graph.nodes[i]['state'] for i in graph.nodes]
    edge_weights = [graph.edges[i, j]['weight'] for i, j in graph.edges]

    plt.figure(figsize=(12, 12))  # Increase the figure size

    nx.draw(graph, pos,
            cmap=plt.get_cmap('cool'), vmin=0, vmax=1, with_labels=False,  # Remove labels
            node_color=node_colors, node_size=50,  # Adjust node size
            edge_color=edge_weights, edge_cmap=plt.get_cmap('binary'),
            edge_vmin=0, edge_vmax=1, alpha=0.5)  # Adjust edge transparency

    plt.savefig(f"{output_path}/{file_name}-{step}.png")
    plt.close()

# --- EXTRA PLOTS
def plot_node_evolution(network_state, node, output_path):
    """
    Plot the evolution of the state of a selected node over time.

    :param network_state: Dictionary where keys are node indices and values are arrays of states over time
    :param node: The node index to plot the state evolution for
    :param output_path: The directory where the plot will be saved
    """
    if node not in network_state:
        print(f"Node {node} not found in the network state.")
        return

    # Extract the states of the selected node
    node_states = network_state[node]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(node_states, marker='o', linestyle='-', color='b')
    plt.title(f'Evolution of Node {node} State Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('State')
    plt.grid(True)

    # Save the plot
    plt.savefig(f"{output_path}/node_{node:02d}_evolution.png")
    plt.close()

# ------------------------------------------- HISTOGRAMS ------------------------------------------------------------- #
def histogram_weights(network, plot_fit=False, mean_val=0.5, std_dev_val=0.1, output_path='.', file_name='histogram'):
    """
    Draw the histogram of weights from a NetworkX graph. Optionally fit a normal distribution.

    :param network: The NetworkX graph to draw.
    :param plot_fit: Boolean to decide whether to plot normal distribution fit.
    :param mean_val: Mean value for the normal distribution used to randomize states.
    :param std_dev_val: Standard deviation for the normal distribution used to randomize states.
    :param output_path: The directory where the image will be saved.
    :param file_name: Base name for the output file.
    """
    # Extract the weights from the network
    weights = [data['weight'] for _, _, data in network.edges(data=True)]

    # Plot histogram
    plt.figure(figsize=(8, 6))
    _, bins, _ = plt.hist(weights, bins=30, density=True, alpha=0.75, color='b')

    if plot_fit:
        # Calculate the PDF of the normal distribution
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean_val, std_dev_val)

        # Plot the PDF
        plt.plot(x, p, 'r', linewidth=2)
        title = "Fit results: mean = %.2f, std_dev = %.2f" % (mean_val, std_dev_val)
    else:
        title = "Histogram of Weights"

    plt.title(title)
    plt.xlabel('weights')
    plt.ylabel('Density')
    plt.savefig(f"{output_path}/{file_name}.png")
    plt.close()

def histogram_states(network, plot_fit=False, mean_val=0.5, std_dev_val=0.1, output_path='.', file_name='histogram'):
    """
    Draw the histogram of states from a NetworkX graph. Optionally fit a normal distribution.

    :param network: The NetworkX graph to draw.
    :param plot_fit: Boolean to decide whether to plot normal distribution fit.
    :param mean_val: Mean value for the normal distribution used to randomize states.
    :param std_dev_val: Standard deviation for the normal distribution used to randomize states.
    :param output_path: The directory where the image will be saved.
    :param file_name: Base name for the output file.
    """
    # Extract the states from the network
    states = [data['state'] for _, data in network.nodes(data=True)]

    # Plot histogram
    plt.figure(figsize=(8, 6))
    _, bins, _ = plt.hist(states, bins=30, density=True, alpha=0.75, color='b')

    if plot_fit:
        # Calculate the PDF of the normal distribution
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean_val, std_dev_val)

        # Plot the PDF
        plt.plot(x, p, 'r', linewidth=2)
        title = "Fit results: mean = %.2f, std_dev = %.2f" % (mean_val, std_dev_val)
    else:
        title = "Histogram of States"

    plt.title(title)
    plt.xlabel('states')
    plt.ylabel('Density')
    plt.savefig(f"{output_path}/{file_name}.png")
    plt.close()

def histogram_degrees(network, output_path='.', file_name='histogram'):
    """
    Draw the histogram of degrees from a NetworkX graph.

    :param network: The NetworkX graph to draw.
    :param output_path: The directory where the image will be saved.
    :param file_name: Base name for the output file.
    """
    # Get the degree sequence sorted in descending order
    degree_sequence = sorted((d for n, d in network.degree()), reverse=True)

    # Create a new figure with a specified size
    plt.figure(figsize=(8, 6))

    # Generate the histogram data where 'np.unique' returns unique degrees and their counts
    degrees, counts = np.unique(degree_sequence, return_counts=True)

    # Create a bar plot
    plt.bar(degrees, counts)

    # Set the title and labels
    plt.title("Degree Histogram")
    plt.xlabel("Degree")
    plt.ylabel("# of Nodes")

    # Save the figure to the specified path
    plt.savefig(f"{output_path}/{file_name}.png")

    # Close the plot to free up memory
    plt.close()
