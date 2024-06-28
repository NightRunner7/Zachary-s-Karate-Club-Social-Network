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
import itertools
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm
from networkx.algorithms.community import girvan_newman

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
    :param file_name:
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

def draw_graph_spectral_with_communities(graph: nx.Graph,
                                         output_path: str, step: str,
                                         file_name: str = 'Twitter-network-in-step') -> None:
    """
    Draw the graph with node states and edge weights, highlighting communities, and save it as an image.
    Here we'll use a specific layout: `spectral_layout`. NOT WORKING !!!

    :param graph: The NetworkX graph to draw
    :param output_path: The directory where the image will be saved
    :param step: The current step of the evolution (used in the filename)
    :param file_name: Base name for the output file
    """
    pos = nx.spectral_layout(graph)
    node_colors = [graph.nodes[i]['state'] for i in graph.nodes]
    edge_weights = [graph.edges[i, j]['weight'] for i, j in graph.edges]

    # Detect communities using Girvan-Newman algorithm
    communities_generator = girvan_newman(graph)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    communities = sorted(map(sorted, next_level_communities))

    # Assign colors based on communities
    community_map = {}
    for community_index, community in enumerate(communities):
        for node in community:
            community_map[node] = community_index

    node_colors = [community_map[node] for node in graph.nodes]

    plt.figure(figsize=(12, 12))

    nx.draw(graph, pos,
            cmap=plt.get_cmap('cool'), vmin=0, vmax=len(communities), with_labels=False,
            node_color=node_colors, node_size=50,
            edge_color=edge_weights, edge_cmap=plt.get_cmap('binary'), edge_vmin=0, edge_vmax=1, alpha=0.5)

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

# ------------------------------------------- CREATE GRAPH ----------------------------------------------------------- #
def create_zahary_club_graph():
    """
    :return:
    """
    # Load the Karate Club graph
    network = nx.karate_club_graph()
    # I find a mistake in the `networkx` <<< !!!
    network.nodes[8]['club'] = 'Officer'

    # Initialize node states
    for node in network.nodes:
        if node == 0:
            network.nodes[node]['state'] = 1.0  # Mr. Hi
            network.nodes[node]['affirmation'] = 'far-right'  # Mr. Hi
        elif node == 33:
            network.nodes[node]['state'] = 0.0  # John A.
            network.nodes[node]['affirmation'] = 'far-left'  # John A.
        else:
            network.nodes[node]['state'] = 0.5  # Everyone else
            network.nodes[node]['affirmation'] = None

    # deal with weights
    for i, j in network.edges:
        network.edges[i, j]['weight'] = network.edges[i, j]['weight'] / 10

    return network

def create_normal_weighted_graph(members, radical_members, mean=0.5, std_dev=0.1):
    """
    Create a Barabási-Albert graph with custom edge weights and initialize the network with political views.

    :param members: Number of members (nodes) in the network.
    :param radical_members: Number of radical members (nodes) to be marked as 'far-left' or 'far-right'.
    :param mean: Mean value for the normal distribution used to randomize edge weights.
    :param std_dev: Standard deviation for the normal distribution used to randomize edge weights.
    :return: A NetworkX graph with initialized node attributes and edge weights.
    """
    # --- Create a Barabási-Albert network
    network = nx.barabasi_albert_graph(members, radical_members)

    # Get the number of edges in the network
    num_edges = network.number_of_edges()

    # --- Randomize the strength of connections (edge weights) between members
    weights_list = np.random.normal(mean, std_dev, num_edges)

    # --- Determine the political affiliation of radical members
    # Create a list indicating the radical left (0) and radical right (1) members
    choose_radical_left = [0, 1] * (radical_members // 2)
    np.random.shuffle(choose_radical_left)

    # --- Identify the most influential nodes based on degree (number of connections)
    degrees = network.degree()
    sorted_degrees = sorted(degrees, key=lambda x: x[1], reverse=True)

    # Select the most influential nodes and set them as radicals
    radical_nodes = [node for node, degree in sorted_degrees[:radical_members]]

    # --- Update the network with political views and states
    for i, node in enumerate(network.nodes):
        if node in radical_nodes:
            if choose_radical_left[radical_nodes.index(node)] == 0:
                network.nodes[node]['affirmation'] = 'far-left'
                network.nodes[node]['state'] = 0.0
            else:
                network.nodes[node]['affirmation'] = 'far-right'
                network.nodes[node]['state'] = 1.0
        else:
            # Neutral members have no political affirmation and a neutral state
            network.nodes[node]['affirmation'] = None
            network.nodes[node]['state'] = 0.5

    # --- Initialize edge weights
    for edge, (i, j) in enumerate(network.edges):
        network.edges[i, j]['weight'] = weights_list[edge]

    return network

def create_constant_weighted_graph(members, radical_members):
    """
    Create a Barabási-Albert graph with custom edge weights and initialize the network with political views.

    :param members: Number of members (nodes) in the network.
    :param radical_members: Number of radical members (nodes) to be marked as 'far-left' or 'far-right'.
    :return: A NetworkX graph with initialized node attributes and edge weights.
    """
    # --- Create a Barabási-Albert network
    network = nx.barabasi_albert_graph(members, radical_members)

    # Get the number of edges in the network
    num_edges = network.number_of_edges()

    # --- Constant the strength of connections (edge weights) between members
    weights_list = np.array([0.5] * num_edges)

    # --- Determine the political affiliation of radical members
    # Create a list indicating the radical left (0) and radical right (1) members
    choose_radical_left = [0, 1] * (radical_members // 2)
    np.random.shuffle(choose_radical_left)

    # --- Identify the most influential nodes based on degree (number of connections)
    degrees = network.degree()
    sorted_degrees = sorted(degrees, key=lambda x: x[1], reverse=True)

    # Select the most influential nodes and set them as radicals
    radical_nodes = [node for node, degree in sorted_degrees[:radical_members]]

    # --- Update the network with political views and states
    for i, node in enumerate(network.nodes):
        if node in radical_nodes:
            if choose_radical_left[radical_nodes.index(node)] == 0:
                network.nodes[node]['affirmation'] = 'far-left'
                network.nodes[node]['state'] = 0.0
            else:
                network.nodes[node]['affirmation'] = 'far-right'
                network.nodes[node]['state'] = 1.0
        else:
            # Neutral members have no political affirmation and a neutral state
            network.nodes[node]['affirmation'] = None
            network.nodes[node]['state'] = 0.5

    # --- Initialize edge weights
    for edge, (i, j) in enumerate(network.edges):
        network.edges[i, j]['weight'] = weights_list[edge]

    return network

# ------------------------------------------- HISTOGRAMS ------------------------------------------------------------- #
# connected to version of the graph from previous section
def histogram_normal_weighted_graph(network, mean_val=0.5, std_dev_val=0.1, output_path='.', file_name='histogram'):
    """
    Draw the histogram.

    :param network: The NetworkX graph to draw
    :param mean_val:
    :param std_dev_val:
    :param output_path: The directory where the image will be saved
    :param file_name:
    """
    # Extract the weights from the network
    networkTwitter = network.return_network()
    weights = [data['weight'] for _, _, data in networkTwitter.edges(data=True)]

    # Plot
    plt.figure(figsize=(8, 6))
    count, bins, ignored = plt.hist(weights, bins=30, density=True, alpha=0.75, color='b')

    # Calculate the PDF of the normal distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_val, std_dev_val)

    # Plot the PDF
    plt.plot(x, p, 'r', linewidth=2)
    title = "Fit results: mean = %.2f,  std_dev = %.2f" % (mean_val, std_dev_val)
    plt.title(title)
    plt.xlabel('weights')
    plt.ylabel('Density')
    plt.savefig(f"{output_path}/{file_name}.png")
    plt.close()

def histogram_weighted_graph(network, output_path='.', file_name='histogram'):
    """
    Draw the histogram.

    :param network: The NetworkX graph to draw
    :param output_path: The directory where the image will be saved
    :param file_name:
    """
    # Extract the weights from the network
    networkTwitter = network.return_network()
    weights = [data['weight'] for _, _, data in networkTwitter.edges(data=True)]

    # Plot
    plt.figure(figsize=(8, 6))
    count, bins, ignored = plt.hist(weights, bins=30, density=True, alpha=0.75, color='b')

    title = "Fit results"
    plt.title(title)
    plt.xlabel('weights')
    plt.ylabel('Density')
    plt.savefig(f"{output_path}/{file_name}.png")
    plt.close()

def histogram_states_graph(network, output_path='.', file_name='histogram'):
    """
    Draw the histogram of states.

    :param network: The NetworkX graph to draw
    :param output_path: The directory where the image will be saved
    :param file_name:
    """
    # Extract the weights from the network
    networkTwitter = network.return_network()
    states = [data['state'] for _, data in networkTwitter.nodes(data=True)]

    # Plot
    plt.figure(figsize=(8, 6))
    count, bins, ignored = plt.hist(states, bins=30, density=True, alpha=0.75, color='b')

    title = "Fit results"
    plt.title(title)
    plt.xlabel('states')
    plt.ylabel('Density')
    plt.savefig(f"{output_path}/{file_name}.png")
    plt.close()
