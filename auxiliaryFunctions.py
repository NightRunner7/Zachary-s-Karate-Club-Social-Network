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
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from PIL import Image


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
def draw_graph(graph, output_path, step):
    """
    Draw the graph with node states and edge weights, and save it as an image.

    :param graph: The NetworkX graph to draw
    :param output_path: The directory where the image will be saved
    :param step: The current step of the evolution (used in the filename)
    """
    node_colors = [graph.nodes[i]['state'] for i in graph.nodes]
    edge_weights = [graph.edges[i, j]['weight'] for i, j in graph.edges]

    nx.draw_spring(graph,
                   cmap=mpl.colormaps['cool'], vmin=0, vmax=1, with_labels=True,
                   node_color=node_colors,
                   edge_cmap=mpl.colormaps['binary'], edge_vmin=0, edge_vmax=1,
                   edge_color=edge_weights)

    plt.savefig(f"{output_path}/Zahary-evolution-in-step-{step}.png")
    plt.close()


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
