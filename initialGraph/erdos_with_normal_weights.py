import numpy as np
import networkx as nx

# ------------------------------------------- HELPFULLY FUNCTION ----------------------------------------------------- #
def normalize_array(arr):
    """
    Normalize the input array to have values in the range [0, 1].

    :param arr: Input array to be normalized.
    :return: Normalized array with values in the range [0, 1].
    """
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Normalize the array
    nor_value = (arr - min_val) / (max_val - min_val)

    # Replace any zero values with 1e-5
    nor_value[nor_value == 0] = 1e-5

    # Replace any 1 values with 1 - 1e-5
    nor_value[nor_value == 1] = 1 - 1e-5

    return nor_value

# ------------------------------------------- CREATE GRAPH ----------------------------------------------------------- #
def create_erdos_basic_graph(members, radical_members, probability):
    """
    Create an Erdos Renyi graph with custom edge weights and initialize the network with political views.

    :param members: Number of members (nodes) in the network.
    :param radical_members: Number of radical members (nodes) to be marked as 'far-left' or 'far-right'.
    :param probability:
    :return: A NetworkX graph with initialized node attributes and edge weights.
    """
    # --- Create a Erdos Renyi network
    network = nx.erdos_renyi_graph(members, probability)

    # Get the number of edges in the network
    num_edges = network.number_of_edges()

    # --- Randomize the strength of connections (edge weights) between members
    # weights_list = np.array([0.5] * num_edges)
    weights_list = np.random.normal(0.5, 0.1, num_edges)

    # Apply normalization to ensure weights are in range [0, 1]
    weights_list_nor = normalize_array(weights_list)

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
        network.edges[i, j]['weight'] = weights_list_nor[edge]

    return network
