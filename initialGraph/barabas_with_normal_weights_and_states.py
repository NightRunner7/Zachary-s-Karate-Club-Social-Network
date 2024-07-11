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
def create_graph(members, radical_members, popular_members, mean=0.5, std_dev=0.1):
    """
    Create a Barabási-Albert graph with custom edge weights and initialize the network with political views.

    :param members: Number of members (nodes) in the network.
    :param radical_members: Number of radical members (nodes) to be marked as 'far-left' or 'far-right'.
    :param popular_members: Number of members, which are popular (hubs in the network).
        `popular_members >=radical_members`.
    :param mean: Mean value for the normal distribution used to randomize edge weights.
    :param std_dev: Standard deviation for the normal distribution used to randomize edge weights.
    :return: A NetworkX graph with initialized node attributes and edge weights.
    """
    # --- Create a Barabási-Albert network
    network = nx.barabasi_albert_graph(members, popular_members)

    # Get the number of edges in the network
    num_edges = network.number_of_edges()

    # --- Randomize the strength of connections (edge weights) between members
    weights_list = np.random.normal(mean, std_dev, num_edges)

    # Apply normalization to ensure weights are in range [0, 1]
    weights_list_nor = normalize_array(weights_list)

    # --- Randomize the strength of connections (edge weights) between members
    non_radical = network.number_of_nodes() - radical_members
    state_list = np.random.normal(mean, std_dev, non_radical)

    # Apply normalization to ensure weights are in range [0, 1]
    state_list_nor = normalize_array(state_list)

    print("len(state_list_nor):", len(state_list_nor))

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
    counter = 0
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
            network.nodes[node]['state'] = state_list_nor[counter]
            counter += 1

    # --- Initialize edge weights
    for edge, (i, j) in enumerate(network.edges):
        network.edges[i, j]['weight'] = weights_list_nor[edge]

    return network
