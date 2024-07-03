import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import norm

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

# ########################################### CREATE GRAPH ########################################################### #
# --- ZACHARY CLUB
def create_zahary_club_graph():
    """
    This function initializes and returns a modified version of the Zachary Karate Club graph.
    The graph is loaded from the NetworkX library, and the node and edge attributes are modified
    as per the given specifications.

    :return: A NetworkX graph object representing the modified Zachary Karate Club graph.
    """
    # Load the Karate Club graph
    network = nx.karate_club_graph()

    # Fix a mistake in the networkx graph where node 8 should belong to the 'Officer' club
    network.nodes[8]['club'] = 'Officer'

    # Initialize node states and attributes
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

    # Normalize edge weights
    for i, j in network.edges:
        network.edges[i, j]['weight'] = network.edges[i, j]['weight'] / 10

    return network

# ------------------------------------------- Barabási-Albert graph -------------------------------------------------- #
# --- NORMAL WEIGHTED GRAPH
def create_normal_weighted_graph(members, radical_members, popular_members, mean=0.5, std_dev=0.1):
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

    print("len(weights_list_nor):", len(weights_list_nor))

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

def create_normal_weighted_graph_v2(members, radical_members, popular_members, mean=0.5, std_dev=0.1):
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


# --- CONSTANT WEIGHTED GRAPH
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

# ------------------------------------------- Erdos Renyi graph ------------------------------------------------------ #
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

def histogram_normal_states_graph(network, mean_val=0.5, std_dev_val=0.1, output_path='.', file_name='histogram'):
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
    states = [data['state'] for _, data in networkTwitter.nodes(data=True)]

    # Plot
    plt.figure(figsize=(8, 6))
    count, bins, ignored = plt.hist(states, bins=30, density=True, alpha=0.75, color='b')

    # Calculate the PDF of the normal distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_val, std_dev_val)

    # Plot the PDF
    plt.plot(x, p, 'r', linewidth=2)
    title = "Fit results: mean = %.2f,  std_dev = %.2f" % (mean_val, std_dev_val)
    plt.title(title)
    plt.xlabel('states')
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
