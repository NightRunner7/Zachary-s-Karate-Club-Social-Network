import random
import numpy as np
import networkx as nx

# ------------------------------------------- HELPFULLY FUNCTION ----------------------------------------------------- #
def return_name():
    """Return the basic name, which indicates the initial graph which you have been chosen"""
    return f"Watts-NS-UW"

def create_name(members, radical_members, k, probability, mean, std_dev):
    """
    Constructs a unique name string based on network attributes including member counts and distribution parameters.

    :param members: Number of members (nodes) in the network.
    :param radical_members: Number of radical members (nodes) to be marked as 'far-left' or 'far-right'.
    :param k: Each node is joined with its k nearest neighbors in a ring topology.
    :param probability: The probability of rewiring each edge.
    :param mean: Mean value for the normal distribution used to randomize edge weights.
    :param std_dev: Standard deviation for the normal distribution used to randomize edge weights.
    :return: A formatted string that encapsulates significant attributes of the network, suitable for identification or
             labeling.
    """
    return f"Watts-NS-UW-N{members}-Nrad{radical_members}-k{k}-p{probability}-mean{mean}-std{std_dev}"

def divide_odds_and_evens(numbers):
    """
    Divide a list of numbers into two separate lists based on odd and even indexes.

    Parameters:
    numbers (list of int): A list of integers to be divided into odd and even numbers.

    Returns:
    tuple of two lists (list of int, list of int):
        - First list contains all odd numbers from the input.
        - Second list contains all even numbers from the input.
    """
    odd_numbers = []  # List to hold odd indexes
    even_numbers = []  # List to hold even indexes

    # Iterate through each number in the input list
    for index in range(0, len(numbers)):
        if index % 2 == 0:
            even_numbers.append(numbers[index])  # Add even number to the even_numbers list
        else:
            odd_numbers.append(numbers[index])  # Add odd number to the odd_numbers list

    return odd_numbers, even_numbers  # Return both lists as a tuple

# ------------------------------------------- CREATE GRAPH ----------------------------------------------------------- #
def create_graph(members, radical_members, k, probability, mean=0.5, std_dev=0.05, set_affiliation_choice=True):
    """
    Create a Watts-Strogatz graph with custom edge weights and initialize the network with political views.

    :param members: Number of members (nodes) in the network.
    :param radical_members: Number of radical members (nodes) to be marked as 'far-left' or 'far-right'.
    :param k: Each node is joined with its k nearest neighbors in a ring topology.
    :param probability: The probability of rewiring each edge.
    :param mean: Mean value for the normal distribution used to randomize edge weights.
    :param std_dev: Standard deviation for the normal distribution used to randomize edge weights.
    :param set_affiliation_choice: Flag to differentiate two different ways of selecting radicals.
    :return: A NetworkX graph with initialized node attributes and edge weights.
    """
    # --- Probability of being radical
    prob_radical_left = (radical_members / members) / 2
    prob_radical_right = (radical_members / members) / 2
    prob_unaffiliated = 1 - prob_radical_left - prob_radical_right

    # --- Create a Watts-Strogatz small-world network
    network = nx.watts_strogatz_graph(members, k, probability)

    # --- Randomize the value of states (nodes)
    non_radical = network.number_of_nodes() - radical_members
    state_list = np.random.normal(mean, std_dev, non_radical)
    state_list_nor = np.clip(state_list, 0, 1)  # Apply normalization to ensure states are in range [0, 1]

    # --- Assign initial states and affiliations
    counter = 0
    if set_affiliation_choice:
        for node in network.nodes():
            # Randomly assign affiliations with a higher chance of being unaffiliated
            affiliation = np.random.choice(['far-left', 'far-right', None], p=[prob_radical_left,
                                                                               prob_radical_right,
                                                                               prob_unaffiliated])
            network.nodes[node]['affirmation'] = affiliation

            # Set initial state based on the political affiliation
            if affiliation == 'far-left':
                network.nodes[node]['state'] = 0.0  # Far-left affiliation starts at state 0
            elif affiliation == 'far-right':
                network.nodes[node]['state'] = 1.0  # Far-right affiliation starts at state 1
            else:
                network.nodes[node]['state'] = state_list_nor[counter]
                counter += 1
    else:
        all_members = range(members)
        # Select radical_members unique random numbers from the all_members
        radical_nodes = random.sample(all_members, radical_members)
        left_nodes, right_nodes = divide_odds_and_evens(radical_nodes)

        for node in network.nodes():
            # Set initial state based on the political affiliation
            if node in left_nodes:
                network.nodes[node]['affirmation'] = 'far-left'
                network.nodes[node]['state'] = 0.0  # Far-left affiliation starts at state 0
            elif node in right_nodes:
                network.nodes[node]['affirmation'] = 'far-right'
                network.nodes[node]['state'] = 1.0  # Far-right affiliation starts at state 1
            else:
                network.nodes[node]['affirmation'] = None
                network.nodes[node]['state'] = state_list_nor[counter]
                counter += 1

    # --- Initialize edge weights
    for edge, (i, j) in enumerate(network.edges):
        network.edges[i, j]['weight'] = np.random.uniform(0.25, 0.75)  # Unaffiliated edges with random weights

    return network
