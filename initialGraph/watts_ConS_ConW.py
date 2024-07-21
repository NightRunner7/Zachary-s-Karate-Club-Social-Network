import random
import numpy as np
import networkx as nx

# ------------------------------------------- HELPFULLY FUNCTION ----------------------------------------------------- #
def return_name():
    """Return the basic name, which indicates the initial graph which you have been chosen"""
    return f"Watts-ConS-ConW"

def create_name(members, radical_members, k, probability):
    """
    Constructs a unique name string based on network attributes including member counts and distribution parameters.

    :param members: Number of members (nodes) in the network.
    :param radical_members: Number of radical members (nodes) to be marked as 'far-left' or 'far-right'.
    :param k: Each node is joined with its k nearest neighbors in a ring topology.
    :param probability: The probability of rewiring each edge.
    :return: A formatted string that encapsulates significant attributes of the network, suitable for identification or
             labeling.
    """
    return f"Watts-ConS-ConW-N{members}-Nrad{radical_members}-k{k}-p{probability}"

def divide_odds_and_evens(numbers):
    """
    Divide a list of numbers into two separate lists based on odd and even values.

    Parameters:
    numbers (list of int): A list of integers to be divided into odd and even numbers.

    Returns:
    tuple of two lists (list of int, list of int):
        - First list contains all odd numbers from the input.
        - Second list contains all even numbers from the input.
    """
    odd_numbers = []  # List to hold odd numbers
    even_numbers = []  # List to hold even numbers

    # Iterate through each number in the input list
    for number in numbers:
        if number % 2 == 0:
            even_numbers.append(number)  # Add even number to the even_numbers list
        else:
            odd_numbers.append(number)  # Add odd number to the odd_numbers list

    return odd_numbers, even_numbers  # Return both lists as a tuple

# ------------------------------------------- CREATE GRAPH ----------------------------------------------------------- #
def create_graph(members, radical_members, k, probability, set_affiliation_choice=True):
    """
    Create a Watts-Strogatz graph with custom edge weights and initialize the network with political views.

    :param members: Number of members (nodes) in the network.
    :param radical_members: Number of radical members (nodes) to be marked as 'far-left' or 'far-right'.
    :param k: Each node is joined with its k nearest neighbors in a ring topology.
    :param probability: The probability of rewiring each edge.
    :param set_affiliation_choice: Flag to differentiate two different ways of selecting radicals.
    :return: A NetworkX graph with initialized node attributes and edge weights.
    """
    # --- Probability of being radical
    prob_radical_left = (radical_members / members) / 2
    prob_radical_right = (radical_members / members) / 2
    prob_unaffiliated = 1 - prob_radical_left - prob_radical_right

    # --- Create a Watts-Strogatz small-world network
    network = nx.watts_strogatz_graph(members, k, probability)

    # --- Assign initial states and affiliations
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
                network.nodes[node]['state'] = 0.5
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
                network.nodes[node]['state'] = 0.5

    # --- Initialize edge weights
    for edge, (i, j) in enumerate(network.edges):
        network.edges[i, j]['weight'] = 0.5

    return network
