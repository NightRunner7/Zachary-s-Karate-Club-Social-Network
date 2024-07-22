import random
import numpy as np

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

# ---------------------------------------- ASSIGN AFFILIATIONS ------------------------------------------------------- #
def assign_random_affiliations(network, prob_radical_left, prob_radical_right, prob_unaffiliated, state_list):
    """
    Randomly assign political affiliations and states to nodes in the network.
    """
    counter = 0  # Counter for non-radical member state assignment
    for node in network.nodes():
        affiliation = np.random.choice(['far-left', 'far-right', None],
                                       p=[prob_radical_left, prob_radical_right, prob_unaffiliated])
        network.nodes[node]['affirmation'] = affiliation
        if affiliation == 'far-left':
            network.nodes[node]['state'] = 0.0
        elif affiliation == 'far-right':
            network.nodes[node]['state'] = 1.0
        else:
            network.nodes[node]['state'] = state_list[counter]
            counter += 1

def assign_deterministic_affiliations(network, radical_members, state_list):
    """
    Deterministically assign political affiliations to a specific subset of nodes.
    """
    all_members = list(range(network.number_of_nodes()))
    radical_nodes = random.sample(all_members, radical_members)
    left_nodes, right_nodes = divide_odds_and_evens(radical_nodes)
    counter = 0  # Counter for non-radical member state assignment

    for node in network.nodes():
        if node in left_nodes:
            network.nodes[node]['affirmation'] = 'far-left'
            network.nodes[node]['state'] = 0.0
        elif node in right_nodes:
            network.nodes[node]['affirmation'] = 'far-right'
            network.nodes[node]['state'] = 1.0
        else:
            network.nodes[node]['affirmation'] = None
            network.nodes[node]['state'] = state_list[counter]
            counter += 1

def setup_node_states_and_affiliations(network, members, radical_members, mean, std_dev, set_affiliation_choice):
    """
    Set up node states and political affiliations within the network based on specified or default parameters.

    Args:
        network (networkx.Graph): Graph
        members (int): Number of members (nodes) in the network.
        radical_members (int): Number of radical members (nodes) to be marked as 'far-left' or 'far-right'.
        mean (float): Mean value for the normal distribution used to randomize edge weights.
        std_dev (float): Standard deviation for the normal distribution used to randomize edge weights.
        set_affiliation_choice (bool): Flag to differentiate two different ways of selecting radicals.

    Returns:
        networkx.Graph: A configured graph with nodes and edges initialized according to the simulation parameters.

    Notes:
        - 'set_affiliation_choice': If True, affiliations are randomly assigned based on probabilities.
          If False, affiliations are deterministically assigned to specific nodes.
    """
    # Normalize state values
    state_list = np.clip(np.random.normal(mean, std_dev, members - radical_members), 0, 1)

    if set_affiliation_choice:
        # Calculate probabilities for affiliations
        prob_radical_left = (radical_members / members) / 2
        prob_radical_right = (radical_members / members) / 2
        prob_unaffiliated = 1 - (prob_radical_left + prob_radical_right)

        assign_random_affiliations(network, prob_radical_left, prob_radical_right, prob_unaffiliated, state_list)
    else:
        assign_deterministic_affiliations(network, radical_members, state_list)
