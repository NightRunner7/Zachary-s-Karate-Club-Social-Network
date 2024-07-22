import numpy as np
import networkx as nx
from .assistantFunctions import setup_node_states_and_affiliations
# ------------------------------------------- RETURN NAME ------------------------------------------------------------ #
def return_name():
    """Return the basic name, which indicates the initial graph which you have been chosen"""
    return f"Watts-NS-UW"

def create_name(members=None, radical_members=None, k=None, probability=None, mean=None, std_dev=None, sim_config=None):
    """
    Constructs a unique name string for a network configuration based on either direct network attributes
    or a configuration dictionary. This function allows for flexibility depending on how data is passed to it.

    Args:
        members (int, optional): Number of members (nodes) in the network.
        radical_members (int, optional): Number of radical members (nodes).
        k (int, optional): Number of nearest neighbors each node is connected to in a ring topology.
        probability (float, optional): Probability of rewiring each edge.
        mean (float, optional): Mean value of the normal distribution for edge weights.
        std_dev (float, optional): Standard deviation of the normal distribution for edge weights.
        sim_config (dict, optional): Configuration dictionary containing all the above parameters. If provided,
                                     it overrides individual parameters.

    Returns:
        str: A formatted string that encapsulates significant attributes of the network, suitable for identification or
             labeling.

    Examples:
        Using direct parameters:
            name = create_name(100, 10, 5, 0.1, 0.5, 0.1)
        Using a configuration dictionary:
            config = {'members': 100, 'radical_members': 10, 'k': 5, 'p': 0.1, 'mean': 0.5, 'std_dev': 0.1}
            name = create_name(sim_config=config)
    """
    # Check if sim_config is provided, if so, override other parameters
    if sim_config:
        members = sim_config.get('members', members)
        radical_members = sim_config.get('radical_members', radical_members)
        k = sim_config.get('k', k)
        probability = sim_config.get('p', probability)
        mean = sim_config.get('mean', mean)
        std_dev = sim_config.get('std_dev', std_dev)

    return f"Watts-NS-UW-N{members}-Nrad{radical_members}-k{k}-p{probability}-mean{mean}-std{std_dev}"

# ------------------------------------------- CREATE GRAPH ----------------------------------------------------------- #
def create_graph(members=None, radical_members=None, k=None, probability=None,
                 mean=0.5, std_dev=0.05,
                 set_affiliation_choice=True,
                 sim_config=None):
    """
    Generates a Watts-Strogatz small-world network graph with specified properties and initializes node states and
    affiliations based on political views. This function allows for dynamic graph generation with custom edge weights
    and node attributes based on provided or configured parameters.

    Args:
        members (int): Number of members (nodes) in the network.
        radical_members (int): Number of radical members (nodes) to be marked as 'far-left' or 'far-right'.
        k (int): Each node is joined with its k nearest neighbors in a ring topology.
        probability (float): The probability of rewiring each edge.
        mean (float): Mean value for the normal distribution used to randomize edge weights.
        std_dev (float): Standard deviation for the normal distribution used to randomize edge weights.
        set_affiliation_choice (bool): Flag to differentiate two different ways of selecting radicals.
        sim_config (dict, optional): Configuration dictionary for simulation parameters. If provided, it overrides
                                     individual parameters.

    Returns:
        networkx.Graph: A configured graph with nodes and edges initialized according to the simulation parameters.

    Notes:
        - 'set_affiliation_choice': If True, affiliations are randomly assigned based on probabilities.
          If False, affiliations are deterministically assigned to specific nodes.
        - The function uses the numpy and random libraries for random number generation and array manipulations,
          and the networkx library for creating and managing the graph.
    """
    # Apply configuration from sim_config if provided
    if sim_config:
        members = sim_config.get('members', members)
        radical_members = sim_config.get('radical_members', radical_members)
        k = sim_config.get('k', k)
        probability = sim_config.get('p', probability)
        mean = sim_config.get('mean', mean)
        std_dev = sim_config.get('std_dev', std_dev)

    # Initialize the graph using the Watts-Strogatz model
    network = nx.watts_strogatz_graph(members, k, probability)

    # Setup node states and affiliations
    setup_node_states_and_affiliations(network, members, radical_members, mean, std_dev, set_affiliation_choice)

    # Initialize random edge weights within a specific range
    for i, j in network.edges:
        network.edges[i, j]['weight'] = np.random.uniform(0.25, 0.75)

    return network
