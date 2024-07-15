import networkx as nx

# ------------------------------------------- HELPFULLY FUNCTION ----------------------------------------------------- #
def create_name():
    """
    Generates a fixed name string.

    :return: Returns the string 'Zachary'.
    """
    return "Zachary"

# ------------------------------------------- CREATE GRAPH ----------------------------------------------------------- #
def create_graph():
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
