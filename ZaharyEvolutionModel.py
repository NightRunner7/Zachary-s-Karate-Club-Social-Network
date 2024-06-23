"""
This module implements the Zahary Evolution Model using NetworkX and NumPy.

Classes:
    - ZaharyEvolutionModel: A class to model the evolution of the Zachary Karate Club network.

Author:
    Krzysztof Szafrański

Last Modified:
    June 23, 2024
"""
import networkx as nx
import numpy as np


class ZaharyEvolutionModel:
    def __init__(self, D=5, beta=10, dt=0.01):
        """
        Initialize the model with given parameters and set up the initial network.

        :param D: Diffusion coefficient
        :param beta: Coupling parameter
        :param dt: Time step
        """
        # Load the Karate Club graph
        network = nx.karate_club_graph()
        # I find a mistake in the `networkx` <<< !!!
        network.nodes[8]['club'] = 'Officer'

        # Initialize node states
        for node in network.nodes:
            if node == 0:
                network.nodes[node]['state'] = 1  # Mr. Hi
            elif node == 33:
                network.nodes[node]['state'] = 0  # John A.
            else:
                network.nodes[node]['state'] = 0.5  # Everyone else

        # Initialize edge weights
        for i, j in network.edges:
            network.edges[i, j]['weight'] = 0.5

        # networks
        self.G = network  # Current network
        self.old_G = network.copy()  # Copy of the network to keep previous state
        # simulation settings
        self.D = D
        self.beta = beta
        self.dt = dt

        # save: network evolution state
        self.networkState = {node: np.array([network.nodes[node]['state']]) for node in network.nodes}

    @staticmethod
    def f(c_i, c_j):
        """
        Compute the interaction function based on node states.

        :param c_i: State of node i
        :param c_j: State of node j
        :return: Interaction value
        """
        x = abs(c_i - c_j)
        return (x - 0.25) ** 3

    # ----------------------------------------- UPDATE FUNCTION ------------------------------------------------------ #
    def update_weights(self):
        """Update the weights of the edges based on the states of the connected nodes."""
        for i, j in self.G.edges:
            weight = self.G.edges[i, j]['weight']
            state_i = self.G.nodes[i]['state']
            state_j = self.G.nodes[j]['state']
            dwij_dt = -self.beta * weight * (1 - weight) * self.f(state_i, state_j)
            self.G.edges[i, j]['weight'] += self.dt * dwij_dt

    def update_states(self):
        """Update the states of the nodes based on the weights of the connecting edges."""
        new_states = {}
        for i in self.G.nodes:
            if i == 0:
                new_states[i] = 1  # Mr. Hi remains the same
            elif i == 33:
                new_states[i] = 0  # John A. remains the same
            else:
                delta_state = 0  # the part under the sum
                for j in self.G.adj[i]:
                    weight = self.G.edges[i, j]['weight']
                    state_i = self.old_G.nodes[i]['state']
                    state_j = self.old_G.nodes[j]['state']
                    delta_state += weight * (state_j - state_i)

                new_states[i] = self.old_G.nodes[i]['state'] + self.dt * self.D * delta_state

        for i in new_states:
            # update network
            self.G.nodes[i]['state'] = new_states[i]

        # we have to update old network as well
        self.old_G = self.G.copy()

    # ----------------------------------------- EXTRA SETTINGS ------------------------------------------------------- #
    def save_network_state(self):
        """Save the current state of each node in the network to the networkState dictionary."""
        for node in self.G.nodes:
            state = self.G.nodes[node]['state']
            self.networkState[node] = np.append(self.networkState[node], state)

    def change_init_network_publication(self):
        """
        Initialize the network based on the publication:
        "An Information Flow Model for Conflict and Fission in Small Groups".

        - Mr. Hi (node 0) is initialized with a state of 1.
        - John A. (node 33) is initialized with a state of 0.
        - All other nodes are initialized with a state of 0.5.
        - Edge weights are scaled down by a factor of 10.

        This function sets up the current network and saves the initial state.
        """
        # Load the Karate Club graph
        network = nx.karate_club_graph()
        # I find a mistake in the `networkx` <<< !!!
        network.nodes[8]['club'] = 'Officer'

        # Initialize node states
        for node in network.nodes:
            if node == 0:
                network.nodes[node]['state'] = 1  # Mr. Hi
            elif node == 33:
                network.nodes[node]['state'] = 0  # John A.
            else:
                network.nodes[node]['state'] = 0.5  # Everyone else

        # Initialize edge weights
        for i, j in network.edges:
            network.edges[i, j]['weight'] = network.edges[i, j]['weight'] / 10

        # Set the current network and its copy to keep the previous state
        self.G = network  # Current network
        self.old_G = network.copy()  # Copy of the network to keep previous state

        # Save the initial state of the network
        self.networkState = {node: np.array([network.nodes[node]['state']]) for node in network.nodes}

    # ----------------------------------------- DEAL WITH EVOLUTION -------------------------------------------------- #
    def evolve(self):
        """Perform one evolution step: update weights and then update states."""
        self.update_weights()
        self.update_states()

    # ----------------------------------------- RETURN --------------------------------------------------------------- #
    def return_network(self):
        """Return the current network."""
        return self.G

    def return_network_evolution_state(self):
        """Return dictionary contains evolution of state of the nodes."""
        return self.networkState
