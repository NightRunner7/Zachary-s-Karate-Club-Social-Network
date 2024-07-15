import networkx as nx
import numpy as np


class TwitterRadicalizationModel:
    def __init__(self, network, D=5., beta=10., dt=0.01):
        """
        Initialize the model with given parameters and set up the initial network.

        :param network:
        :param D: Diffusion coefficient
        :param beta: Coupling parameter
        :param dt: Time step
        """
        self.D = D
        self.beta = beta
        self.dt = dt

        # Load the Karate Club graph
        self.network = network

        # Initialize node states
        self.s_vec = np.array([data['state'] for _, data in self.network.nodes(data=True)])

        # Initialize weight matrices using sparse matrix
        self.w_mat = nx.to_scipy_sparse_array(self.network, format='csr')

        # Save the initial state
        self.networkState = {node: np.array([state]) for node, state in enumerate(self.s_vec)}

    # ----------------------------------------- INTERACTIONS --------------------------------------------------------- #
    def f(self):
        """
        Compute the interaction function for all node pairs.

        :return: Interaction matrix
        """
        rows, cols = self.w_mat.nonzero()
        state_diff = np.abs(self.s_vec[rows] - self.s_vec[cols])
        f_values = (state_diff - 0.25) ** 3
        return f_values

    # ----------------------------------------- UPDATE FUNCTION ------------------------------------------------------ #
    def update_weights(self):
        """Update the weights of the edges based on the states of the connected nodes."""
        interaction = self.f()
        dwij_dt = -self.beta * self.w_mat.data * (1 - self.w_mat.data) * interaction

        # Update weights
        self.w_mat.data += self.dt * dwij_dt

    def update_states(self):
        """Update the states of the nodes based on the weights of the connecting edges."""
        # Compute the state changes using sparse matrix operations
        delta_s_vec = self.D * (self.w_mat @ self.s_vec - self.s_vec * self.w_mat.sum(axis=1))

        # Update states
        self.s_vec += self.dt * delta_s_vec

        # Ensure that radical members retain their initial states
        affirmation_to_state = {'far-left': 0.0, 'far-right': 1.0}
        for node, data in self.network.nodes(data=True):
            if data['affirmation'] in affirmation_to_state:
                self.s_vec[node] = affirmation_to_state[data['affirmation']]

    def update_networkState(self):
        """Update states of the network in the self.networkState"""
        # Save the updated states
        for node in range(self.s_vec.shape[0]):
            self.networkState[node] = np.append(self.networkState[node], self.s_vec[node])

    # ----------------------------------------- EXTRA SETTINGS ------------------------------------------------------- #
    def connection_strength_of_division(self):
        """
        Calculate the connection strength of division in the network.

        This function measures the total weight of connections between different factions
        (Mr. Hi's faction and Officer's faction) in the network. It computes the sum of the
        weights of edges that connect nodes from different factions.

        :return: The connection strength of division (float)
        """
        # Binary vectors indicating faction membership
        r_member = np.where(self.s_vec >= 0.5, 1, 0)  # 1 if far-right member, else 0
        l_member = np.where(r_member == 0, 1, 0)  # 1 if far-left member, else 0

        # Calculate internal connection weights for each faction
        r_weights = r_member @ self.w_mat @ r_member.T  # Total weight of connections within far-right's faction
        l_weights = l_member @ self.w_mat @ l_member.T  # Total weight of connections within far-left's faction

        # Calculate the total weight of all connections in the network
        sum_weights = np.sum(self.w_mat)

        # Calculate the connection strength of division by excluding internal connections
        division_strength = sum_weights - r_weights - l_weights

        return division_strength

    # ----------------------------------------- DEAL WITH EVOLUTION -------------------------------------------------- #
    def evolve_with_update_networkState(self):
        """Perform one evolution step: update weights and then update states."""
        self.update_weights()
        self.update_states()
        self.update_networkState()

    def evolve(self):
        """Perform one evolution step: update weights and then update states."""
        self.update_weights()
        self.update_states()

    # ----------------------------------------- RETURN --------------------------------------------------------------- #
    def return_network(self):
        """Update and return the current network."""
        # Update node states
        for i in range(len(self.s_vec)):
            self.network.nodes[i]['state'] = self.s_vec[i]

        # Update edge weights
        row, col = self.w_mat.nonzero()
        for i, j in zip(row, col):
            self.network[i][j]['weight'] = self.w_mat[i, j]

        return self.network

    def return_state_vector(self):
        """Return the current state vector"""
        return self.s_vec

    def return_network_evolution_state(self):
        """Return dictionary containing the evolution of the states of the nodes."""
        return self.networkState
