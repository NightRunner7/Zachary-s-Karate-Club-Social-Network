import networkx as nx
import numpy as np


class TwitterRadicalizationModel:
    def __init__(self, network, D=5., beta=10., dt=0.01):
        """
        Initialize the model with given parameters and set up the initial network.

        :param network: A NetworkX graph with nodes that may have 'affirmation' attributes.
        :param D: Diffusion coefficient.
        :param beta: Coupling parameter.
        :param dt: Time step.
        """
        self.D = D
        self.beta = beta
        self.dt = dt

        # Load the graph
        self.network = network
        self.N = len(self.network.nodes)  # number of nodes (members in the graph)
        self.N_left_init, self.N_right_init = self.count_affirmations()  # number of left and right radical members
        self.N_rad = self.N_left_init + self.N_right_init  # number of radical members

        # Initialize node states
        self.s_vec = np.array([data['state'] for _, data in self.network.nodes(data=True)])

        # Initialize weight matrices using sparse matrix
        self.w_mat = nx.to_scipy_sparse_array(self.network, format='csr')

        # Save the initial state
        self.networkState = {node: np.array([state]) for node, state in enumerate(self.s_vec)}

    # ----------------------------------------- BASE METHODS --------------------------------------------------------- #
    def count_affirmations(self, out_of_class=False, network=None):
        """
        Count nodes with 'far-left' and 'far-right' affirmations in a network.

        This method supports both an internally stored network or an externally provided one.
        Use the `out_of_class` parameter to specify which network to use: set it to True to use
        `network` parameter or False to use the network stored within the class.

        :param out_of_class: A boolean flag to specify whether to use the network passed to the method.
                             If False, uses the class's internal network.
        :param network: Optional; a NetworkX graph with nodes that may have 'affirmation' attributes.
                        Required if `out_of_class` is True.
        :return: A tuple containing counts (count_far_left, count_far_right)
        """
        count_far_left = 0
        count_far_right = 0

        # Determine which network to use based on the out_of_class flag
        target_network = network if out_of_class else self.network

        # Check if the network is properly provided
        if target_network is None:
            raise ValueError("No network provided for the operation.")

        # Iterate over all nodes and their data in the network
        for node, data in target_network.nodes(data=True):
            affirmation = data.get('affirmation')  # Safely get the 'affirmation' attribute

            # Check and update counts based on the affirmation
            if affirmation == 'far-left':
                count_far_left += 1
            elif affirmation == 'far-right':
                count_far_right += 1

        return count_far_left, count_far_right

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

    # ----------------------------------------- DEAL WITH PHASES ----------------------------------------------------- #
    def find_the_phase(self, epsilon=0.05, neutral_width=0.4, division_threshold=0.2, wall_threshold=0.2,
                       out_of_class=False, network=None):
        """
        Determines the phase of the system based on the distribution of state values at the end of an evolutionary
        process.

        Args:
            epsilon (float): The tolerance used to classify left and right members as radicals.
            neutral_width (float): The width of the neutral zone centered at 0.5, within which members are considered
                neutral.
            division_threshold (float): The threshold ratio of radical change needed to declare a division phase.
            wall_threshold (float): The threshold ratio of neutral members needed to declare a wall phase.
            out_of_class (bool): Flag to determine whether to use an external network passed as an argument.
            network (NetworkX graph): The external network to analyze if out_of_class is True.

        Returns:
            float: A numeric code representing different phases of the system:
                4.0 - 'nonrecognition'
                3.0 - 'domination'
                1.0 to 2.0 - 'division' (variable based on the degree of radical change)
                0.0 to 1.0 - 'wall' (variable, inversely related to the degree of neutrality)
        """
        # --- SETTING PARAMETERS OF NETWORK
        # Choose between internal state vector or an external one based on `out_of_class`
        s_vec = np.array([state for _, state in network.nodes(data='state', default='Not Available')]) if out_of_class \
            else self.s_vec
        # Determine initial counts of radical members, considering external or internal data source
        N_left_init, N_right_init = self.count_affirmations(out_of_class=True, network=network) if out_of_class \
            else (self.N_left_init, self.N_right_init)

        # Total number of members in the network
        N = len(network.nodes) if out_of_class else self.N
        # Total initial radical members
        N_rad = N_left_init + N_right_init

        # --- COUNTING NECESSARY MEMBERS IN DIFFERENT GROUPS
        # Count members within specific state ranges at the end of the evolution
        N_left_end = np.sum(s_vec <= epsilon)
        N_right_end = np.sum(s_vec >= 1.0 - epsilon)
        N_neutral_end = np.sum((s_vec > 0.5 - neutral_width / 2) & (s_vec < 0.5 + neutral_width / 2))

        # Calculate the change in the number of radical members from initial to final
        delta_N_rad = (N_left_end + N_right_end) - (N_left_init + N_right_init)

        # Population not initially identified as radical
        remaining_non_radical = N - N_rad

        # --- TAKE CARE ABOUT PHASES
        # Determine the phase based on conditions involving changes in radical and neutral members
        if N_left_end == N - N_right_init or N_right_end == N - N_left_init:
            phase = 3.0  # Domination phase indicates complete shift to one radical side
        elif (delta_N_rad / remaining_non_radical) >= division_threshold:
            phase = 1.0 + (delta_N_rad / remaining_non_radical)  # Division phase indicates significant radicalization
        elif (N_neutral_end / remaining_non_radical) >= wall_threshold:
            phase = 1.0 - (N_neutral_end / remaining_non_radical)  # Wall phase indicates a significant neutral buffer
        else:
            phase = 4.0  # Nonrecognition phase indicates no significant change or pattern

        return phase

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
