"""
    Here I provide the information how typically simulation setting look like:

    sim_config = {
        # Base parameters
        'members': 1000,
        'radical_members': radical_members,
        'k': k,
        'p': probability,
        # Weights distribution, set zero if they not normal distributed
        'mean': 0.5,
        'std_dev': 0.05,
        # Effective diffusion
        'Deff': val_Deff,
        # Other
        'run': run
    }
"""

import networkx as nx
import numpy as np
from scipy.stats import entropy

# ---------------------------------- FUNCTIONS: SETS PARAMETERS ------------------------------------------------------ #
def set_network_evolution_parameters(sim_config, effective_diffusion):
    """
    Configures network evolution parameters by calculating the beta value required to achieve a specified effective
    diffusion rate (Deff) given a diffusion rate (D). This function updates the simulation configuration dictionary
    with the effective diffusion.

    This setup is crucial for studies where understanding the interaction between diffusion and edge weight adjustment
    dynamics is essential, such as validating theoretical models or conducting sensitivity analyses.

    Args:
        sim_config (dict): A dictionary where network parameters are stored and updated.
        effective_diffusion (float): The target effective diffusion rate (Deff), which is a key parameter in the model
            influencing the network dynamics. This rate combines the effects of node state diffusion (D) and edge
            weight adjustments.

    Returns:
        dict: The updated simulation configuration dictionary with the newly set parameters.

    Updates:
        - 'Deff': Sets the effective diffusion rate.

    Note:
        - It is critical to ensure that the effective_diffusion rate provided is attainable with the given diffusion
        parameter, as incorrect configurations can lead to unrealistic or unstable simulation behavior.
    """
    sim_config.update({
        "Deff": effective_diffusion,
    })

    return sim_config

def adjust_time_for_diffusion(sim_config,
                              base_dt=0.005,
                              base_time_end=1000,
                              check_interval=200,
                              draw_interval=200,
                              update_interval=200):
    """
    This function will be used to define time steps and duration. We have to keep in mind, we are using the
    dimensionless units of time in this approach [t^hat = t * D], where [D] = [1/t]: diffusion. We believe that
    way helps maintain consistent simulation dynamics.

    It is important to understand the concept of the effective diffusion rate (Deff), which is D (diffusion rate)
    divided by β (beta), the rate of edge weight adjustments. Deff provides a conceptual understanding of the
    interaction between node state diffusion and edge dynamics.

    Args:
        sim_config (dict): Configuration dictionary for simulation parameters.
        base_dt (float): Initial time step before scaling.
        base_time_end (int): Initial total simulation time before scaling.
        check_interval (int): How many times you want to stability phase checks.
        draw_interval (int):  How many times you want to redraw of the network graph.
        update_interval (int): How many times you want to update data during simulation.

    Returns:
        dict: Updated simulation configuration with adjusted time parameters.

    Notes:
        - Deff (D/β) is not directly used in this function but is crucial for understanding how diffusion and
          edge dynamics interact in the broader model. β (beta) is typically defined elsewhere in the model where
          edge weights are dynamically adjusted based on node interactions.
        - The file `initNetworkSame_checkHypothesis.py` shows teh universal graph evolution with same values of
          effective diffusion (Deff).
        - The file `initNetwork_dt.py` shows the maximal value of `dt`, which gives us stable evolution of the
          network. It seems a good choice is: `0.001 >= dt`.
    """
    adjusted_dt = base_dt
    adjusted_time_end = base_time_end
    total_steps = int(adjusted_time_end / adjusted_dt)

    sim_config.update({
        "dt": adjusted_dt,
        "time_end": adjusted_time_end,
        "timeSteps": total_steps,
        "timeStepsToCheck": int(total_steps / check_interval),
        "timeStepsDraw": int(total_steps / draw_interval),
        "timeStepsUpdateData": int(total_steps / update_interval)
    })

    return sim_config

# ---------------------------------- DEAL WITH NETWORK EVOLUTION ----------------------------------------------------- #
class NetworkEvolutionDeff:
    def __init__(self, network, Deff=0.5, dt=0.001):
        """
        Initialize the model with given parameters and set up the initial network.

        :param network: A NetworkX graph with nodes that may have 'affirmation' attributes.
        :param Deff: Effective diffusion coefficient.
        :param dt: Time step.
        """
        # --- Evolution graph coefficients
        self.Deff = Deff
        self.dt = dt

        # --- Load the initial graph
        self.network = network
        self.N = len(self.network.nodes)  # number of nodes (members in the graph)
        self.N_left_init, self.N_right_init = self.count_affirmations()  # number of left and right radical members
        self.N_rad = self.N_left_init + self.N_right_init  # number of radical members

        # --- Initialize necessary matrices to describe evolution of states and weight
        # Initialize node states
        self.s_vec = np.array([data['state'] for _, data in self.network.nodes(data=True)])

        # Initialize weight matrices using sparse matrix
        self.w_mat = nx.to_scipy_sparse_array(self.network, format='csr')

        # Save the initial state
        self.networkState = {node: np.array([state]) for node, state in enumerate(self.s_vec)}

        # --- Defaults settings of calculate phases, which distinguishable behaviour of whole network
        # set constants
        self.epsilon = 0.05
        self.neutral_width = 0.4
        self.division_threshold = 0.2
        self.wall_threshold = 0.2
        # set counters, which stop simulations (we achieve stabilize phase)
        self.counter_of_domination_phase = 0  # we reach `dominant` phase
        self.counter_of_fullDivision_phase = 0  # we reach `full division` phase
        self.counter_of_same_phase = 0  # how many times we have reached exactly same phase
        self.upperLimit_of_same_phase = 10  # how many times we can reach exactly same phase before stop simulation
        self.lowerTime_of_same_phase = 80  # the first steps can lead to same phase, we have take that into account
        # set as: 'nonrecognition' phase
        self.previous_phase = np.nan
        self.present_phase = np.nan
        # set flag for stable graf evolution
        self.stable_evolution = True

        # --- Defaults settings of calculate phases, which distinguishable behaviour of whole network: second approach
        self.neutral_nodes_indices = self.get_neutral_nodes()

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

    def change_setting_of_finding_phases(self, dictionary):
        """
        Updates the settings used for determining the phases of a system within a class instance.
        This method accepts a dictionary with keys corresponding to different configurable properties
        of the phase detection process. Depending on the key provided, it updates the class attributes
        associated with that key.

        The function supports the modification of various thresholds and counters related to the phase
        detection mechanism of a system, allowing dynamic adjustment based on runtime analysis or external
        inputs.

        Args:
            dictionary (Dict[str, int or float]): A dictionary containing key-value pairs where the keys
            are the names of the attributes to be updated, and the values are the new settings for these attributes.

        Possible keys include:
            - "epsilon": Sets the tolerance level used to define boundaries for phases.
            - "neutral_width": Sets the width of the neutral zone centered around a baseline value.
            - "division_threshold": Sets the threshold for determining a division phase.
            - "wall_threshold": Sets the threshold for determining a wall phase.
            - "counter_of_domination_phase": Sets the counter for domination phase occurrences.
            - "counter_of_fullDivision_phase": Sets the counter for full division phase occurrences.
            - "counter_of_same_phase": Sets the counter for detecting repeated occurrences of the same phase.
            - "upperLimit_of_same_phase": Sets the upper limit for the counter of the same phase to stop simulation.
            - "lowerTime_of_same_phase": Sets the lower time limit for considering the duration of the same phase occurrence.

        Returns:
            None: This method does not return a value but updates the instance attributes directly.
        """
        if "epsilon" in dictionary:
            self.epsilon = dictionary["epsilon"]
        elif "neutral_width" in dictionary:
            self.neutral_width = dictionary["neutral_width"]
        elif "division_threshold" in dictionary:
            self.division_threshold = dictionary["division_threshold"]
        elif "wall_threshold" in dictionary:
            self.wall_threshold = dictionary["wall_threshold"]
        elif "counter_of_domination_phase" in dictionary:
            self.counter_of_domination_phase = dictionary["counter_of_domination_phase"]
        elif "counter_of_fullDivision_phase" in dictionary:
            self.counter_of_fullDivision_phase = dictionary["counter_of_fullDivision_phase"]
        elif "counter_of_same_phase" in dictionary:
            self.counter_of_same_phase = dictionary["counter_of_same_phase"]
        elif "upperLimit_of_same_phase" in dictionary:
            self.upperLimit_of_same_phase = dictionary["upperLimit_of_same_phase"]
        elif "lowerTime_of_same_phase" in dictionary:
            self.lowerTime_of_same_phase = dictionary["lowerTime_of_same_phase"]

    def get_neutral_nodes(self):
        """
        Retrieve the states of neutral nodes in a network.

        :return: A list of states corresponding to neutral nodes.
        """
        neutral_nodes = []
        for node in self.network.nodes():
            if self.network.nodes[node].get('affirmation') is None:
                neutral_nodes.append(node)

        return neutral_nodes

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
        dwij_dt = -1/self.Deff * self.w_mat.data * (1 - self.w_mat.data) * interaction

        # Update weights
        self.w_mat.data += self.dt * dwij_dt

    def update_states(self):
        """Update the states of the nodes based on the weights of the connecting edges."""
        # Compute the state changes using sparse matrix operations
        delta_s_vec = (self.w_mat @ self.s_vec - self.s_vec * self.w_mat.sum(axis=1))

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

    # ----------------------------------------- SHANON ENTROPY ------------------------------------------------------- #
    def find_the_entropy(self):
        """
        Determine the entropy of the system based on the distribution of state values and using the definition
        of shannon entropy. These entropy can be also useful to determine the evolution of the network
        (probably can replace the phases' definition).

        Returns:
            entropy (float): the shanon entropy, which represents the stage of evolution of the system.

        """
        neutral_states = [self.network.nodes[node]['state'] for node in self.neutral_nodes_indices]

        # Define bin edges from 0 to 1 with specified bin_width
        bin_edges = np.arange(0, 1 + self.epsilon, self.epsilon)

        # Histogram the state values into these bins
        state_counts, _ = np.histogram(neutral_states, bins=bin_edges)

        # Compute probabilities by normalizing the counts
        probabilities = state_counts / np.sum(state_counts)

        # Filter out zero probabilities for valid entropy calculation
        probabilities = probabilities[probabilities > 0]

        # Calculate Shannon entropy
        return entropy(probabilities, base=2)

    # ----------------------------------------- DEAL WITH PHASES AND STOP EVOLUTION ---------------------------------- #
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
                np.nan - 'nonrecognition'
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
            phase = np.nan  # Nonrecognition phase indicates no significant change or pattern

        return phase

    def stop_simulation_criteria(self, time):
        """
        Evaluates if the simulation should be terminated based on the stability of observed phases.
        The function checks if specific phases indicating stability or stabilization have been met repeatedly,
        suggesting no further significant evolution is expected.

        Args:
            time (float): The current simulation time, used to ensure certain criteria are only evaluated
                          after a minimum time threshold to prevent premature termination.

        Returns:
            bool: True if the simulation meets the criteria for termination, False otherwise.
        """
        # ### FIRST CRITERIA: even one state is bigger than `1.0` or less than `0.0`
        # Sometimes, if `dt` is not small enough then some states become out of the range: [0, 1]
        if np.any(self.s_vec > 1.0):
            print("At least one state is bigger than `1.0`. Evolution is not stable anymore.")
            print("Take smaller `dt` to obtain correct graf evolution.")
            self.stable_evolution = False
            self.present_phase = np.nan  # set as: 'nonrecognition' phase
            return True  # Terminate if repeated stable phases are observed.
        elif np.any(self.s_vec < 0.0):
            print("At least one state is smaller than `0.0`. Evolution is not stable anymore.")
            print("Take smaller `dt` to obtain correct graf evolution.")
            self.stable_evolution = False
            self.present_phase = np.nan  # set as: 'nonrecognition' phase
            return True  # Terminate if repeated stable phases are observed.

        # ### SECOND CRITERIA: identify stable or quasi stable phase
        # --- Find previous and present phases
        self.previous_phase = self.present_phase
        self.present_phase = self.find_the_phase(
            epsilon=0.05,
            neutral_width=0.4,
            division_threshold=0.2,
            wall_threshold=0.2
        )

        # --- STABLE PHASE
        # Check for repeated occurrences of fully stable phases (division or domination)
        if self.present_phase in (2.0, 3.0):
            if self.present_phase == 2.0:
                self.counter_of_fullDivision_phase += 1
            elif self.present_phase == 3.0:
                self.counter_of_domination_phase += 1

            if self.counter_of_fullDivision_phase > 2 or self.counter_of_domination_phase > 2:
                return True  # Terminate if repeated stable phases are observed.

        # --- QUASI-STABLE PHASE
        # Reset the counter for quasi-stable phases if phase changes
        if self.previous_phase != self.present_phase:
            self.counter_of_same_phase = 0
        else:
            # Increment counter if the phase remains the same and is within a specific range
            if 1.0 < self.present_phase < 2.0:
                self.counter_of_same_phase += 1

        # Terminate if a quasi-stable phase persists beyond the set threshold and time
        if self.counter_of_same_phase >= self.upperLimit_of_same_phase and time > self.lowerTime_of_same_phase:
            return True

        return False

    def stop_simulation_basic_criteria(self):
        """
        Evaluates if the simulation should be terminated based on the stability of tne network evolution.
        The function checks if any node (member) `state` get out from the range [0,1]. If it does, it means
        our network evolution is unstable. This also indicates we have to set lower value of time step (`dt`)
        next time, if we're looking for network evolution for specific values of parameters.

        Returns:
            bool: True if the simulation meets the criteria for termination, False otherwise.
        """
        # ### FIRST CRITERIA: even one state is bigger than `1.0` or less than `0.0`
        # Sometimes, if `dt` is not small enough then some states become out of the range: [0, 1]
        if np.any(self.s_vec > 1.0):
            print("At least one state is bigger than `1.0`. Evolution is not stable anymore.")
            print("Take smaller `dt` to obtain correct graf evolution.")
            self.stable_evolution = False
            self.present_phase = np.nan  # set as: 'nonrecognition' phase
            return True  # Terminate if repeated stable phases are observed.
        elif np.any(self.s_vec < 0.0):
            print("At least one state is smaller than `0.0`. Evolution is not stable anymore.")
            print("Take smaller `dt` to obtain correct graf evolution.")
            self.stable_evolution = False
            self.present_phase = np.nan  # set as: 'nonrecognition' phase
            return True  # Terminate if repeated stable phases are observed.

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
        # clip states: try to avoid unstable network evolution
        self.s_vec = np.clip(self.s_vec, 0, 1)

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

    def return_phase_value(self):
        """Return present value of the phase"""
        return self.present_phase

    def return_stable_evolution(self):
        """Return flags, which differentiate stable and non-stable evolution of the network"""
        return self.stable_evolution

    def return_entropy(self):
        """Return the present value of the shanon entropy"""
        val_entropy = self.find_the_entropy()
        return val_entropy
