import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, entropy

# --- IMPORT FROM FILES
# Add the parent directory to the Python module search path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import auxiliaryFunctions as auxFun
from TwitterRadicalizationModel import TwitterRadicalizationModel,\
    set_network_evolution_parameters, adjust_time_for_diffusion
# --- IMPORT TO MAKE PLOTS
from makePlots.plot_connection_strength import plot_connection_strength
from makePlots.plot_phase_value_over_time import plot_phase_value_over_time

# --- SETS THE INITIAL GRAPH: USER
from initialGraph.watts_NS_UW import create_graph, create_name

def calculate_entropy_from_continuous_states(states, bin_width=0.05, print_flag=False):
    # Define bin edges from 0 to 1 with specified bin_width
    bin_edges = np.arange(0, 1 + bin_width, bin_width)

    # Histogram the state values into these bins
    state_counts, _ = np.histogram(states, bins=bin_edges)

    # Compute probabilities by normalizing the counts
    probabilities = state_counts / np.sum(state_counts)

    # Filter out zero probabilities for valid entropy calculation
    probabilities = probabilities[probabilities > 0]

    if print_flag:
        print("probabilities:", probabilities)

    # Calculate Shannon entropy
    return entropy(probabilities, base=2)

# --- FIXED PARAMETERS: USER
sim_config = {
    # Base parameters
    'members': 1000,
    'radical_members': 120,
    'k': 80,
    'p': 0.02,
    # Weights distribution, set zero if they not normal distributed
    'mean': 0.5,
    'std_dev': 0.05,
    # Diffusion and effective diffusion
    'D': 5.0,
    'Deff': 0.5,
    # Other
    'run_number': 2
    }
# Flags
plot_fit_weight = False
plot_fit_states = True
makePlot = True

# Localization of output
main_dir = "OutputResults"

# Manage time settings
time_end = 200
dt = 0.001
number_of_plots_and_checks = 200

# --- FIXED PARAMETERS: AUTOMATIC
# set beta parameter
sim_config = set_network_evolution_parameters(sim_config, sim_config['Deff'], diffusion=sim_config['D'])

# set time step and ending time
sim_config = adjust_time_for_diffusion(sim_config,
                                       sim_config['D'],
                                       base_time_end=time_end,
                                       base_dt=dt,
                                       check_interval=number_of_plots_and_checks,
                                       draw_interval=number_of_plots_and_checks,
                                       update_interval=number_of_plots_and_checks)
# create network and name
init_network = create_graph(set_affiliation_choice=False, sim_config=sim_config)
directory_name = create_name(sim_config=sim_config)

# --- PREPARE EVOLUTION: AUTOMATIC
# basic setting
output_main = f"{main_dir}/{directory_name}"
network_dynamics = dict()
# for all time steps
network_dynamics['time_arr'] = np.array([])
network_dynamics['connection_str'] = np.array([])
network_dynamics['phase_arr'] = np.array([])
network_dynamics['skewness'] = np.array([])
network_dynamics['kurtosis'] = np.array([])
network_dynamics['entropy'] = np.array([])

# Make directory
output_evolutionGraph = f"{output_main}/evolutionGraph"
output_evolutionHistoWeights = f"{output_main}/evolutionHistoWeights"
output_evolutionHistoState = f"{output_main}/evolutionHistoState"

if makePlot:
    auxFun.make_directory(output_main)
    auxFun.make_directory(output_evolutionHistoWeights)
    auxFun.make_directory(output_evolutionHistoState)
    # auxFun.make_directory(output_evolutionGraph)

# --- CREATE MODEL
TwitterModel = TwitterRadicalizationModel(init_network,
                                          D=sim_config['D'],
                                          beta=sim_config['beta'],
                                          dt=sim_config['dt'])

# ---------------------------------- EVOLVE NETWORK ------------------------------------------------------------------ #
for step in range(sim_config['timeSteps']):
    time = step * sim_config['dt']

    if makePlot and step % sim_config['timeStepsDraw'] == 0:
        print("I m doing step:", step, "time:", time)
        timeStr = f'{time:.1f}'
        name = f'histogram_at_{timeStr}'
        network = TwitterModel.return_network()

        # --- draw a network of Twitter Network
        # auxFun.draw_graph_spring(network, initial_positions,
        #                          output_path=output_evolutionGraph,
        #                          step=timeStr, file_name="network_at")

        # --- draw a histogram weights of Twitter Network
        auxFun.histogram_weights(network,
                                 plot_fit=plot_fit_weight,
                                 mean_val=sim_config['mean'], std_dev_val=sim_config['std_dev'],
                                 output_path=output_evolutionHistoWeights, file_name=name)

        # --- draw a histogram states of Twitter Network
        auxFun.histogram_states(network,
                                plot_fit=plot_fit_states,
                                mean_val=sim_config['mean'], std_dev_val=sim_config['std_dev'],
                                output_path=output_evolutionHistoState, file_name=name)

    # --- strength of connection, calculate phase and update data
    # strength of connection
    connection_strength = TwitterModel.connection_strength_of_division()
    network_dynamics['connection_str'] = np.append(network_dynamics['connection_str'], connection_strength)
    network_dynamics['time_arr'] = np.append(network_dynamics['time_arr'], time)
    # calculate phase
    phase_val = TwitterModel.find_the_phase()
    network_dynamics['phase_arr'] = np.append(network_dynamics['phase_arr'], phase_val)
    # find skewness and kurtosis
    s_neutral_vec = TwitterModel.return_neutral_state_vector()
    kur = kurtosis(s_neutral_vec)
    network_dynamics['kurtosis'] = np.append(network_dynamics['kurtosis'], kur)
    skw = skew(s_neutral_vec)
    network_dynamics['skewness'] = np.append(network_dynamics['skewness'], skw)

    entropy_val = calculate_entropy_from_continuous_states(s_neutral_vec, bin_width=0.05)
    network_dynamics['entropy'] = np.append(network_dynamics['entropy'], entropy_val)

    if step % sim_config['timeStepsToCheck'] == 0:
        # Periodically check the network's phase status, stop simulation if: achieve stable phase, non-stable network.
        # phase_val = TwitterModel.return_phase_value()
        # print("Phase val:", phase_val)

        time_moment = step * sim_config['dt']
        stop_simulation_flag = TwitterModel.stop_simulation_criteria(time_moment)
        if stop_simulation_flag:
            break

    # --- save data and do evolution step
    # do evolution step
    TwitterModel.evolve()

# --- STRENGTH CONNECTION PLOT: LOG
plot_connection_strength(network_dynamics, output_main, log_scale=True)

# --- STRENGTH CONNECTION PLOT: BASE
plot_connection_strength(network_dynamics, output_main, log_scale=False)

# --- PHASE EVOLUTION PLOT
plot_phase_value_over_time(network_dynamics, output_main)

# Plotting
plt.figure(figsize=(10, 5))

# Plot for skewness
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(network_dynamics['time_arr'], network_dynamics['skewness'], marker='o', linestyle='-')
plt.title('Skewness Over Time')
plt.xlabel('Time')
plt.ylabel('Skewness')

# Plot for kurtosis
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(network_dynamics['time_arr'], network_dynamics['kurtosis'], marker='o', linestyle='-')
plt.title('Kurtosis Over Time')
plt.xlabel('Time')
plt.ylabel('Kurtosis')

plt.savefig(f"{output_main}/skewness_and_kurtosis_over_time.png")
plt.close()

# print last values of two parameters
print("last skewness:", network_dynamics['skewness'][-1])
print("last kurtosis:", network_dynamics['kurtosis'][-1])
print("last entropy:", network_dynamics['entropy'][-1])

# s_neutral_vec = TwitterModel.return_neutral_state_vector()
# entropy_val = calculate_entropy_from_continuous_states(s_neutral_vec, bin_width=0.05, print_flag=True)
# print("last entropy:", entropy_val)

plt.subplot(1, 1, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(network_dynamics['time_arr'], network_dynamics['entropy'], marker='o', linestyle='-')
plt.title('Entropy Over Time')
plt.xlabel('Time')
plt.ylabel('Entropy')

plt.savefig(f"{output_main}/entropy_over_time.png")
plt.close()
