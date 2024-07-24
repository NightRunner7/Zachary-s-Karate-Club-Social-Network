import os
import sys
import numpy as np
# --- IMPORT FROM FILES
# Add the parent directory to the Python module search path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import config as cfg
import auxiliaryFunctions as auxFun
from TwitterRadicalizationModel import TwitterRadicalizationModel
# --- IMPORT TO MAKE PLOTS
from makePlots.plot_connection_strength import plot_connection_strength
from makePlots.plot_phase_value_over_time import plot_phase_value_over_time

# --- SETS THE INITIAL GRAPH: USER
from initialGraph.watts_NS_UW import create_graph, create_name

# --- FIXED PARAMETERS: USER
sim_config = {
    # Base parameters
    'members': 1000,
    'radical_members': 340,
    'k': 150,
    'p': 0.05,
    # Weights distribution, set zero if they not normal distributed
    'mean': 0.5,
    'std_dev': 0.05,
    # Diffusion and effective diffusion
    'D': 5.0,
    'Deff': 0.5,
    # Other
    'run_number': 1
    }
# Flags
plot_fit_weight = False
plot_fit_states = True
plot_phase_log_scale = True
makePlot = True

# Localization of output
main_dir = "OutputResults"

# Manage time settings
time_end = 200
dt = 0.001
number_of_plots_and_checks = 200

# --- FIXED PARAMETERS: AUTOMATIC
# set beta parameter
sim_config = cfg.set_network_evolution_parameters(sim_config, sim_config['Deff'], diffusion=sim_config['D'])

# set time step and ending time
sim_config = cfg.adjust_time_for_diffusion(sim_config,
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

    if step % sim_config['timeStepsToCheck'] == 0:
        # Periodically check the network's phase status, stop simulation if: achieve stable phase, non-stable network.
        time_moment = step * sim_config['dt']
        stop_simulation_flag = TwitterModel.stop_simulation_criteria(time_moment)
        if stop_simulation_flag:
            break

    # --- strength of connection, calculate phase and update data
    # strength of connection
    connection_strength = TwitterModel.connection_strength_of_division()
    network_dynamics['connection_str'] = np.append(network_dynamics['connection_str'], connection_strength)
    network_dynamics['time_arr'] = np.append(network_dynamics['time_arr'], time)
    # calculate phase
    phase_val = TwitterModel.find_the_phase()
    network_dynamics['phase_arr'] = np.append(network_dynamics['phase_arr'], phase_val)

    # --- save data and do evolution step
    # do evolution step
    TwitterModel.evolve()

# --- STRENGTH CONNECTION PLOT
plot_connection_strength(network_dynamics, output_main, log_scale=plot_phase_log_scale)

# --- PHASE EVOLUTION PLOT
plot_phase_value_over_time(network_dynamics, output_main)
