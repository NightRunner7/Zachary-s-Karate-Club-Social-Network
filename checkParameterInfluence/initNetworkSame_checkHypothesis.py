"""
Here we'll show that we have somehow general behaviour for evolution of our network if the proportion of
two evolution parameters is the same. So my hypothesis is, if `val_D / beta` will be the same it leads
to exactly same network evolution.
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
# --- IMPORT FROM FILES
# Add the parent directory to the Python module search path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import auxiliaryFunctions as auxFun
import config as cfg
from TwitterRadicalizationModel import TwitterRadicalizationModel
from initialGraph.watts_NS_UW import create_graph, create_name


# --- SETTINGS BASE FLAGS: USER
makePlot = True
plot_fit_weight = False
plot_fit_states = True

# --- SETTINGS OF INITIAL GRAPH: USER
sim_config = {
    # base parameters
    'members': 1000,
    'radical_members':  28,
    'k': 12,
    'p': 0.02,
    # Weights distribution, set zero if they not normal distributed
    'mean': 0.5,
    'std_dev': 0.05
    }

# create network and name
init_network = create_graph(set_affiliation_choice=False, sim_config=sim_config)
name = create_name(sim_config=sim_config)

# --- SETTING OF SIMULATION: USER
sim_config['Deff'] = 17/10  # D / beta
val_beta_arr = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
val_D_arr = [beta * sim_config['Deff'] for beta in val_beta_arr]
number_runs = len(val_D_arr)

# --- SETTINGS OF SIMULATIONS: AUTOMATIC
main_dir = f"{name}-Deff{sim_config['Deff']:.1f}"

# --- SCAN OVER RUNS
for run in range(0, number_runs):
    # --- PREPARE RUN: WHAT PARAMETERS CHANGE IN THESE RUNS
    sim_config['D'] = val_D_arr[run]
    sim_config['beta'] = val_beta_arr[run]
    sim_config = cfg.adjust_time_for_diffusion_vol2(sim_config,
                                                    sim_config['D'],
                                                    base_time_end=100)
    output_main = f"{main_dir}/D{sim_config['D']:.1f}"  # specific dir name

    # --- PREPARE RUN: BASIC SETTINGS
    # basic setting
    network_dynamics = dict()
    # for all time steps
    network_dynamics['time_step_arr'] = np.array([])
    network_dynamics['connection_str_arr'] = np.array([])
    network_dynamics['phase_arr'] = np.array([])

    # set time of reaching stable phase, which have been reset in each run
    time_moment = 0.0

    # create necessary directories
    output_evolutionHistoWeights = f"{output_main}/evolutionHistoWeights"
    output_evolutionHistoState = f"{output_main}/evolutionHistoState"

    if makePlot:
        auxFun.make_directory(main_dir)
        auxFun.make_directory(output_main)
        auxFun.make_directory(output_evolutionHistoWeights)
        auxFun.make_directory(output_evolutionHistoState)

    # --- CREATE MODEL
    TwitterModel = TwitterRadicalizationModel(init_network.copy(),
                                              D=sim_config['D'],
                                              beta=sim_config['beta'],
                                              dt=sim_config['dt'])  # COPY NEEDED !!!

    # --- EVOLVE NETWORK
    start_time = time.time()  # Start timing the simulation for performance measurement.
    for step in range(sim_config['timeSteps']):
        # Periodically check the network's phase status: stop simulation if achieve stable phase.
        if step % sim_config['timeStepsToCheck'] == 0:
            time_moment = step * sim_config['dt']
            stop_simulation_flag = TwitterModel.stop_simulation_criteria(time_moment)
            if stop_simulation_flag:
                break

        if makePlot and step % sim_config['timeStepsDraw'] == 0:
            stepStr = f'{step:04d}'
            name = f'histogram_at_step_{stepStr}'
            network = TwitterModel.return_network()

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
        network_dynamics['connection_str_arr'] = np.append(network_dynamics['connection_str_arr'], connection_strength)
        # calculate phase
        phase_val = TwitterModel.find_the_phase()
        network_dynamics['phase_arr'] = np.append(network_dynamics['phase_arr'], phase_val)
        # update time step
        network_dynamics['time_step_arr'] = np.append(network_dynamics['time_step_arr'], step)

        # Perform an evolution step in the model.
        TwitterModel.evolve()

    # --- PRINT LAST PHASE
    phase_val = TwitterModel.return_phase_value()
    end_time = time.time()  # Capture the end time
    print("val_D:", sim_config['D'], "val_beta:", sim_config['beta'])
    print("time ending of simulation:", time_moment, "phase_val in the end of simulation:", phase_val)
    print(f"Execution time: {end_time - start_time} seconds")
    print("-------------------------------------------------------------------------------------------------------")

    # --- STRENGTH CONNECTION PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(network_dynamics['time_step_arr'], network_dynamics['connection_str_arr'],
             marker='o', linestyle='-', color='b', label='Connection Strength')
    plt.title('Connection Strength of Division Over Time')
    plt.xlabel('Time')
    plt.ylabel('Connection Strength')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_main}/connection_strength_over_time.png")
    plt.close()

    # --- STRENGTH CONNECTION PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(network_dynamics['time_step_arr'], network_dynamics['phase_arr'],
             marker='o', linestyle='-', color='b', label='Phase value')
    plt.title('Phase value Over Time')
    plt.xlabel('Time step')
    plt.ylabel('Phase value ')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_main}/phase_value_over_time.png")
    plt.close()
