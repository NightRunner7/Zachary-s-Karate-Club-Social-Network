import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
# --- IMPORT FROM FILES
# Add the parent directory to the Python module search path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import config as cfg
import auxiliaryFunctions as auxFun
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
    'radical_members': 6,
    'k': 40,
    'p': 0.01,
    # Weights distribution, set zero if they not normal distributed
    'mean': 0.5,
    'std_dev': 0.05
    }

# Manage time settings
time_end = 200
dt = 0.005
number_of_plots_and_checks = 200

# create network and name
init_network = create_graph(set_affiliation_choice=False, sim_config=sim_config)
name = create_name(sim_config=sim_config)

# --- SETTING OF SIMULATION: USER
# basic setting
dt_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
print("dt_list", dt_list)
number_runs = len(dt_list)

# settings of dynamic evolution: CLASS
Deff = 5/10
sim_config = cfg.set_network_evolution_parameters(sim_config, Deff, diffusion=5.0)

# --- SETTINGS OF SIMULATIONS: AUTOMATIC
main_dir = f"{name}-Deff{sim_config['Deff']:.1f}-D{sim_config['D']:.1f}"

# --- SCAN OVER RUNS
for run in range(0, number_runs):
    # --- PREPARE RUN: WHAT PARAMETERS CHANGE IN THESE RUNS
    sim_config = cfg.adjust_time_for_diffusion(sim_config,
                                               sim_config['D'],
                                               base_time_end=time_end,
                                               base_dt=dt,
                                               check_interval=number_of_plots_and_checks,
                                               draw_interval=number_of_plots_and_checks,
                                               update_interval=number_of_plots_and_checks)
    output_main = f"{main_dir}/dt{sim_config['dt']:.3f}"

    # --- PREPARE RUN: BASIC SETTINGS
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

        # Periodically draw histogram of states and weights.
        if makePlot and step % sim_config['timeStepsDraw'] == 0:
            time_val = step * sim_config['dt']
            name = f'histogram_at_time_{time_val:.1f}'
            network = TwitterModel.return_network()

            # --- Draw a histogram weights of Twitter Network
            auxFun.histogram_weights(network,
                                     plot_fit=plot_fit_weight,
                                     mean_val=sim_config['mean'], std_dev_val=sim_config['std_dev'],
                                     output_path=output_evolutionHistoWeights, file_name=name)

            # --- Draw a histogram states of Twitter Network
            auxFun.histogram_states(network,
                                    plot_fit=plot_fit_states,
                                    mean_val=sim_config['mean'], std_dev_val=sim_config['std_dev'],
                                    output_path=output_evolutionHistoState, file_name=name)

        # --- Strength of connection, calculate phase and update data
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

    # --- STORE DATA
    phase_val = TwitterModel.return_phase_value()
    end_time = time.time()  # Capture the end time
    print("val_dt:", dt_list[run], "sim_config['dt']:", sim_config['dt'], "sim_config['D']", sim_config['D'])
    print("sim_config['timeSteps']:", sim_config['timeSteps'],
          "sim_config['timeStepsToCheck']:", sim_config['timeStepsToCheck'])
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
             marker='o', linestyle='-', color='b', label='Connection Strength')
    plt.title('Phase value Over Time')
    plt.xlabel('Time step')
    plt.ylabel('Phase value ')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_main}/phase_value_over_time.png")
    plt.close()
