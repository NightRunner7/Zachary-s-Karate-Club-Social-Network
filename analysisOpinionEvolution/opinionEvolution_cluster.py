import os
import sys
import argparse
import numpy as np
import networkx as nx
# --- IMPORT FROM FILES
# Add the parent directory to the Python module search path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import config as cfg
import auxiliaryFunctions as auxFun
from TwitterRadicalizationModel import TwitterRadicalizationModel
# --- IMPORT TO MAKE PLOTS
from makePlots.plot_heatmaps import plot_closeness_centrality_heatmap, plot_eigenvector_centrality_heatmap
from makePlots.plot_network_dynamics import plot_network_dynamics
from makePlots.plot_connection_strength import plot_connection_strength
from makePlots.plot_phase_value_over_time import plot_phase_value_over_time

# --- SETS THE INITIAL GRAPH: USER
from initialGraph.watts_NS_UW import create_graph, create_name

# ######################################### MAIN FUNCTION ############################################################ #
def main(radical_members, k, probability, val_D, val_Deff, run, time):
    # --- FIXED PARAMETERS: USER
    sim_config = {
        # Base parameters
        'members': 1000,
        'radical_members': radical_members,
        'k': k,
        'p': probability,
        # Weights distribution, set zero if they not normal distributed
        'mean': 0.5,
        'std_dev': 0.05,
        # Diffusion and effective diffusion
        'D': val_D,
        'Deff': val_Deff,
        # Other
        'run_number': run
    }
    # Flags
    plot_fit_weight = False
    plot_fit_states = True
    plot_phase_log_scale = True
    makePlot = True
    makeUpdateData = True

    # Localization of output
    main_dir = "OutputResults"

    # Manage time settings
    time_end = time
    dt = 0.001
    number_of_plots_and_checks = time

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
    network_dynamics['eigVec_central_mat'] = np.zeros((len(init_network.nodes()), int(sim_config['time_end'])))
    network_dynamics['closeness_central_mat'] = np.zeros((len(init_network.nodes()), int(sim_config['time_end'])))
    # for select time steps
    network_dynamics['time_dyn'] = np.array([])
    network_dynamics['diameter'] = np.array([])
    network_dynamics['average_path_length'] = np.array([])
    network_dynamics['average_clustering'] = np.array([])

    # Make directory
    output_saveGraph = f"{output_main}/saveGraph"
    output_evolutionGraph = f"{output_main}/evolutionGraph"
    output_evolutionHistoWeights = f"{output_main}/evolutionHistoWeights"
    output_evolutionHistoState = f"{output_main}/evolutionHistoState"

    if makePlot:
        auxFun.make_directory(output_main)
        auxFun.make_directory(output_saveGraph)
        auxFun.make_directory(output_evolutionGraph)
        auxFun.make_directory(output_evolutionHistoWeights)
        auxFun.make_directory(output_evolutionHistoState)

    # --- CREATE MODEL
    TwitterModel = TwitterRadicalizationModel(init_network,
                                              D=sim_config['D'],
                                              beta=sim_config['beta'],
                                              dt=sim_config['dt'])

    # init positions
    network = TwitterModel.return_network()
    initial_positions = nx.spring_layout(network)  # You can adjust this with different parameters

    # ---------------------------------- EVOLVE NETWORK -------------------------------------------------------------- #
    for step in range(sim_config['timeSteps']):
        time = step * sim_config['dt']

        if (makePlot and step % sim_config['timeStepsDraw'] == 0) or \
                (makeUpdateData and step % sim_config['timeStepsUpdateData'] == 0):
            network = TwitterModel.return_network()

        if makePlot and step % sim_config['timeStepsDraw'] == 0:
            print("I m doing step:", step, "time:", time)
            timeStr = f'{time:.1f}'
            name = f'histogram_at_{timeStr}'

            # --- save network in file
            auxFun.save_network(network,
                                output_path=output_saveGraph,
                                step=timeStr, file_name="network_at")

            # --- draw a network of Twitter Network
            auxFun.draw_graph_spring(network, initial_positions,
                                     output_path=output_evolutionGraph,
                                     step=timeStr, file_name="network_at")

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

        if makeUpdateData and step % sim_config['timeStepsUpdateData'] == 0:
            # --- time list
            network_dynamics['time_dyn'] = np.append(network_dynamics['time_dyn'], time)

            # --- average path:
            average_l = nx.average_shortest_path_length(network, weight='weight')
            network_dynamics['average_path_length'] = np.append(network_dynamics['average_path_length'], average_l)

            # --- diameter
            diameter = nx.diameter(network, weight='weight')
            network_dynamics['diameter'] = np.append(network_dynamics['diameter'], diameter)

            # --- Clustering Coefficient: averaged
            average_clustering = nx.average_clustering(network, weight='weight')
            network_dynamics['average_clustering'] = np.append(network_dynamics['average_clustering'],
                                                               average_clustering)

            # --- eigenvector centrality matrix update data
            centrality = nx.eigenvector_centrality_numpy(network, weight='weight')
            # Store centrality values in the matrix
            network_dynamics['eigVec_central_mat'][:, int(time)] = [centrality[node] for node in network.nodes()]

            # --- Calculate closeness centrality update data
            centrality = nx.closeness_centrality(network, distance='weight')
            # Store centrality values in the matrix
            network_dynamics['closeness_central_mat'][:, int(time)] = [centrality[node] for node in network.nodes()]

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

    if makeUpdateData:
        # --- PLOTTING THE HEATMAP: EIGENVECTOR CENTRALITY
        plot_eigenvector_centrality_heatmap(network_dynamics['eigVec_central_mat'], output_main)

        # --- PLOTTING THE HEATMAP: CLOSENESS CENTRALITY
        plot_closeness_centrality_heatmap(network_dynamics['closeness_central_mat'], output_main)

        # --- PLOTTING: NETWORK DYNAMICS
        plot_network_dynamics(network_dynamics, output_main)

# ######################################### RUN VIA TERMINAL ######################################################### #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run network simulation.")
    parser.add_argument("--radical_members", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--probability", type=float, required=True)
    parser.add_argument("--val_D", type=float, required=True)
    parser.add_argument("--val_Deff", type=float, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--time", type=int, required=True)

    args = parser.parse_args()
    main(args.radical_members, args.k, args.probability, args.val_D, args.val_Deff, args.run, args.time)
