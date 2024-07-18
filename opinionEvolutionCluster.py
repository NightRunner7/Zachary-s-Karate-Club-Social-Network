import argparse
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

# Import your custom modules
import auxiliaryFunctions as auxFun
from TwitterRadicalizationModel import TwitterRadicalizationModel
from initialGraph.watts_NS_UW import create_graph, create_name


def main(radical_members, k, probability, val_D, run, time):
    # Sets the initial graph: USER
    members = 1000  # Assuming members count is constant
    mean = 0.5
    std_dev = 0.05
    plot_fit_weight = False
    plot_fit_states = True

    # Create network and name
    init_network = create_graph(members, radical_members, k, probability,
                                set_affiliation_choice=False, mean=mean, std_dev=std_dev)
    name = create_name(members, radical_members, k, probability, mean, std_dev)

    # --- SETTING OF SIMULATION: USER
    # basic setting
    makePlot = True
    makeUpdateData = True
    main_dir = "./OutputResults"

    # settings of dynamic evolution: CLASS
    val_beta = 10
    val_dt = 0.001

    # settings of dynamic evolution: EVOLUTION
    timeSteps = int((time / val_dt) * (5 / val_D) * (10 / val_beta))
    timeStepsDraw = int((1 / val_dt) * (5 / val_D) * (10 / val_beta))
    timeStepsUpdateData = int((1 / val_dt) * (5 / val_D) * (10 / val_beta))

    # --- SETTING OF SIMULATION: AUTOMATIC
    # basic setting
    output_main = f"{main_dir}/{name}-D{val_D:.1f}-beta{val_beta:.1f}-dt{val_dt}-Run{run}"
    network_dynamics = dict()
    # for all time steps
    network_dynamics['time_str'] = np.array([])
    network_dynamics['connection_str'] = np.array([])
    network_dynamics['eigVec_central_mat'] = np.zeros((len(init_network.nodes()), int(time)))
    network_dynamics['closeness_central_mat'] = np.zeros((len(init_network.nodes()), int(time)))
    # for select time steps
    network_dynamics['time_dyn'] = np.array([])
    network_dynamics['diameter'] = np.array([])
    network_dynamics['average_path_length'] = np.array([])
    network_dynamics['average_clustering'] = np.array([])

    # Make directory
    # output_saveGraph = f"{output_main}/saveGraph"
    # output_evolutionGraph = f"{output_main}/evolutionGraph"
    output_evolutionHistoWeights = f"{output_main}/evolutionHistoWeights"
    output_evolutionHistoState = f"{output_main}/evolutionHistoState"

    if makePlot:
        auxFun.make_directory(output_main)
        # auxFun.make_directory(output_saveGraph)
        # auxFun.make_directory(output_evolutionGraph)
        auxFun.make_directory(output_evolutionHistoWeights)
        auxFun.make_directory(output_evolutionHistoState)

    # --- CREATE MODEL
    TwitterModel = TwitterRadicalizationModel(init_network, D=val_D, beta=val_beta, dt=val_dt)

    # init positions
    # network = TwitterModel.return_network()
    # initial_positions = nx.spring_layout(network)  # You can adjust this with different parameters

    # ---------------------------------- EVOLVE NETWORK -------------------------------------------------------------- #
    for step in range(0, timeSteps):
        time = step * val_dt
        timeStr = f'{time:.1f}'

        if makePlot and step % timeStepsDraw == 0:
            print("I m doing step:", step)
            network = TwitterModel.return_network()

            # --- save network in file
            # auxFun.save_network(network,
            #                     output_path=output_saveGraph,
            #                     step=timeStr, file_name="network_at")

            # --- draw a network of Twitter Network
            # auxFun.draw_graph_spring(network, initial_positions,
            #                          output_path=output_evolutionGraph,
            #                          step=timeStr, file_name="network_at")

            # --- draw a histogram weights of Twitter Network
            name = f'histogram_at_{timeStr}'
            auxFun.histogram_weights(network,
                                     plot_fit=plot_fit_weight, mean_val=mean, std_dev_val=std_dev,
                                     output_path=output_evolutionHistoWeights, file_name=name)

            # --- draw a histogram states of Twitter Network
            auxFun.histogram_states(network,
                                    plot_fit=plot_fit_states, mean_val=mean, std_dev_val=std_dev,
                                    output_path=output_evolutionHistoState, file_name=name)

        if makeUpdateData and step % timeStepsUpdateData == 0:
            network = TwitterModel.return_network()

            # --- time list
            time = step * val_dt
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
            network = TwitterModel.return_network()
            centrality = nx.eigenvector_centrality_numpy(network, weight='weight')
            # Store centrality values in the matrix
            network_dynamics['eigVec_central_mat'][:, int(time)] = [centrality[node] for node in network.nodes()]

            # --- Calculate closeness centrality update data
            centrality = nx.closeness_centrality(network, distance='weight')
            # Store centrality values in the matrix
            network_dynamics['closeness_central_mat'][:, int(time)] = [centrality[node] for node in network.nodes()]

        # --- strength of connection update data
        connection_strength = TwitterModel.connection_strength_of_division()
        network_dynamics['connection_str'] = np.append(network_dynamics['connection_str'], connection_strength)
        time = step * val_dt
        network_dynamics['time_str'] = np.append(network_dynamics['time_str'], time)

        # --- save data and do evolution step
        # do evolution step
        TwitterModel.evolve()

    # --- STRENGTH CONNECTION PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(network_dynamics['time_str'], network_dynamics['connection_str'],
             marker='o', linestyle='-', color='b', label='Connection Strength')
    plt.title('Connection Strength of Division Over Time')
    plt.xlabel('Time')
    plt.ylabel('Connection Strength')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_main}/connection_strength_over_time.png")
    plt.close()

    if makeUpdateData:
        # --- PLOTTING THE HEATMAP: EIGENVECTOR CENTRALITY
        plt.figure(figsize=(10, 8))
        sns.heatmap(network_dynamics['eigVec_central_mat'], cmap="viridis")
        plt.title("Eigenvector Centrality Evolution Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Node Index")
        plt.savefig(f"{output_main}/eigenvector_centrality_over_time.png")
        plt.close()

        # --- PLOTTING THE HEATMAP: CLOSENESS CENTRALITY
        plt.figure(figsize=(10, 8))
        sns.heatmap(network_dynamics['closeness_central_mat'], cmap="viridis")
        plt.title("Closeness Centrality Evolution Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Node Index")
        plt.savefig(f"{output_main}/closeness_centrality_over_time.png")
        plt.close()

        # --- Creating a multi-plot layout
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Adjust size as needed

        # Plotting each metric in a subplot
        axs[0, 0].plot(network_dynamics['time_dyn'], network_dynamics['average_path_length'],
                       marker='o', linestyle='-', color='blue')
        axs[0, 0].set_title('Average Shortest Path Length')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Path Length')

        axs[0, 1].plot(network_dynamics['time_dyn'], network_dynamics['diameter'],
                       marker='o', linestyle='-', color='red')
        axs[0, 1].set_title('Diameter')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Diameter')

        axs[1, 0].plot(network_dynamics['time_dyn'], network_dynamics['average_clustering'],
                       marker='o', linestyle='-', color='green')
        axs[1, 0].set_title('Average Clustering Coefficient')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylabel('Clustering Coefficient')

        fig.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(f"{output_main}/network_dynamics.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run network simulation.")
    parser.add_argument("--radical_members", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--probability", type=float, required=True)
    parser.add_argument("--val_D", type=float, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--time", type=int, required=True)

    args = parser.parse_args()
    main(args.radical_members, args.k, args.probability, args.val_D, args.run, args.time)
