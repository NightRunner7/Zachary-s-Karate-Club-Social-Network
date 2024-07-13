import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from time import perf_counter
# --- IMPORT FROM FILES
import auxiliaryFunctions as auxFun
from TwitterRadicalizationModel import TwitterRadicalizationModel

# --- SETS THE INITIAL GRAPH: USER
from initialGraph.watts_US_UW import create_graph, create_name
# grap parameters
members = 1000
radical_members = 100
k = 20
probability = 0.1
# Weights distribution, set zero if they nor normal distributed
mean = 0
std_dev = 0
# flags
plot_fit_weight = False
plot_fit_states = False

# create network and name
init_network = create_graph(members, radical_members, k, probability)
name = create_name(members, radical_members, k, probability)

# --- SETTING OF SIMULATION: USER
# basic setting
makePlot = True
run = 1
main_dir = "./ResultsToSegregation"

# settings of dynamic evolution: CLASS
val_D = 5
val_beta = 10
val_dt = 0.001

# settings of dynamic evolution: EVOLUTION
time = 40
timeSteps = int((time / val_dt) * (5 / val_D) * (10 / val_beta))
timeStepsDraw = int((1 / val_dt) * (5 / val_D) * (10 / val_beta))

# --- CREATE MODEL
TwitterModel = TwitterRadicalizationModel(init_network, D=val_D, beta=val_beta, dt=val_dt)

# ---------------------------------- EVOLVE NETWORK ------------------------------------------------------------------ #
for step in range(0, timeSteps):

    if step % timeStepsDraw == 0:
        network = TwitterModel.return_network()
        # average path values:
        start_time = perf_counter()
        average_l = nx.average_shortest_path_length(network, weight='weight')
        end_time = perf_counter()
        print("average shortest path length:", average_l, "time:", end_time - start_time)

        # diameter
        start_time = perf_counter()
        diameter = nx.diameter(network, weight='weight')
        end_time = perf_counter()
        print("real diameter:", diameter, "time:", end_time - start_time)

        # DO NOT GIVES PROPER RESULTS
        # start_time = perf_counter()
        # diameter_approx = nx.approximation.diameter(network)
        # end_time = perf_counter()
        # print("approx diameter:", diameter_approx, "time:", end_time - start_time)

        # Clustering Coefficient: averaged
        average_clustering = nx.average_clustering(network, weight='weight')
        print("average clustering coefficient:", average_l)

        # edge density
        edge_density = nx.density(network)
        print("edge density:", edge_density)

        # Centrality: (read in https://networkx.org/documentation/stable/reference/algorithms/centrality.html)
        centrality = nx.eigenvector_centrality_numpy(network)
        print("averaged eigenvector centrality for the graph:", sum(centrality) / len(centrality))

        # end up analysis
        print("-----------------------------------------------------------------------------------------------")




    # --- save data and do evolution step
    # do evolution step
    TwitterModel.evolve()

