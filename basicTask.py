"""
This script runs the simulation of the Zachary Evolution Model, handling the initialization,
evolution, and visualization of the network.

Modules:
    - auxiliaryFunctions as auxFun: Contains auxiliary functions for directory management, GIF creation, and plotting.
    - ZaharyEvolutionModel: A class to model the evolution of the Zachary Karate Club network.

Settings:
    - makePlot: Boolean flag to enable/disable plotting.
    - timeStepsDraw: Interval of time steps at which to draw and save the network plot.
    - timeSteps: Total number of time steps to simulate.
    - val_D: Diffusion coefficient for the model.
    - val_beta: Coupling parameter for the model.
    - val_dt: Time step size for the simulation.
    - run: Run number for the simulation (used in naming output directories).

Simulation Steps:
    1. Create output directories.
    2. Initialize the network model.
    3. Perform the evolution of the network over the specified time steps.
    4. Optionally, generate and save plots and a GIF of the network evolution.
    5. Plot the evolution of individual node states over time.
    6. Compare and print the simulated factions of nodes with the actual factions after fission.

Author:
    Krzysztof Szafrański

Last Modified:
    June 23, 2024
"""
import numpy as np
import matplotlib.pyplot as plt
# --- IMPORT FROM FILES
import auxiliaryFunctions as auxFun
from ZaharyEvolutionModel import ZaharyEvolutionModel, ZaharyEvolutionModelMatrix

# --- SETTINGS OF SIMULATION
makePlot = True
changeInitNetwork = True
timeStepsDraw = 100
timeSteps = 5500
val_D = 5
val_beta = 10
val_dt = 0.01
run = 1
connection_strength_arr = np.array([])
time_arr = np.array([])

# Make directory
if changeInitNetwork:
    # output_main = f"./ExtraTask-D-{val_D:.1f}-beta-{val_beta:.1f}-dt-{val_dt}-Run-{run}"
    output_main = f"./MatrixExtraTask-D-{val_D:.1f}-beta-{val_beta:.1f}-dt-{val_dt}-Run-{run}"
else:
    # output_main = f"./D-{val_D:.1f}-beta-{val_beta:.1f}-dt-{val_dt}-Run-{run}"
    output_main = f"./MatrixD-{val_D:.1f}-beta-{val_beta:.1f}-dt-{val_dt}-Run-{run}"

output_evolutionNodes = f"{output_main}/evolutionNodes"
output_evolutionGraph = f"{output_main}/evolutionGraph"

if makePlot:
    auxFun.make_directory(output_main)
    auxFun.make_directory(output_evolutionNodes)
    auxFun.make_directory(output_evolutionGraph)

# Create network
# NetworkModel = ZaharyEvolutionModel(D=val_D, beta=val_beta, dt=val_dt)
NetworkModel = ZaharyEvolutionModelMatrix(D=val_D, beta=val_beta, dt=val_dt)

# change or not change init network
if changeInitNetwork:
    NetworkModel.change_init_network_publication()  # change initial network << !!!

# --- EVOLVE NETWORK
for step in range(0, timeSteps):
    print("I m doing step:", step)
    stepStr = f'{step:04d}'

    # --- draw a network of Zachary's Club
    if makePlot and step % timeStepsDraw == 0:
        network = NetworkModel.return_network()
        auxFun.draw_graph(network, output_evolutionGraph, stepStr)

    # --- strength of connection update data
    connection_strength = NetworkModel.connection_strength_of_division()
    connection_strength_arr = np.append(connection_strength_arr, connection_strength)
    time = step * val_dt
    time_arr = np.append(time_arr, time)

    # --- save data and do evolution step
    # do evolution step
    NetworkModel.evolve_with_update_networkState()

# make gif of network evolution
if makePlot:
    auxFun.make_gif(output_evolutionGraph, "Zahary-evolution")

# --- EVOLUTION OF THE STATE OF NODES
if makePlot:
    networkState = NetworkModel.return_network_evolution_state()
    nodes_list = networkState.keys()
    for node in nodes_list:
        # make a plot: evolution of node state
        auxFun.plot_node_evolution(networkState, node, output_evolutionNodes)

# --- STRENGTH CONNECTION PLOT
plt.figure(figsize=(10, 6))
plt.plot(time_arr, connection_strength_arr, marker='o', linestyle='-', color='b', label='Connection Strength')
plt.title('Connection Strength of Division Over Time')
plt.xlabel('Time')
plt.ylabel('Connection Strength')
plt.grid(True)
plt.legend()
plt.savefig(f"{output_main}/connection_strength_over_time.png")
plt.close()

# --- OUR RESUL VS REAL SITUATION
network = NetworkModel.return_network()

# print("network.nodes.data()", network.nodes.data())
# print("network.edges.data()", network.edges.data())

for i in network.nodes:
    faction = 'Mr. Hi' if network.nodes[i]['state'] > 0.5 else 'Officer'
    print(f"Number node: {i}, faction (my results): {faction}, "
          f"club after fission (real results): {network.nodes[i]['club']}")
