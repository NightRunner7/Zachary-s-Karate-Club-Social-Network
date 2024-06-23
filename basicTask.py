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
# --- IMPORT FROM FILES
import auxiliaryFunctions as auxFun
from ZaharyEvolutionModel import ZaharyEvolutionModel

# --- SETTINGS OF SIMULATION
makePlot = False
timeStepsDraw = 10
timeSteps = 5500
val_D = 5
val_beta = 10
val_dt = 0.01
run = 3

# Make directory
output_main = f"./D-{val_D:.1f}-beta-{val_beta:.1f}-dt-{val_dt}-Run-{run}"
# output_main = f"./ExtraTask-D-{val_D:.1f}-beta-{val_beta:.1f}-dt-{val_dt}-Run-{run}"  # change initial network << !!!
output_evolutionNodes = f"{output_main}/evolutionNodes"

if makePlot:
    auxFun.make_directory(output_main)
    auxFun.make_directory(output_evolutionNodes)

# Create network
NetworkModel = ZaharyEvolutionModel()
# NetworkModel.change_init_network_publication()  # change initial network << !!!

# --- EVOLVE NETWORK
for step in range(0, timeSteps):
    print("I m doing step:", step)
    stepStr = f'{step:04d}'

    # draw a network of Zachary's Club
    if makePlot and step % timeStepsDraw == 0:
        network = NetworkModel.return_network()
        auxFun.draw_graph(network, output_main, stepStr)

    # do evolution step
    NetworkModel.evolve()
    # save network state of this time step
    NetworkModel.save_network_state()

# make gif of network evolution
if makePlot:
    auxFun.make_gif(output_main, "Zahary-evolution")

# --- EVOLUTION OF THE STATE OF NODES
if makePlot:
    networkState = NetworkModel.return_network_evolution_state()
    nodes_list = networkState.keys()
    for node in nodes_list:
        # make a plot: evolution of node state
        auxFun.plot_node_evolution(networkState, node, output_evolutionNodes)

# --- OUR RESUL VS REAL SITUATION
network = NetworkModel.return_network()

# print("network.nodes.data()", network.nodes.data())
# print("network.edges.data()", network.edges.data())

for i in network.nodes:
    faction = 'Mr. Hi' if network.nodes[i]['state'] > 0.5 else 'Officer'
    print(f"Number node: {i}, faction (my results): {faction}, "
          f"club after fission (real results): {network.nodes[i]['club']}")
