import numpy as np
import matplotlib.pyplot as plt
# --- IMPORT FROM FILES
import auxiliaryFunctions as auxFun
from TwitterRadicalizationModel import TwitterRadicalizationModel

# --- SETS THE INITIAL GRAPH
from initialGraph.zachary_club import create_graph, create_name

# create network and name
init_network = create_graph()
name = create_name()

# --- SETTING OF SIMULATION: USER
# basic setting
makePlot = True
run = 2
main_dir = "./ResultsToSegregation"

# settings of dynamic evolution: CLASS
val_D = 5
val_beta = 10
val_dt = 0.001

# settings of dynamic evolution: EVOLUTION
time = 60
timeSteps = int((time / val_dt) * (5 / val_D) * (10 / val_beta))
timeStepsDraw = int((1 / val_dt) * (5 / val_D) * (10 / val_beta))

# --- SETTING OF SIMULATION: AUTOMATIC
# basic setting
output_main = f"{main_dir}/{name}-D{val_D:.1f}-beta{val_beta:.1f}-dt{val_dt}-Run{run}"
connection_strength_arr = np.array([])
time_arr = np.array([])

# Make directory
output_evolutionGraph = f"{output_main}/evolutionGraph"
output_evolutionHistoWeights = f"{output_main}/evolutionHistoWeights"
output_evolutionHistoState = f"{output_main}/evolutionHistoState"

if makePlot:
    auxFun.make_directory(output_main)
    auxFun.make_directory(output_evolutionGraph)
    auxFun.make_directory(output_evolutionHistoWeights)
    auxFun.make_directory(output_evolutionHistoState)


# --- CREATE MODEL
TwitterModel = TwitterRadicalizationModel(init_network, D=val_D, beta=val_beta, dt=val_dt)

# ---------------------------------- EVOLVE NETWORK ------------------------------------------------------------------ #
for step in range(0, timeSteps):
    stepStr = f'{step:04d}'

    # --- draw a network of Twitter Network
    if makePlot and step % timeStepsDraw == 0:
        print("I m doing step:", step)
        network = TwitterModel.return_network()
        # auxFun.draw_graph_kamada_kawai(network, output_evolutionGraph, stepStr)
        # auxFun.draw_graph_spectral(network, output_evolutionGraph, stepStr)
        auxFun.draw_graph(network, output_evolutionGraph, stepStr)

    # --- draw a histogram weights of Twitter Network
    if makePlot and step % timeStepsDraw == 0:
        name = f'histogram_at_step_{stepStr}'
        auxFun.histogram_weights(TwitterModel, plot_fit=False,
                                 output_path=output_evolutionHistoWeights, file_name=name)

    # --- draw a histogram states of Twitter Network
    if makePlot and step % timeStepsDraw == 0:
        name = f'histogram_at_step_{stepStr}'
        auxFun.histogram_states(TwitterModel, plot_fit=False,
                                output_path=output_evolutionHistoState, file_name=name)

    # --- strength of connection update data
    connection_strength = TwitterModel.connection_strength_of_division()
    connection_strength_arr = np.append(connection_strength_arr, connection_strength)
    time = step * val_dt
    time_arr = np.append(time_arr, time)

    # --- save data and do evolution step
    # do evolution step
    TwitterModel.evolve()

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
