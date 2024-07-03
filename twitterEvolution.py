import numpy as np
import matplotlib.pyplot as plt
# --- IMPORT FROM FILES
import auxiliaryFunctions as auxFun
import initialGraph as iniGra
from TwitterRadicalizationModel import TwitterRadicalizationModel

# ----- SETTINGS OF INITIAL GRAPH
# Select initial graph, Name pattern: Initial graph type + weight distribution type
name_init_graph = 'AlbertNormal'

# Configuration dictionary for different network types
network_config = {}
if name_init_graph == 'AlbertNormal':
    # set value
    members = 1000
    popular_members = 80
    radical_members = 80
    mean_val = 0.5
    std_dev_val = 0.1
    # set dictionary
    network_config = {
        'name': 'AlbertNormal',
        'members': members,
        'radical_members': radical_members,
        'mean': mean_val,
        'std_dev': std_dev_val,
        'network': iniGra.create_normal_weighted_graph_v2(members, radical_members, popular_members,
                                                          mean=mean_val, std_dev=std_dev_val),
        # 'network': iniGra.create_normal_weighted_graph(members, radical_members, popular_members,
        #                                                mean=mean_val, std_dev=std_dev_val),
        'parameters': [members, radical_members, popular_members, mean_val, std_dev_val]
    }
elif name_init_graph == 'AlbertConst':
    # set value
    members = 1000
    radical_members = 2
    # set dictionary
    network_config = {
        'name': 'AlbertConst',
        'members': members,
        'radical_members': radical_members,
        'network': iniGra.create_constant_weighted_graph(members, radical_members),
        'parameters': [members, radical_members]
    }
elif name_init_graph == 'Zahary':
    # set dictionary
    network_config = {
        'name': 'AlbertConst',
        'network': iniGra.create_zahary_club_graph()
    }
elif name_init_graph == 'ErdosBasic':
    # set value
    members = 1000
    radical_members = 100
    probability = 0.007
    # set dictionary
    network_config = {
        'name': 'ErdosBasic',
        'members': members,
        'radical_members': radical_members,
        'probability': probability,
        'network': iniGra.create_erdos_basic_graph(members, radical_members, probability),
        'parameters': [members, radical_members, probability]
    }

# ----- SETTING OF SIMULATION
# --- USER
# basic setting
makePlot = True
run = 1
output_main = "."
connection_strength_arr = np.array([])
time_arr = np.array([])

# settings of dynamic evolution: CLASS
val_D = 5
val_beta = 10
val_dt = 0.001

# settings of dynamic evolution: EVOLUTION
time = 60
timeSteps = int((time / val_dt) * (5 / val_D) * (10 / val_beta))
timeStepsDraw = int((1 / val_dt) * (5 / val_D) * (10 / val_beta))

# --- AUTOMATIC
# Set main directory
if name_init_graph == 'AlbertNormal':
    members, radical_members, popular_members, mean_val, std_dev_val = network_config['parameters']
    output_main = f"./{name_init_graph}-N-{members}-Nrad-{radical_members}-Npop-{popular_members}" \
                  f"-Mean-{mean_val:.1f}-dev-{std_dev_val:.1f}" \
                  f"-D-{val_D:.1f}-beta-{val_beta:.1f}-dt-{val_dt}-Run-{run}"
elif name_init_graph == 'AlbertConst':
    members, radical_members = network_config['parameters']
    output_main = f"./{name_init_graph}N-{members}-Nrad-{radical_members}" \
                  f"-D-{val_D:.1f}-beta-{val_beta:.1f}-dt-{val_dt}-Run-{run}"
elif name_init_graph == 'Zahary':
    output_main = f"./{name_init_graph}" \
                  f"-D-{val_D:.1f}-beta-{val_beta:.1f}-dt-{val_dt}-Run-{run}"
elif name_init_graph == 'ErdosBasic':
    members, radical_members, probability = network_config['parameters']
    output_main = f"./{name_init_graph}-N-{members}-Nrad-{radical_members}-p-{probability}" \
                  f"-D-{val_D:.1f}-beta-{val_beta:.1f}-dt-{val_dt}-Run-{run}"

# Make directory
output_evolutionGraph = f"{output_main}/evolutionGraph"
output_evolutionHistoWeights = f"{output_main}/evolutionHistoWeights"
output_evolutionHistoState = f"{output_main}/evolutionHistoState"

if makePlot:
    auxFun.make_directory(output_main)
    auxFun.make_directory(output_evolutionGraph)
    auxFun.make_directory(output_evolutionHistoWeights)
    auxFun.make_directory(output_evolutionHistoState)

# ----- CREATE MODEL
# create model
TwitterModel = TwitterRadicalizationModel(network_config['network'], D=val_D, beta=val_beta, dt=val_dt)

# ---------------------------------- EVOLVE NETWORK ------------------------------------------------------------------ #
for step in range(0, timeSteps):
    stepStr = f'{step:04d}'

    # --- draw a network of Twitter Network
    if makePlot and step % timeStepsDraw == 0:
        print("I m doing step:", step)
        network = TwitterModel.return_network()
        # auxFun.draw_graph_kamada_kawai(network, output_evolutionGraph, stepStr)
        # auxFun.draw_graph_spectral(network, output_evolutionGraph, stepStr)
        # auxFun.draw_graph_spectral_with_communities(network, output_evolutionGraph, stepStr)
        # auxFun.draw_graph(network, output_evolutionGraph, stepStr)

    # --- draw a histogram weights of Twitter Network
    if makePlot and step % timeStepsDraw == 0:
        name = f'histogram_at_step_{stepStr}'
        # different histograms
        if name_init_graph == 'AlbertNormal':
            mean_val = network_config['mean']
            std_dev_val = network_config['std_dev']
            iniGra.histogram_normal_weighted_graph(TwitterModel,
                                                   mean_val=mean_val, std_dev_val=std_dev_val,
                                                   output_path=output_evolutionHistoWeights, file_name=name)
        else:
            iniGra.histogram_weighted_graph(TwitterModel,
                                            output_path=output_evolutionHistoWeights, file_name=name)

    # --- draw a histogram states of Twitter Network
    if makePlot and step % timeStepsDraw == 0:
        name = f'histogram_at_step_{stepStr}'
        iniGra.histogram_states_graph(TwitterModel,
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
