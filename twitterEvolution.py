import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# --- IMPORT FROM FILES
import auxiliaryFunctions as auxFun
from TwitterRadicalizationModel import TwitterRadicalizationModel


# --- SETTING OF SIMULATION
makePlot = True
run = 1
connection_strength_arr = np.array([])
time_arr = np.array([])
# settings of network
members = 4000
radical_members = 2
mean_val = 0.5
std_dev_val = 0.1
# settings of dynamic evolution
timeSteps = 5500
val_D = 5
val_beta = 10
val_dt = 0.01

# Make directory
output_main = f"./Albert-N-{members}-Nrad-{radical_members}-Mean-{mean_val:.1f}-dev-{std_dev_val:.1f}" \
              f"-D-{val_D:.1f}-beta-{val_beta:.1f}-dt-{val_dt}-Run-{run}"

output_evolutionGraph = f"{output_main}/evolutionGraph"
output_evolutionHisto = f"{output_main}/evolutionHisto"

if makePlot:
    auxFun.make_directory(output_main)
    auxFun.make_directory(output_evolutionGraph)

# --- CREATE NETWORK AND MODEL
# create network
network = auxFun.create_normal_weighted_graph(members, radical_members, mean=mean_val, std_dev=std_dev_val)
# create model
TwitterModel = TwitterRadicalizationModel(network, D=val_D, beta=val_beta, dt=val_dt)

# --- PLOTTING THE HISTOGRAM
# Extract the weights from the network
networkTwitter = TwitterModel.return_network()
weights = [data['weight'] for _, _, data in networkTwitter.edges(data=True)]

plt.figure(figsize=(8, 6))
count, bins, ignored = plt.hist(weights, bins=30, density=True, alpha=0.75, color='b')

# Calculate the PDF of the normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_val, std_dev_val)

# Plot the PDF
plt.plot(x, p, 'r', linewidth=2)
title = "Fit results: mean = %.2f,  std_dev = %.2f" % (mean_val, std_dev_val)
plt.title(title)
plt.xlabel('weights')
plt.ylabel('Density')
plt.savefig(f"{output_main}/histogram_at_the_beginning.png")
plt.close()

# ---------------------------------- EVOLVE NETWORK ------------------------------------------------------------------ #
for step in range(0, timeSteps):

    # print("I m doing step:", step)
    stepStr = f'{step:04d}'

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

# --- PLOTTING THE HISTOGRAM
# Extract the weights from the network
networkTwitter = TwitterModel.return_network()
weights = [data['weight'] for _, _, data in networkTwitter.edges(data=True)]

plt.figure(figsize=(8, 6))
count, bins, ignored = plt.hist(weights, bins=30, density=True, alpha=0.75, color='b')

# Calculate the PDF of the normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_val, std_dev_val)

# Plot the PDF
plt.plot(x, p, 'r', linewidth=2)
title = "Fit results: mean = %.2f,  std_dev = %.2f" % (mean_val, std_dev_val)
plt.title(title)
plt.xlabel('weights')
plt.ylabel('Density')
plt.savefig(f"{output_main}/histogram_at_the_end.png")
plt.close()
