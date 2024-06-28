import numpy as np
from scipy.stats import norm
import networkx as nx
import matplotlib.pyplot as plt
# --- IMPORT FROM FILES
import auxiliaryFunctions as auxFun


# --- Example of usage different graphs
showPlot = False
n = 20
er = nx.erdos_renyi_graph(n, 0.7)
ws = nx.newman_watts_strogatz_graph(n, 4, 0.1)
ba = nx.barabasi_albert_graph(n, 4)

# --- PLOTS
if showPlot:
    plt.figure(figsize=(10, 10))
    # erdos_renyi_graph
    plt.subplot(2, 2, 1)
    nx.draw_circular(er, node_size=10)
    # newman_watts_strogatz_graph
    plt.subplot(2, 2, 2)
    nx.draw_circular(ws, node_size=10)
    # barabasi_albert_graph
    plt.subplot(2, 2, 3)
    nx.draw_circular(ba, node_size=10)

    plt.show()

print("ba.nodes.data", ba.nodes.data())
print("ba.edges.data", ba.edges.data())
print("ba.number_of_edges()", ba.number_of_edges())

# --- Work a bit with barabasi_albert_graph
members = 400
radical_members = 4

network_twitter = nx.barabasi_albert_graph(members, radical_members)

# Draw the graph
plt.figure(figsize=(10, 10))
nx.draw_circular(network_twitter, node_size=10)
plt.show()


# ################### SETTING OF SIMULATION
# settings of network
members = 500
radical_members = 4
mean_val = 0.5
std_dev_val = 0.1

# create network which we're going to check
network = auxFun.create_normal_weighted_graph(members, radical_members, mean=mean_val, std_dev=std_dev_val)


# --- PLOTTING THE HISTOGRAM
# Extract the weights from the network
weights = [data['weight'] for _, _, data in network.edges(data=True)]

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
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

