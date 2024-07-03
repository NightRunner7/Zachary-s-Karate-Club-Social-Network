import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def generate_sbm(N, p1, p2, p_intra, p_inter):
    sizes = [int(N * p1), int(N * p2)]
    p_matrix = [[p_intra, p_inter], [p_inter, p_intra]]
    G = nx.stochastic_block_model(sizes, p_matrix)
    return G

N = 100  # total number of nodes
p1 = 0.5  # proportion of nodes in block 1
p2 = 0.5  # proportion of nodes in block 2
p_intra = 0.1  # high probability of edges within blocks
p_inter = 0.01  # low probability of edges between blocks

G = generate_sbm(N, p1, p2, p_intra, p_inter)

# Draw the graph
plt.figure(figsize=(10, 10))
nx.draw_circular(G, node_size=10)
plt.show()
