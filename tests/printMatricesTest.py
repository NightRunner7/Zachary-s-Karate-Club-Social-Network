import time
# --- IMPORT FROM FILES
from ZaharyEvolutionModel import ZaharyEvolutionModel, ZaharyEvolutionModelMatrix
import networkx as nx
import numpy as np
from pprint import pprint

# --- FUNCTIONS
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

# --- SETTINGS OF SIMULATION
changeInitNetwork = False
timeStepsDraw = 10
timeSteps = 5500
val_D = 5
val_beta = 10
val_dt = 0.01

# Create network
NetworkModelSimple = ZaharyEvolutionModel(D=val_D, beta=val_beta, dt=val_dt)
NetworkModelSimple.change_init_network_publication()
NetworkModelMatrix = ZaharyEvolutionModelMatrix(D=val_D, beta=val_beta, dt=val_dt)
NetworkModelMatrix.change_init_network_publication()

# --- print how look different matrices
networkMat = NetworkModelMatrix.return_network()
w_mat = nx.to_numpy_array(networkMat)
adj_mat = w_mat * 10  # see definition in the ZaharysEvolution Model
s_vec = NetworkModelMatrix.return_state_vector()

print("adj_mat:")
matprint(adj_mat)

print()
print("w_mat:")
matprint(w_mat)

print()
print("play with state vector and weight matrices: s_vec * w_mat * s_vec.T")
ins_member = np.where(s_vec >= 0.5, 1, 0)
off_member = np.where(s_vec < 0.5, 1, 0)

ins_weights = ins_member @ w_mat @ ins_member.T
off_weights = off_member @ w_mat @ off_member.T
sum_weights = np.sum(w_mat)
print("ins_weights: ", ins_weights)
print("ins_weights: ", off_weights)
print("sum_weights: ", sum_weights)

