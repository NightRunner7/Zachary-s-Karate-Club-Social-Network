import numpy as np
import networkx as nx
from pprint import pprint

# ------------ Helpfull functions ------------ #
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

# ------------ test with matrices ------------ #
mat_B = np.mat([
    [1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1]
])

mat_Im = np.identity(7)

mat_AL = mat_B.T @ mat_B - 2 * mat_Im

print(mat_AL)

network = nx.karate_club_graph()
for i in network.nodes:
    print(f"club after fission (real results): {network.nodes[i]['club']}, i: {i}")

ngb_list = list(network.adj[0])
print("ngb_list:", ngb_list)

# state vector
s_vec = np.array([1 if node == 0 else 0 if node == 33 else 0.5 for node in network.nodes])
print("s_vec:", s_vec)

# adjacency matrix
adj_mat = nx.to_numpy_array(network)
print("adj_mat:")
matprint(adj_mat)


# diff
diff = np.abs(s_vec[:, np.newaxis] - s_vec)
print()
print("diff")
matprint(diff)

# networkState
networkState = {node: np.array([state]) for node, state in enumerate(s_vec)}
print("networkState")
print(networkState)

# w_mat
w_mat = np.where(adj_mat > 0, 0.5, 0)  # Initial weights set to 0.5 where there is an edge
print("w_mat")
matprint(w_mat)

mat_1 = w_mat @ s_vec
print("mat_1:", mat_1)



# input two matrices
mat1 = np.mat([
    [1, 6, 5],
    [3, 4, 8],
    [2, 12, 3]
])
mat2 = np.mat([
    [3, 4, 6],
    [5, 6, 7],
    [6, 56, 7]
])

print("mat1:", mat1)
# This will return matrix product of two array
res = mat1 @ mat2
# print resulted matrix
print(res)



