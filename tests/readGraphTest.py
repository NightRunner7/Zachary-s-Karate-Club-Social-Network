import pickle
import auxiliaryFunctions as auxFun

# Replace 'path_to_your_file.pkl' with the actual path to your PKL file
file_path = '../analysisOpinionEvolution/Results/Watts-p0.02-Deff0.5/Parameters-lead-to-All-Phases/Watts-NS-UW-N1000-Nrad210-k170-p0.02-mean0.5-std0.05-D5.0-Deff0.5-dt0.001-run2/saveGraph/network_at_199.0.pkl'

# Open the file in binary mode and load the data
with open(file_path, 'rb') as file:
    graph = pickle.load(file)

# Now 'data' contains the deserialized Python object
print(graph[0])

# Accessing the state of a specific node
for node_id in range(0, 1000):
    node_state = graph.nodes[node_id]['state']
    print(f"The state of node {node_id} is {node_state}")

print(len(graph.nodes))


time_values = [time for _, time in graph.nodes(data='state', default='Not Available')]
print(time_values)

phase_val = auxFun.find_the_phase(graph)
print("phase of out network is:", phase_val)
