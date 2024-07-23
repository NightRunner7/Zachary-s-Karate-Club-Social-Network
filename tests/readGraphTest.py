import pickle
import auxiliaryFunctions as auxFun

# Replace 'path_to_your_file.pkl' with the actual path to your PKL file
file_path = '../analysisOpinionEvolution/Results/ResultsToSegregation/Watts-NS-UW-N1000-Nrad160-k40-p0.01-mean0.5-std0.05-D5.0-beta10.0-dt0.001-Run7/saveGraph/network_at__99.0.pkl'

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
