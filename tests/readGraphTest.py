import pickle

# Replace 'path_to_your_file.pkl' with the actual path to your PKL file
file_path = '../ResultsToSegregation/Watts-NS-UW-N1000-Nrad160-k40-p0.01-mean0.5-std0.05-D5.0-beta10.0-dt0.001-Run7/saveGraph/network_at__0.0.pkl'

# Open the file in binary mode and load the data
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Now 'data' contains the deserialized Python object
print(data[0])

