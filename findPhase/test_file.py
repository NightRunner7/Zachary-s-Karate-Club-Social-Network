import os
import numpy as np

def load_data_from_file(str_parameters, directory="OutputPhase"):
    """
    Reads the data from a text file and converts it into two lists of numbers.

    Args:
        str_parameters (str): The string used in the file name, to identify the correct file.
        directory (str): The directory where the file is stored.

    Returns:
        tuple: A tuple containing two lists (k_arr, phase_arr).
    """
    # Construct the file name from the provided parameters
    filename = os.path.join(directory, f"{str_parameters}.txt")

    # Check if the file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No file found at {filename}")

    k_arr = []
    phase_arr = []

    # Open and read the file
    with open(filename, 'r') as file:
        for line in file:
            # Check if the line contains k_arr or phase_arr data
            if line.startswith('k_arr:'):
                # Split the line on tabs, skip the first element (label), and convert to floats or ints
                k_arr = [float(num) for num in line.split('\t')[1:]]
            elif line.startswith('phase_arr:'):
                # Similar process for phase_arr
                phase_arr = [float(num) for num in line.split('\t')[1:]]

    return k_arr, phase_arr

# Example usage
k_arr_loaded = np.array([])
phase_arr_loaded = np.array([])

try:
    str_parameters = "phasePoints_Watts-NS-UW-N1000-Nrad162-p0.02-mean0.5-std0.05-Run1"
    directory = ".\ResultsPhase"
    k_arr_loaded, phase_arr_loaded = load_data_from_file(str_parameters, directory=directory)
except FileNotFoundError as e:
    print(e)


for i in range(0, len(k_arr_loaded)):
    print("connectivity (k):", k_arr_loaded[i], "phase value:", phase_arr_loaded[i])


