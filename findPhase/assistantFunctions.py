import os
import numpy as np

# ----------------------------------- DEAL WITH DATA ----------------------------------------------------------------- #
def save_phase_scan_over_connectivity(k_arr, phase_arr, time_arr, stable_evolution_arr, entropy_arr,
                                      str_nrad, directory_name, directory_localization="./OutputPhase"):
    """
    Saves two lists into a text file with a custom format and name, using tab delimiters for readability and
    easy parsing. The first column will contain values from k_arr and the second column from phase_arr.
    Files are saved in a specified directory.

    Args:
        k_arr (list or np.ndarray): A list containing fixed connectivity of the graph.
        phase_arr (list or np.ndarray): A list of last phase value of specific simulation.
        time_arr (list or np.ndarray): A list of time moment reaching the stable or quasi-stable phase.
        stable_evolution_arr (list or np.ndarray): Flag, to differentiate stable to non-stable network evolution.
        str_nrad (str): A value used to customize the file name.
        directory_name (str): The name of the directory where the file will be saved.
        directory_localization (str): The base directory path. Defaults to './OutputPhase'.
    """
    # Ensure the main directory exists; create if it doesn't
    full_directory_path = os.path.join(directory_localization, directory_name)
    os.makedirs(full_directory_path, exist_ok=True)
    # Ensure the directory for files exists; create if it doesn't
    full_files_path = f"{full_directory_path}/phaseSpace"
    os.makedirs(full_files_path, exist_ok=True)

    # Format the filename with the specified directory
    filename = os.path.join(full_files_path, f"phasePoints_{str_nrad}.txt")

    # Open the file to write
    with open(filename, 'w') as file:
        # Write headers
        file.write("k\ttime\tphase\tstable evolution\n")

        # Determine the maximum length of the lists
        max_length = max(len(k_arr), len(phase_arr), len(time_arr), len(stable_evolution_arr))

        # Write data row by row
        for i in range(max_length):
            k_val = k_arr[i] if i < len(k_arr) else ""
            time_val = time_arr[i] if i < len(time_arr) else ""
            phase_val = phase_arr[i] if i < len(phase_arr) else ""
            stable_evolution = stable_evolution_arr[i] if i < len(stable_evolution_arr) else ""
            entropy_val = entropy_arr[i] if i < len(entropy_arr) else ""
            file.write(f"{k_val}\t{time_val}\t{phase_val}\t{stable_evolution}\t{entropy_val}\n")

    print(f"Data successfully saved in {filename}")

def load_phase_scan_data(file_name, localization="./OutputPhase", with_entropy=False):
    """
    Reads a formatted text file from a specified directory and returns three lists containing the values of each column:
    k-values, time-values, and phase-values.

    Args:
        file_name (str): The name of the file to read.
        localization (str): The directory path where the file is located. Defaults to './OutputPhase'.

    Returns:
        tuple of three lists: (k_values, time_values, phase_values)
    """
    k_values = []
    time_values = []
    phase_values = []
    stable_evo_values = []
    entropy_values = []

    # Construct the full file path
    file_path = os.path.join(localization, file_name)

    try:
        with open(file_path, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) >= 4:  # Ensure there are enough parts in the line
                    k_values.append(float(parts[0]) if parts[0] else None)
                    time_values.append(float(parts[1]) if parts[1] else None)
                    phase_values.append(float(parts[2]) if parts[2] else None)
                    # Convert 'True'/'False' to 1/0
                    stable_evo_values.append(1 if parts[3] == 'True' else 0)
                    if with_entropy:
                        entropy_values.append(float(parts[4]) if parts[4] else None)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return [], [], [], []
    except Exception as e:
        print(f"Error reading from file: {e}")
        return [], [], [], []

    if with_entropy:
        return k_values, time_values, phase_values, stable_evo_values, entropy_values
    else:
        return k_values, time_values, phase_values, stable_evo_values

def compile_phase_data(localization="./OutputPhase", half_max_nrad=250, with_entropy=False):
    """
    Compiles data from multiple files into a 2D matrix where rows correspond to Nrad (extracted from filenames)
    and columns correspond to k-values found in each file.

    Args:
        localization (str): The directory path where the files are located. Defaults to './OutputPhase'.
        half_max_nrad (int): Half of the largest `Nrad` parameter for which we produce .txt files.

    Returns:
        np.ndarray: A 2D numpy array containing the phase data.
    """
    files = [f for f in os.listdir(localization) if f.startswith('phasePoints_') and f.endswith('.txt')]
    max_nrad = half_max_nrad
    max_k = 0

    # Dictionary to hold data temporarily
    phase_data = {}

    for file in files:
        nrad = int(file[len('phasePoints_'):-len('.txt')])
        if with_entropy:
            k_arr, time_arr, phase_arr, stable_evolution_arr, entropy_arr = load_phase_scan_data(file, localization, with_entropy=True)
        else:
            k_arr, time_arr, phase_arr, stable_evolution_arr = load_phase_scan_data(file, localization)

        # all values of k and Nrad are even, so we have to divide those values by 2
        nrad = int(nrad / 2)
        k_arr = [int(k / 2) for k in k_arr]

        if k_arr:
            max_k = max(max_k, max(k_arr))  # Update max_k if necessary
            if with_entropy:
                phase_data[nrad] = (k_arr, phase_arr, time_arr, stable_evolution_arr, entropy_arr)
            else:
                phase_data[nrad] = (k_arr, phase_arr, time_arr, stable_evolution_arr)

    # Initialize the matrix
    phase_matrix = np.full((max_nrad, max_k), np.nan)  # Adjust indices for 0-based
    time_matrix = np.full((max_nrad, max_k), np.nan)  # Adjust indices for 0-based
    stable_evolution_matrix = np.full((max_nrad, max_k), np.nan)  # Adjust indices for 0-based

    if with_entropy:
        entropy_matrix = np.full((max_nrad, max_k), np.nan)  # Adjust indices for 0-based

        # Fill the matrix
        for nrad, (k_arr, phase_arr, time_arr, stable_evolution_arr, entropy_arr) in phase_data.items():
            for k, phase, time, stable, entropy in zip(k_arr, phase_arr, time_arr, stable_evolution_arr, entropy_arr):
                if k is not None and phase is not None:
                    phase_matrix[nrad - 1, int(k) - 1] = phase
                    time_matrix[nrad - 1, int(k) - 1] = time
                    stable_evolution_matrix[nrad - 1, int(k) - 1] = stable
                    entropy_matrix[nrad - 1, int(k) - 1] = entropy

        return phase_matrix, time_matrix, stable_evolution_matrix, entropy_matrix
    else:
        # Fill the matrix
        for nrad, (k_arr, phase_arr, time_arr, stable_evolution_arr) in phase_data.items():
            for k, phase, time, stable in zip(k_arr, phase_arr, time_arr, stable_evolution_arr):
                if k is not None and phase is not None:
                    phase_matrix[nrad - 1, int(k) - 1] = phase
                    time_matrix[nrad - 1, int(k) - 1] = time
                    stable_evolution_matrix[nrad - 1, int(k) - 1] = stable

        return phase_matrix, time_matrix, stable_evolution_matrix
