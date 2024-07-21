import os

# ----------------------------------- DEAL WITH DATA ----------------------------------------------------------------- #
def save_phase_scan_over_connectivity(k_arr, phase_arr, time_arr,
                                      str_nrad, directory_name, directory_localization="./OutputPhase"):
    """
    Saves two lists into a text file with a custom format and name, using tab delimiters for readability and
    easy parsing. The first column will contain values from k_arr and the second column from phase_arr.
    Files are saved in a specified directory.

    Args:
        k_arr (list or np.ndarray): A list containing fixed connectivity of the graph.
        phase_arr (list or np.ndarray): A list of last phase value of specific simulation.
        time_arr (list or np.ndarray): A list of time moment reaching the stable or quasi-stable phase.
        str_nrad (str): A value used to customize the file name.
        directory_name (str): The name of the directory where the file will be saved.
        directory_localization (str): The base directory path. Defaults to './OutputPhase'.
    """
    # Ensure the directory exists; create if it doesn't
    full_directory_path = os.path.join(directory_localization, directory_name)
    os.makedirs(full_directory_path, exist_ok=True)

    # Format the filename with the specified directory
    filename = os.path.join(full_directory_path, f"phasePoints_{str_nrad}.txt")

    # Open the file to write
    with open(filename, 'w') as file:
        # Write headers
        file.write("k\ttime\tphase\n")

        # Determine the maximum length of the lists
        max_length = max(len(k_arr), len(phase_arr), len(time_arr))

        # Write data row by row
        for i in range(max_length):
            k_val = k_arr[i] if i < len(k_arr) else ""
            time_val = time_arr[i] if i < len(phase_arr) else ""
            phase_val = phase_arr[i] if i < len(phase_arr) else ""
            file.write(f"{k_val}\t{time_val}\t{phase_val}\n")

    print(f"Data successfully saved in {filename}")
