import os
import sys
import argparse
import numpy as np
# --- IMPORT FROM FILES
# Add the parent directory to the Python module search path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from TwitterRadicalizationModel import TwitterRadicalizationModel
from initialGraph.watts_NS_UW import create_graph, return_name

# ---------------------------------------- HELPFUL FUNCTION ---------------------------------------------------------- #
def save_data_to_file(k_arr, phase_arr, str_parameters, directory="OutputPhase"):
    """
    Saves two lists into a text file with a custom format and name, using tab delimiters for readability and
    easy parsing. Files are saved in a specified directory.

    Args:
        k_arr (list or array): A list of numbers.
        phase_arr (list or array): Another list of numbers.
        str_parameters (str): A value used to customize the file name.
        directory (str): The directory where the file will be saved. Defaults to 'output'.
    """
    # Ensure the directory exists; create if it doesn't
    os.makedirs(directory, exist_ok=True)

    # Format the filename with the specified directory
    filename = os.path.join(directory, f"phasePoints_{str_parameters}.txt")

    # Open the file to write
    with open(filename, 'w') as file:
        # Writing k_arr values
        file.write("k_arr:\t")  # Start with a label and a tab
        file.write("\t".join(map(str, k_arr)))  # Join list elements with tabs
        file.write("\n")  # New line for the next array

        # Writing phase_arr values
        file.write("phase_arr:\t")  # Start with a label and a tab
        file.write("\t".join(map(str, phase_arr)))  # Join list elements with tabs

    print(f"Data successfully saved in {filename}")

# ---------------------------------------- MAIN FUNCTION ------------------------------------------------------------- #
def main(radical_members, probability, val_D, run, time):
    # grap parameters
    members = 1000
    half_k = np.arange(1, 2, 1, dtype=int)
    k_arr = 2 * half_k
    # Weights distribution, set zero if they not normal distributed
    mean = 0.5
    std_dev = 0.05

    # --- SETTING OF SIMULATION: USER
    # settings of dynamic evolution: CLASS
    val_beta = 10
    val_dt = 0.001

    # settings of dynamic evolution: EVOLUTION
    timeSteps = int((time / val_dt) * (5 / val_D) * (10 / val_beta))
    timeStepsToCheck = int((1 / val_dt) * (5 / val_D) * (10 / val_beta))

    # --- SETTING OF SIMULATION: AUTOMATIC
    # basic setting
    counter_of_domination_state = 0
    counter_of_fullDivision_state = 0
    counter_of_same_state = 0
    phase_val = 0.0
    phase_arr = np.array([])
    base_name = return_name()
    str_parameters = f"{base_name}-N{members}-Nrad{radical_members}-p{probability}-mean{mean}-std{std_dev}-Run{run}"

    for k_val in k_arr:
        # --- CREATE MODEL
        # create network
        init_network = create_graph(members, radical_members, k_val, probability,
                                    set_affiliation_choice=False, mean=mean, std_dev=std_dev)

        TwitterModel = TwitterRadicalizationModel(init_network, D=val_D, beta=val_beta, dt=val_dt)

        # ---------------------------------- EVOLVE NETWORK ---------------------------------------------------------- #
        for step in range(timeSteps):
            # Periodically check and report the network's phase status.
            if step % timeStepsToCheck == 0:

                previous_phase_val = phase_val  # Store previous phase value to detect state changes.

                # Calculate the current phase of the network based on defined thresholds and network state.
                phase_val = TwitterModel.find_the_phase(
                    epsilon=0.05,
                    neutral_width=0.4,
                    division_threshold=0.2,
                    wall_threshold=0.2
                )

                # Monitor the occurrence of specific phases to determine when the simulation can be terminated early.
                if phase_val == 2.0:
                    counter_of_fullDivision_state += 1  # Count occurrences of the full division phase.
                if phase_val == 3.0:
                    counter_of_domination_state += 1  # Count occurrences of the domination phase.

                # Check if the phase has remained the same over multiple checks to infer stabilization.
                if previous_phase_val == phase_val and phase_val > 1.0:
                    counter_of_same_state += 1
                else:
                    counter_of_same_state = 0  # Reset counter if the phase changes.

                # Terminate early if certain conditions are met, indicating no further significant evolution.
                if counter_of_fullDivision_state > 1 or counter_of_domination_state > 1:
                    break  # Stop if a phase has occurred multiple times, suggesting dominance or division stability.
                if counter_of_same_state == 10:
                    break  # Stop if the same phase persists across multiple checks, suggesting stabilization.

            # Perform an evolution step in the model.
            TwitterModel.evolve()

        phase_val = TwitterModel.find_the_phase(
            epsilon=0.05,
            neutral_width=0.4,
            division_threshold=0.2,
            wall_threshold=0.2
        )
        print("phase_val in the end of simulation:", phase_val, "connectivity (k):", k_val)

        phase_arr = np.append(phase_arr, phase_val)

    # save our data
    save_data_to_file(k_arr, phase_arr, str_parameters=str_parameters, directory="OutputPhase")

# ---------------------------------------- RUN VIA TERMINAL ---------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run network simulation.")
    parser.add_argument("--radical_members", type=int, required=True)
    parser.add_argument("--probability", type=float, required=True)
    parser.add_argument("--val_D", type=float, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--time", type=int, required=True)

    args = parser.parse_args()
    main(args.radical_members, args.probability, args.val_D, args.run, args.time)
