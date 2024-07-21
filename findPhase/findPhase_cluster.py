import os
import sys
import argparse
import numpy as np
# --- IMPORT FROM FILES
from assistantFunctions import save_phase_scan_over_connectivity
# Add the parent directory to the Python module search path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from TwitterRadicalizationModel import TwitterRadicalizationModel
from initialGraph.watts_NS_UW import create_graph, return_name


# ---------------------------------------- MAIN FUNCTION ------------------------------------------------------------- #
def main(radical_members, probability, val_D, run, time):
    # --- SETTINGS OF INITIAL GRAPH: USER
    members = 1000
    half_k = np.arange(1, 151, 1, dtype=int)
    k_arr = 2 * half_k
    # Weights distribution, set zero if they not normal distributed
    mean = 0.5
    std_dev = 0.05
    # necessary strings
    base_name = return_name()
    str_nrad = f"{radical_members}"

    # --- SETTING OF SIMULATION: USER
    # settings of dynamic evolution: CLASS
    val_beta = 10
    val_dt = 0.001

    # settings of dynamic evolution: EVOLUTION
    timeSteps = int((time / val_dt) * (5 / val_D) * (10 / val_beta))
    timeStepsToCheck = int((1 / val_dt) * (5 / val_D) * (10 / val_beta))

    # --- SETTING OF SIMULATION: AUTOMATIC
    # collect data and path/name
    phase_arr = np.array([])
    time_moment_arr = np.array([])
    directory_name = f"{base_name}-N{members}-p{probability}-mean{mean}-std{std_dev}-Run{run}"

    # --- SCAN OVER CONNECTIVITY
    for k_val in k_arr:
        # set time of reaching stable phase, which have been reset in each connectivity (k) step
        time_moment = 0.0

        # --- CREATE MODEL
        init_network = create_graph(members, radical_members, k_val, probability,
                                    set_affiliation_choice=False, mean=mean, std_dev=std_dev)
        TwitterModel = TwitterRadicalizationModel(init_network, D=val_D, beta=val_beta, dt=val_dt)

        # --- EVOLVE NETWORK
        for step in range(timeSteps):
            # Periodically check the network's phase status: stop simulation if achieve stable phase.
            if step % timeStepsToCheck == 0:
                time_moment = step * val_dt
                stop_simulation_flag = TwitterModel.stop_simulation_criteria(time_moment)
                if stop_simulation_flag:
                    break

            # Perform an evolution step in the model.
            TwitterModel.evolve()

        # --- STORE DATA
        # Store value of phase and time of reaching stable phase
        phase_val = TwitterModel.return_phase_value()
        phase_arr = np.append(phase_arr, phase_val)
        time_moment_arr = np.append(time_moment_arr, time_moment)

    # save our data
    save_phase_scan_over_connectivity(k_arr, phase_arr, time_moment_arr,
                                      str_nrad, directory_name, directory_localization="./OutputPhase")

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
