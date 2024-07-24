import os
import sys
import argparse
import numpy as np
# --- IMPORT FROM FILES
from assistantFunctions import save_phase_scan_over_connectivity
# Add the parent directory to the Python module search path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import config as cfg
from TwitterRadicalizationModel import TwitterRadicalizationModel
from initialGraph.watts_NS_UW import create_graph, create_name


# ---------------------------------------- MAIN FUNCTION ------------------------------------------------------------- #
def main(radical_members, probability, val_D, val_Deff, run, time):
    # --- FIXED PARAMETERS: USER
    sim_config = {
        # Base parameters
        'members': 1000,
        'p': probability,
        # Weights distribution, set zero if they not normal distributed
        'mean': 0.5,
        'std_dev': 0.05,
        # Diffusion and effective diffusion
        'D': val_D,
        'Deff': val_Deff,
        # Other
        'run': run
    }

    # Manage time settings
    time_end = time
    dt = 0.001
    number_of_plots_and_checks = time

    # set the range of connectivity
    half_k = np.arange(1, 150, 1, dtype=int)
    k_arr = 2 * half_k

    # --- FIXED PARAMETERS: AUTOMATIC
    # set beta parameter
    sim_config = cfg.set_network_evolution_parameters(sim_config, sim_config['Deff'], diffusion=sim_config['D'])

    # set time step and ending time
    sim_config = cfg.adjust_time_for_diffusion(sim_config,
                                               sim_config['D'],
                                               base_time_end=time_end,
                                               base_dt=dt,
                                               check_interval=number_of_plots_and_checks,
                                               draw_interval=number_of_plots_and_checks,
                                               update_interval=number_of_plots_and_checks)
    # collect data and path/name
    phase_arr = np.array([])
    time_moment_arr = np.array([])
    stable_evolution_arr = np.array([])
    directory_name = create_name(sim_config=sim_config)

    # update value of radical members
    sim_config['radical_members'] = radical_members
    str_nrad = f"{sim_config['radical_members']}"

    # --- SCAN OVER CONNECTIVITY
    for k_val in k_arr:
        # --- PREPARE RUN: WHAT PARAMETERS CHANGE IN THESE RUNS
        sim_config['k'] = k_val

        # --- PREPARE RUN: BASIC SETTINGS
        # set time of reaching stable phase, which have been reset in each connectivity (k) step
        time_moment = 0.0

        # --- CREATE MODEL
        init_network = create_graph(set_affiliation_choice=False, sim_config=sim_config)

        TwitterModel = TwitterRadicalizationModel(init_network,
                                                  D=sim_config['D'],
                                                  beta=sim_config['beta'],
                                                  dt=sim_config['dt'])

        # --- EVOLVE NETWORK
        for step in range(sim_config['timeSteps']):
            # Periodically check the network's phase status: stop simulation if achieve stable phase.
            if step % sim_config['timeStepsToCheck'] == 0:
                time_moment = step * sim_config['dt']
                stop_simulation_flag = TwitterModel.stop_simulation_criteria(time_moment)
                if stop_simulation_flag:
                    break

            # Perform an evolution step in the model.
            TwitterModel.evolve()

        # --- STORE DATA
        phase_val = TwitterModel.return_phase_value()
        phase_arr = np.append(phase_arr, phase_val)
        stable_evolution = TwitterModel.return_stable_evolution()
        stable_evolution_arr = np.append(stable_evolution_arr, stable_evolution)
        time_moment_arr = np.append(time_moment_arr, time_moment)
        print("phase_val in the end of simulation:", phase_val, "time of ending simulation:", time_moment)

    # save our data
    save_phase_scan_over_connectivity(k_arr, phase_arr, time_moment_arr, stable_evolution_arr,
                                      str_nrad, directory_name, directory_localization="./OutputPhase")


# ---------------------------------------- RUN VIA TERMINAL ---------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run network simulation.")
    parser.add_argument("--radical_members", type=int, required=True)
    parser.add_argument("--probability", type=float, required=True)
    parser.add_argument("--val_D", type=float, required=True)
    parser.add_argument("--val_Deff", type=float, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--time", type=int, required=True)

    args = parser.parse_args()
    main(args.radical_members, args.probability, args.val_D, args.val_Deff, args.run, args.time)
