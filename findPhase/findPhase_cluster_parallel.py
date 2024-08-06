import os
import gc
import sys
import argparse
import numpy as np
from multiprocessing import Pool
# --- IMPORT FROM FILES
from assistantFunctions import save_phase_scan_over_connectivity
# Add the parent directory to the Python module search path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
# --- SET MODEL !!!
from NetworkEvolutionDeff import NetworkEvolutionDeff, \
    set_network_evolution_parameters, adjust_time_for_diffusion
# --- SET INITIAL GRAP !!!
from initialGraph.watts_NS_UW import create_graph, create_name


# ---------------------------------------- SIMULATE WITH FIXED PARAMS ------------------------------------------------ #
def simulate_phase_point(radical_members, k, probability, val_D, val_Deff, run, time):
    # --- FIXED PARAMETERS: USER
    sim_config = {
        # Base parameters
        'members': 1000,
        'radical_members': radical_members,
        'k': k,
        'p': probability,
        # Weights distribution, set zero if they not normal distributed
        'mean': 0.5,
        'std_dev': 0.05,
        # Diffusion and effective diffusion
        'Deff': val_Deff,
        # Other
        'run': run
    }

    # Manage time settings
    time_end = time
    dt = 0.001
    time_moment = 0.0  # set time of reaching stable phase

    # --- INITIALIZE MODEL
    sim_config = set_network_evolution_parameters(sim_config=sim_config,
                                                  effective_diffusion=sim_config['Deff'])

    sim_config = adjust_time_for_diffusion(sim_config=sim_config,
                                           base_time_end=time_end,
                                           base_dt=dt,
                                           check_interval=time_end,
                                           draw_interval=time_end,
                                           update_interval=time_end)

    init_network = create_graph(set_affiliation_choice=False,
                                sim_config=sim_config)

    TwitterModel = NetworkEvolutionDeff(init_network,
                                        Deff=sim_config['Deff'],
                                        dt=sim_config['dt'])

    # --- EVOLVE NETWORK
    for step in range(sim_config['timeSteps']):
        # Periodically check the network's phase status, stop simulation if: achieve stable phase, non-stable network.
        if step % sim_config['timeStepsToCheck'] == 0:
            time_moment = step * sim_config['dt']
            stop_simulation_flag = TwitterModel.stop_simulation_criteria(time_moment)
            if stop_simulation_flag:
                break

        # Perform an evolution step in the model.
        TwitterModel.evolve()

    # --- CLEANUP AND DATA COLLECTION
    phase = TwitterModel.return_phase_value()
    stable_evolution = TwitterModel.return_stable_evolution()
    del init_network, TwitterModel
    gc.collect()

    return [time_moment, phase, stable_evolution]

# ---------------------------------------- MAIN FUNCTION ------------------------------------------------------------- #
def main(radical_members, probability, val_D, val_Deff, run, time, num_processes):
    # Collect data and path/name
    str_nrad = f"{radical_members}"
    directory_name = create_name(members=1000, probability=probability,
                                 mean=0.5, std_dev=0.05,
                                 D=val_D, Deff=val_Deff,
                                 run_number=run)

    # Set the range of connectivity
    k_arr = np.arange(2, 302, 2)
    # k_arr = np.arange(30, 34, 2)
    params = [(radical_members, k, probability, val_D, val_Deff, run, time) for k in k_arr]

    with Pool(processes=num_processes) as pool:  # Utilize as many cores as are beneficial
        results_list = pool.starmap(simulate_phase_point, params)

    # Prepare arrays to store results
    time_moment_arr = np.array([result[0] for result in results_list])
    phase_arr = np.array([result[1] for result in results_list])
    stable_evolution_arr = np.array([result[2] for result in results_list])

    # Save our data
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
    parser.add_argument("--num_processes", type=int, default=20, help="Number of processes to use in the Pool")

    args = parser.parse_args()
    main(args.radical_members, args.probability, args.val_D, args.val_Deff, args.run, args.time, args.num_processes)
