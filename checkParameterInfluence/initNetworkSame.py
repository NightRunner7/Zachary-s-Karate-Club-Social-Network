import os
import sys
import time
# --- IMPORT FROM FILES
# Add the parent directory to the Python module search path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import config as cfg
from TwitterRadicalizationModel import TwitterRadicalizationModel
from initialGraph.watts_NS_UW import create_graph


# --- SETTINGS OF INITIAL GRAPH: USER
sim_config = {
    # base parameters
    'members': 1000,
    'radical_members':  28,
    'k': 12,
    'p': 0.02,
    # Weights distribution, set zero if they not normal distributed
    'mean': 0.5,
    'std_dev': 0.05
    }
# create network and name
init_network = create_graph(set_affiliation_choice=False, sim_config=sim_config)

# --- SETTING OF SIMULATION: USER
# basic setting
number_runs = 3

# settings of dynamic evolution: CLASS
Deff = 5/10
sim_config = cfg.set_network_evolution_parameters(sim_config, Deff, diffusion=5.0)
sim_config = cfg.adjust_time_for_diffusion(sim_config, sim_config['D'], base_time_end=80, base_dt=0.005)

# --- SCAN OVER RUNS
for run in range(0, number_runs):
    # set time of reaching stable phase, which have been reset in each run
    time_moment = 0.0

    # --- CREATE MODEL
    TwitterModel = TwitterRadicalizationModel(init_network.copy(),
                                              D=sim_config['D'],
                                              beta=sim_config['beta'],
                                              dt=sim_config['dt'])  # COPY NEEDED !!!

    # --- EVOLVE NETWORK
    start_time = time.time()  # Start timing the simulation for performance measurement.

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
    end_time = time.time()  # Capture the end time
    print("-------------------------------------------------------------------------------------------------------")
    print("time ending of simulation:", time_moment, "phase_val in the end of simulation:", phase_val)
    print(f"Execution time: {end_time - start_time} seconds")
