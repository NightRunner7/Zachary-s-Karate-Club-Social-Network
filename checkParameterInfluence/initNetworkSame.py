import os
import sys
import time
# --- IMPORT FROM FILES
# Add the parent directory to the Python module search path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from TwitterRadicalizationModel import TwitterRadicalizationModel
from initialGraph.watts_NS_UW import create_graph


# --- SETTINGS OF INITIAL GRAPH: USER
members = 1000
radical_members = 162
k = 50
probability = 0.02
# Weights distribution, set zero if they not normal distributed
mean = 0.5
std_dev = 0.05

# --- SETTING OF SIMULATION: USER
# basic setting
number_runs = 3

# settings of dynamic evolution: CLASS
val_D = 5.0
val_beta = 10
val_dt = 0.001

# settings of dynamic evolution: EVOLUTION
time_end = 80
timeSteps = int((time_end / val_dt) * (5 / val_D) * (10 / val_beta))
timeStepsToCheck = int((1 / val_dt) * (5 / val_D) * (10 / val_beta))

# --- SCAN OVER RUNS
# create initial graph
init_network = create_graph(members, radical_members, k, probability,
                            set_affiliation_choice=False, mean=mean, std_dev=std_dev)

for run in range(0, number_runs):
    # set time of reaching stable phase, which have been reset in each run
    time_moment = 0.0

    # --- CREATE MODEL
    TwitterModel = TwitterRadicalizationModel(init_network, D=val_D, beta=val_beta, dt=val_dt)

    # --- EVOLVE NETWORK
    start_time = time.time()  # Start timing the simulation for performance measurement.

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
    phase_val = TwitterModel.return_phase_value()
    end_time = time.time()  # Capture the end time
    print("-------------------------------------------------------------------------------------------------------")
    print("time of simulation:", time_moment, "phase_val in the end of simulation:", phase_val)
    print(f"Execution time: {end_time - start_time} seconds")
