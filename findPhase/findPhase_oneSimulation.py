import time
# --- IMPORT FROM FILES
from TwitterRadicalizationModel import TwitterRadicalizationModel

# --- SETS THE INITIAL GRAPH: USER
from initialGraph.watts_NS_UW import create_graph
# grap parameters
members = 1000
radical_members = 160
k = 50
probability = 0.02
# Weights distribution, set zero if they not normal distributed
mean = 0.5
std_dev = 0.05

# create network and name
init_network = create_graph(members, radical_members, k, probability,
                            set_affiliation_choice=False, mean=mean, std_dev=std_dev)

# --- SETTING OF SIMULATION: USER
# settings of dynamic evolution: CLASS
val_D = 5
val_beta = 10
val_dt = 0.001

# settings of dynamic evolution: EVOLUTION
time_val = 200  # 80
time_moment = 0.0
timeSteps = int((time_val / val_dt) * (5 / val_D) * (10 / val_beta))
timeStepsToCheck = int((1 / val_dt) * (5 / val_D) * (10 / val_beta))

# --- CREATE MODEL
TwitterModel = TwitterRadicalizationModel(init_network, D=val_D, beta=val_beta, dt=val_dt)

# ---------------------------------- EVOLVE NETWORK ------------------------------------------------------------------ #
start_time = time.time()  # Start timing the simulation for performance measurement.

for step in range(timeSteps):
    # Periodically check and report the network's phase status.
    if step % timeStepsToCheck == 0:
        time_moment = step * val_dt
        stop_simulation_flag = TwitterModel.stop_simulation_criteria(time_moment)

        # Calculate the current phase of the network based on defined thresholds and network state.
        phase_val = TwitterModel.return_phase_value()
        print("time of simulation:", time_moment, "phase_val in the end of simulation:", phase_val)

        # stop simulation
        if stop_simulation_flag:
            break

    # Perform an evolution step in the model.
    TwitterModel.evolve()

# Note the end time and calculate the total duration of the simulation.
print("-------------------------------------------------------------------------------------------------------")
end_time = time.time()  # Capture the end time
print(f"Execution time: {end_time - start_time} seconds")

phase_val = TwitterModel.return_phase_value()
print("time of simulation:", time_moment, "phase_val in the end of simulation:", phase_val)
