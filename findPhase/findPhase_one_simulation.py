import time
# --- IMPORT FROM FILES
from TwitterRadicalizationModel import TwitterRadicalizationModel

# --- SETS THE INITIAL GRAPH: USER
from initialGraph.watts_NS_UW import create_graph, create_name
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
name = create_name(members, radical_members, k, probability, mean, std_dev)

# --- SETTING OF SIMULATION: USER
# basic setting
counter_of_domination_state = 0
counter_of_fullDivision_state = 0
counter_of_same_state = 0
phase_val = 0.0

# settings of dynamic evolution: CLASS
val_D = 5
val_beta = 10
val_dt = 0.001

# settings of dynamic evolution: EVOLUTION
time_val = 200  # 80
timeSteps = int((time_val / val_dt) * (5 / val_D) * (10 / val_beta))
timeStepsDraw = int((1 / val_dt) * (5 / val_D) * (10 / val_beta))

# --- CREATE MODEL
TwitterModel = TwitterRadicalizationModel(init_network, D=val_D, beta=val_beta, dt=val_dt)

# ---------------------------------- EVOLVE NETWORK ------------------------------------------------------------------ #
start_time = time.time()  # Start timing the simulation for performance measurement.

for step in range(timeSteps):
    # Periodically check and report the network's phase status.
    if step % timeStepsDraw == 0:
        print("I'm doing step:", step)

        previous_phase_val = phase_val  # Store previous phase value to detect state changes.

        # Calculate the current phase of the network based on defined thresholds and network state.
        phase_val = TwitterModel.find_the_phase(
            epsilon=0.05,
            neutral_width=0.4,
            division_threshold=0.2,
            wall_threshold=0.2
        )
        print("Phase value:", phase_val)

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

# Note the end time and calculate the total duration of the simulation.
end_time = time.time()  # Capture the end time
print(f"Execution time: {end_time - start_time} seconds")

phase_val = TwitterModel.find_the_phase(epsilon=0.05, neutral_width=0.4, division_threshold=0.2, wall_threshold=0.2)
print("phase_val:", phase_val)
