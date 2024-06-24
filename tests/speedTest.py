import time
# --- IMPORT FROM FILES
from ZaharyEvolutionModel import ZaharyEvolutionModel, ZaharyEvolutionModelMatrix

# --- SETTINGS OF SIMULATION
changeInitNetwork = False
timeStepsDraw = 10
timeSteps = 5500
val_D = 5
val_beta = 10
val_dt = 0.01

# Create network
NetworkModelSimple = ZaharyEvolutionModel(D=val_D, beta=val_beta, dt=val_dt)
NetworkModelMatrix = ZaharyEvolutionModelMatrix(D=val_D, beta=val_beta, dt=val_dt)

# change or not change init network
if changeInitNetwork:
    NetworkModelSimple.change_init_network_publication()  # change initial network << !!!
    NetworkModelMatrix.change_init_network_publication()  # change initial network << !!!


# --- EVOLVE NETWORK SIMPLE
start_time = time.time()  # Start time

for step in range(0, timeSteps):
    # do evolution step
    NetworkModelSimple.evolve()
    # save network state of this time step
    # NetworkModelSimple.save_network_state()

end_time = time.time()  # End time
elapsed_time = end_time - start_time  # Calculate elapsed time
print("Elapsed time:", elapsed_time, "seconds: simple model")


# --- EVOLVE NETWORK MATRIX
start_time = time.time()  # Start time

for step in range(0, timeSteps):
    # do evolution step
    NetworkModelSimple.evolve()
    # save network state of this time step
    # NetworkModelSimple.save_network_state()

end_time = time.time()  # End time
elapsed_time = end_time - start_time  # Calculate elapsed time
print("Elapsed time:", elapsed_time, "seconds: matrix model")
