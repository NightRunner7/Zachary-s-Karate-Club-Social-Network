import numpy as np
from assistantFunctions import compile_phase_data, compile_phase_data_vol2
import matplotlib.pyplot as plt

# --- Flags
savePlot = True

# --- Read the data from directory
localization_main = "./ResultsPhase/Watts-NS-UW-N1000-p0.02-mean0.5-std0.05-Run1"
localization_files = f"{localization_main}/phasePoints"
val_p = 0.02
val_Deff = 0.5
val_D = 5.0

phase_matrix = compile_phase_data_vol2(localization=localization_files, half_max_nrad=250)
Nrow = len(phase_matrix)
Ncol = len(phase_matrix[0, :])
print("Nrow:", Nrow, "Ncol:", Ncol)

# --- Make sure how many rows, cluster didn't calculate
for row in range(len(phase_matrix)):
    phase_row = phase_matrix[row, :]

    # Check if all elements in the row are NaN
    if np.all(np.isnan(phase_row)):
        print(f"In row: {row}, cluster did not complete calculations.")

# --- Creating the grid, which stars
full_k_range = np.arange(1, Ncol + 1, 1)
full_nrad_range = np.arange(1, Nrow + 1, 1)
# Initialize and fill the full grid
full_grid = np.full((len(full_nrad_range), len(full_k_range)), 3.0)  # Initialize with -1

# Map the existing data to the new full grid
k_index_map = {k: i for i, k in enumerate(full_k_range)}
nrad_index_map = {n: i for i, n in enumerate(full_nrad_range)}

for i, nrad in enumerate(full_nrad_range):
    for j, k in enumerate(full_k_range):
        full_grid[nrad_index_map[nrad], k_index_map[k]] = phase_matrix[i][j]

# --------------------------------------- IMSHOW GRID: HOT ----------------------------------------------------------- #
# Define the colormap and color boundaries
cmap = plt.get_cmap('hot')
bounds = [0, 1, 2, 3.0, 4.0]

# Plotting using imshow
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)
cax = ax.imshow(phase_matrix, interpolation='nearest', cmap=cmap, aspect='auto', origin='lower')
cax.set_clim(vmin=0, vmax=4.0)
cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3.0, 4.0], orientation='vertical')

# Adding custom text annotations within the colorbar
cbar.ax.text(3, 0.1, 'Wall', transform=cbar.ax.transAxes, va='bottom', ha='right', fontsize=12, color='black')
cbar.ax.text(4, 0.357, 'Division', transform=cbar.ax.transAxes, va='center', ha='right', fontsize=12, color='black')
cbar.ax.text(5, 0.76, 'Domination', transform=cbar.ax.transAxes, va='top', ha='right', fontsize=12, color='black')
cbar.ax.text(5.8, 1.0, 'No recognition', transform=cbar.ax.transAxes, va='top', ha='right', fontsize=12, color='black')

# label
plt.xlabel('Connectivity (k / 2)')
plt.ylabel('Number of Radicals (Nrad / 2)')
plt.title('Phase Transition Behavior Grid: p = 0.02')

# Adjust ticks on the x and y-axis to display every 10 units
ax.set_xticks([i for i, k in enumerate(full_k_range) if k % 10 == 0])
ax.set_xticklabels([k for k in full_k_range if k % 10 == 0])

ax.set_yticks([i for i, n in enumerate(full_nrad_range) if n % 10 == 0])
ax.set_yticklabels([n for n in full_nrad_range if n % 10 == 0])

if savePlot:
    plt.savefig(f"{localization_main}/phase_space_grid_hot.png")
    plt.close()
else:
    plt.show()

# --------------------------------------- IMSHOW GRID: SEGMENTED ----------------------------------------------------- #
# Define a custom segmented colormap
from matplotlib.colors import ListedColormap, BoundaryNorm
colors = ['yellow', 'green', 'blue', 'red', 'black']  # Adjust colors as needed
cmap = ListedColormap(colors)
norm = BoundaryNorm(bounds, cmap.N)

# Plotting using imshow
fig, ax = plt.subplots(figsize=(12, 10))
cax = ax.imshow(phase_matrix, interpolation='nearest', cmap=cmap, norm=norm, aspect='auto', origin='lower')
cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3.0, 4.0], orientation='vertical')

# Adding custom text annotations within the colorbar
cbar.ax.text(3, 0.1, 'Wall', transform=cbar.ax.transAxes, va='bottom', ha='right', fontsize=12, color='black')
cbar.ax.text(4, 0.35, 'Division', transform=cbar.ax.transAxes, va='center', ha='right', fontsize=12, color='black')
cbar.ax.text(5, 0.65, 'Full Division', transform=cbar.ax.transAxes, va='top', ha='right', fontsize=12, color='black')
cbar.ax.text(4.7, 0.90, 'Domination', transform=cbar.ax.transAxes, va='top', ha='right', fontsize=12, color='black')

# Labels and Title
plt.xlabel(r'Connectivity $\left( \frac{k}{2} \right)$', fontsize=14)
plt.ylabel(r'Number of Radicals $\left(\frac{N_{rad}}{2} \right)$', fontsize=14)
plt.title(f'Phase Space Grid: p={val_p}, Deff={val_Deff}, D={val_D}', fontsize=18)

# Adjust ticks on the x and y-axis to display every 10 units
ax.set_xticks([i for i, k in enumerate(full_k_range) if k % 10 == 0])
ax.set_xticklabels([k for k in full_k_range if k % 10 == 0])

ax.set_yticks([i for i, n in enumerate(full_nrad_range) if n % 10 == 0])
ax.set_yticklabels([n for n in full_nrad_range if n % 10 == 0])

# Grid lines for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')

if savePlot:
    plt.savefig(f"{localization_main}/phase_space_grid_segmented.png")
    plt.close()
else:
    plt.show()
