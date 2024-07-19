import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# --------------------------------------- CASE P = 0.02 -------------------------------------------------------------- #
# --- RESULTS OF OUR SIMULATIONS
# Given data with scaling down by half
k_values = np.array([6, 10, 20, 40, 50, 80, 100, 160, 200, 300]) // 2  # columns
nrad_values = np.array([6, 20, 40, 50, 80, 100, 160, 200, 300, 500]) // 2  # rows

# This assumes the data matrix you provided, restructured for clarity
data_matrix = [
    [0.50, 2.00, 2.25, 2.25, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50],
    [0.50, 1.00, 1.50, 2.00, 2.25, 2.50, 2.50, 2.50, 2.50, 2.50],
    [0.75, 1.25, 1.50, 1.75, 2.00, 2.00, 2.25, 2.50, 2.50, 2.50],
    [1.00, 1.25, 1.50, 2.00, 2.00, 2.00, 2.25, 2.50, 2.50, 2.50],
    [1.25, 1.50, 1.50, 1.50, 1.50, 2.00, 2.00, 2.50, 2.25, 2.50],
    [1.50, 1.50, 1.50, 1.50, 1.50, 2.00, 2.00, 2.25, 2.00, 2.50],
    [1.50, 1.50, 1.50, 1.50, 1.75, 1.75, 1.50, 2.00, 2.00, 2.25],
    [1.50, 1.50, 1.50, 1.25, 1.50, 1.50, 1.50, 2.00, 2.00, 2.25],
    [1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 2.00, 2.00],
    [1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50]
]

# ------------------------- SIMPLEST GRID
# Create DataFrame
df = pd.DataFrame(data_matrix, index=nrad_values, columns=k_values)

# Create a custom colormap
cmap = mcolors.ListedColormap(['#ff9999', '#ffff99', '#99ff99'])
bounds = [0, 1, 2, 2.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plotting
plt.figure(figsize=(10, 8))
ax = sns.heatmap(df, annot=True, cmap=cmap, norm=norm, cbar_kws={"ticks": bounds})
plt.title('Phase Transition Behavior Grid: p = 0.02')
plt.xlabel('Connectivity (k / 2)')
plt.ylabel('Number of Radicals (Nrad / 2)')
plt.gca().invert_yaxis()  # Invert y-axis to have increasing Nrad values go up

# Retrieve the color bar
cbar = ax.collections[0].colorbar

# Adding custom text annotations within the colorbar
cbar.ax.text(3, 0.17, 'Wall', transform=cbar.ax.transAxes, va='bottom', ha='right', fontsize=12, color='black')
cbar.ax.text(4, 0.5, 'Division', transform=cbar.ax.transAxes, va='center', ha='right', fontsize=12, color='black')
cbar.ax.text(5, 0.85, 'Domination', transform=cbar.ax.transAxes, va='top', ha='right', fontsize=12, color='black')

plt.show()

# ------------------------- WHOLE GRID
# Creating the full grid with float dtype
full_k_range = np.arange(min(k_values), max(k_values)+1, 1)
full_nrad_range = np.arange(min(nrad_values), max(nrad_values)+1, 1)
full_grid = pd.DataFrame(index=full_nrad_range, columns=full_k_range, dtype=float)
full_grid[:] = -1.0  # Initialize with NaN (for float compatibility)

# Filling the grid with existing data
for i, nrad in enumerate(nrad_values):
    for j, k in enumerate(k_values):
        if k in full_grid.columns and nrad in full_grid.index:
            full_grid.at[nrad, k] = data_matrix[i][j]

# Define the colormap
cmap = mcolors.ListedColormap(['white', '#ff9999', '#ffff99', '#99ff99'])
bounds = [-1.0, 0.0, 1.0, 2.0, 2.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plotting
plt.figure(figsize=(12, 10))
ax = sns.heatmap(full_grid, annot=False, cmap=cmap, norm=norm, cbar_kws={"ticks": [-1, 0, 1, 2, 2.5]})
plt.title('Phase Transition Behavior Grid: p = 0.02')
plt.xlabel('Connectivity (k / 2)')
plt.ylabel('Number of Radicals (Nrad / 2)')
plt.gca().invert_yaxis()  # Invert y-axis to have increasing Nrad values go up

# Retrieve the color bar
cbar = ax.collections[0].colorbar

# Adding custom text annotations within the colorbar
cbar.ax.text(4, 0.1, 'No data', transform=cbar.ax.transAxes, va='top', ha='right', fontsize=12, color='black')
cbar.ax.text(3, 0.35, 'Wall', transform=cbar.ax.transAxes, va='bottom', ha='right', fontsize=12, color='black')
cbar.ax.text(4, 0.6, 'Division', transform=cbar.ax.transAxes, va='center', ha='right', fontsize=12, color='black')
cbar.ax.text(5, 0.9, 'Domination', transform=cbar.ax.transAxes, va='top', ha='right', fontsize=12, color='black')

plt.show()

# ------------------------- IMSHOW GRID
# Initialize and fill the full grid
full_grid = np.full((len(full_nrad_range), len(full_k_range)), 3.0)  # Initialize with -1

# Map the existing data to the new full grid
k_index_map = {k: i for i, k in enumerate(full_k_range)}
nrad_index_map = {n: i for i, n in enumerate(full_nrad_range)}

for i, nrad in enumerate(nrad_values):
    for j, k in enumerate(k_values):
        full_grid[nrad_index_map[nrad], k_index_map[k]] = data_matrix[i][j]

# Define the colormap and color boundaries
cmap = plt.get_cmap('hot')
bounds = [0, 1, 2, 2.5, 3.0]

# Plotting using imshow
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)
cax = ax.imshow(full_grid, interpolation='nearest', cmap=cmap, aspect='auto', origin='lower')
cax.set_clim(vmin=0, vmax=3.0)
cbar = fig.colorbar(cax, ticks=[0, 1, 2, 2.5, 3.0], orientation='vertical')

# Adding custom text annotations within the colorbar
cbar.ax.text(3, 0.1, 'Wall', transform=cbar.ax.transAxes, va='bottom', ha='right', fontsize=12, color='black')
cbar.ax.text(4, 0.5, 'Division', transform=cbar.ax.transAxes, va='center', ha='right', fontsize=12, color='black')
cbar.ax.text(5, 0.75, 'Domination', transform=cbar.ax.transAxes, va='top', ha='right', fontsize=12, color='black')
cbar.ax.text(4, 0.94, 'No data', transform=cbar.ax.transAxes, va='top', ha='right', fontsize=12, color='black')

# label
plt.xlabel('Connectivity (k / 2)')
plt.ylabel('Number of Radicals (Nrad / 2)')
plt.title('Phase Transition Behavior Grid: p = 0.02')

# Adjust ticks on the x and y-axis to display every 10 units
ax.set_xticks([i for i, k in enumerate(full_k_range) if k % 10 == 0])
ax.set_xticklabels([k for k in full_k_range if k % 10 == 0])
ax.set_yticks([i for i, n in enumerate(full_nrad_range) if n % 10 == 0])
ax.set_yticklabels([n for n in full_nrad_range if n % 10 == 0])

plt.show()
