# Zachary Evolution Model Simulation

This repository contains Python code for simulating the evolution of the Zachary Karate Club network using the Zachary Evolution Model. The model investigates how node states and edge weights evolve over time, aiming to understand dynamics in social networks and small group behaviors.

## Features
- Network Evolution Simulation: Simulate the evolution of node states and edge weights over a specified number of time steps.
- Visualization: Plot and save visual representations of the network at intervals and create GIFs to visualize its evolution.
- Node State Analysis: Plot the evolution of individual node states over time and compare simulated faction outcomes with real-world results.
- Publication-based Initialization: Initialize the network based on the publication "An Information Flow Model for Conflict and Fission in Small Groups" for comparative analysis.
  
## Getting Started
### Installation
Clone the repository:

```bash
git clone https://github.com/NightRunner7/Zachary-s-Karate-Club-Social-Network.git
cd Zachary-s-Karate-Club-Social-Network
```

### Usage
1. Modify simulation settings in basicTask.py to customize parameters such as diffusion coefficient, coupling parameter, and time step.

2. Run the simulation:

```bash
python basicTask.py
```
3. Explore the output:

  - View generated plots and GIFs in the output directories (./D-{D}-beta-{beta}-dt-{dt}-Run-{run} and ./D-{D}-beta-{beta}-dt-{dt}-Run-{run}/evolutionNodes) for visual analysis.
  - Compare simulated faction outcomes with actual club factions post-fission.

## Dependencies
- Python 3.x
- NetworkX
- NumPy
- Matplotlib
- Pillow (PIL)

## Author
Krzysztof Szafrański
