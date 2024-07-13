import timeit
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Parameters for the Poisson distribution
lambda_poisson = 10  # Average rate (lambda)
size = 1000         # Number of samples

# Generate Poisson-distributed integers
poisson_values = np.random.poisson(lam=lambda_poisson, size=size)

# Scale to [0, 1]
scaled_poisson_values = poisson_values / np.max(poisson_values)

# Parameters for the Beta distribution
a, b = 2, 5  # Shape parameters

# Generate Beta-distributed values
beta_values = beta.rvs(a, b, size=size)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Histogram of scaled Poisson values
ax[0].hist(scaled_poisson_values, bins=40, color='blue', alpha=0.7)
ax[0].set_title('Scaled Poisson Distribution')
ax[0].set_xlabel('Values')
ax[0].set_ylabel('Frequency')

# Histogram of Beta distribution values
ax[1].hist(beta_values, bins=40, color='green', alpha=0.7)
ax[1].set_title('Beta Distribution')
ax[1].set_xlabel('Values')

plt.tight_layout()
plt.show()
