import numpy as np
def adaptive_sampling(data, threshold):
    # Example threshold-based adaptive sampling logic
    new_points = []
    for i in range(len(data) - 1):
        if abs(data[i+1] - data[i]) > threshold:
            new_point = (data[i+1] + data[i]) / 2
            if new_point not in data:
                new_points.append(new_point)
    return new_points

# Example usage
current_data = np.array([20, 40, 80, 160])  # Example Nrad values
threshold = 10  # Define a threshold for when to sample between points
new_samples = adaptive_sampling(current_data, threshold)
print("New points to sample:", new_samples)


