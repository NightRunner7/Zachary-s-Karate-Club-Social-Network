import numpy as np

# Define the vector c and the matrix w
c = np.array([1, 2, 3])
w = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Compute the term sum_j c_j * w_ij for all i (matrix multiplication)
term1 = np.dot(w, c)

# Compute the term c_i * sum_j w_ij for all i
sum_w = np.sum(w, axis=1)  # sum_j w_ij for each i
term2 = c * sum_w  # element-wise multiplication

# Compute A
A = term1 - term2

print("A:", A)
