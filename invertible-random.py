import numpy as np

# Create a well-conditioned, invertible random matrix by replacing the singular values
m = 1000
n = 500

A = np.random.rand(m,n)
print(f"Condition# A: {np.linalg.cond(A)}")
print(f"Determinant A'@A: {np.linalg.det(A.T@A)}")

# SVD
U, _, Vt = np.linalg.svd(A, full_matrices=False)

# Select a condition number to keep the determinant under 1000
# The condition number should equal det^(2/min(m,n)), see documentation for details
# Resulting condition number will uniformly be between 1 - 1000^(2/min(m,n))
s = np.logspace(0, np.log10(np.random.uniform(1, 1000**(2/min(m,n)))), min(m,n))
B = np.dot(U * s, Vt)
print(f"Condition# B: {np.linalg.cond(B)}")
print(f"Determinant B'@B: {np.linalg.det(B.T@B)}")
