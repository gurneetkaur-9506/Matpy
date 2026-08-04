import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])

print('A(1,1) (first row, first col):')
print(A[0, 0])

print('A(2,3) (second row, third col):')
print(A[1, 2])

print('Matrix multiplication A * B:')
print(A @ B)

print('Element-wise multiplication A .* A:')
print(A * A)

print('First row of A via 1-based indexing:')
print(A[0, :])
