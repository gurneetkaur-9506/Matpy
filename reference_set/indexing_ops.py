import numpy as np

# MATLAB: % comment -> Python: # comment (MATLAB comment replaced by Python comment)

# MATLAB: clear; close all; clc; -> Python: re-initialize state (no-op here)

# MATLAB: A = [1 2 3; 4 5 6]; -> Python: numpy array, ';' row separator -> list rows
A = np.array([[1, 2, 3], [4, 5, 6]])

# MATLAB: B = [7 8; 9 10; 11 12]; -> Python: numpy array with rows from ';'
B = np.array([[7, 8], [9, 10], [11, 12]])

# MATLAB: disp('A(1,1) (first row, first col):'); -> Python: print string
print('A(1,1) (first row, first col):')

# MATLAB: disp(A(1,1)); -> Python: 1-based index converted to 0-based (row 0, col 0)
print(A[0, 0])

# MATLAB: disp('A(2,3) (second row, third col):'); -> Python: print string
print('A(2,3) (second row, third col):')

# MATLAB: disp(A(2,3)); -> Python: 1-based index converted to 0-based (row 1, col 2)
print(A[1, 2])

# MATLAB: disp('Matrix multiplication A * B:'); -> Python: print string
print('Matrix multiplication A * B:')

# MATLAB: disp(A * B); -> Python: '@' for matrix multiplication
print(A @ B)

# MATLAB: disp('Element-wise multiplication A .* A:'); -> Python: print string
print('Element-wise multiplication A .* A:')

# MATLAB: disp(A .* A); -> Python: '*' for element-wise (broadcast) multiplication
print(A * A)

# MATLAB: disp('First row of A via 1-based indexing:'); -> Python: print string
print('First row of A via 1-based indexing:')

# MATLAB: disp(A(1, :)); -> Python: 1-based index converted to 0-based, ':' slice preserved
print(A[0, :])
