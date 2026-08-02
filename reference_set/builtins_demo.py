import numpy as np

# MATLAB: % comment -> Python: # comment (MATLAB comment replaced by Python comment)

# MATLAB: clear; close all; clc; -> Python: re-initialize state (no-op here)

# MATLAB: x = linspace(0, 2*pi, 10); -> Python: numpy linspace, same signature
x = np.linspace(0, 2 * np.pi, 10)

# MATLAB: m = zeros(2, 5); -> Python: numpy zeros with dimensions as a tuple
m = np.zeros((2, 5))

# MATLAB: m(1, :) = sin(x(1:5)); -> Python: 1-based row -> 0-based row 0,
#      ':' slice kept, 1:5 -> 0:5 (end exclusive), numpy element-wise sin
m[0, :] = np.sin(x[0:5])

# MATLAB: m(2, :) = sin(x(6:10)); -> Python: 1-based row -> 0-based row 1,
#      6:10 -> 5:10 (end exclusive)
m[1, :] = np.sin(x[5:10])

# MATLAB: disp('Size of m:'); -> Python: print string
print('Size of m:')

# MATLAB: disp(size(m)); -> Python: numpy shape
print(m.shape)

# MATLAB: r = reshape(m, 5, 2); -> Python: numpy reshape, same arguments
r = np.reshape(m, (5, 2))

# MATLAB: disp('Reshaped m (5x2):'); -> Python: print string
print('Reshaped m (5x2):')

# MATLAB: disp(r); -> Python: print array
print(r)

# MATLAB: disp('Size of x:'); -> Python: print string
print('Size of x:')

# MATLAB: disp(size(x)); -> Python: numpy shape
print(x.shape)
