% Python: import numpy as np -> MATLAB: no import; arrays built with [ ] brackets

% Python: A = np.array([[1, 2, 3], [4, 5, 6]]) -> MATLAB: nested lists -> rows
% separated by ';', elements by spaces
A = [1 2 3; 4 5 6];

% Python: B = np.array([[7, 8], [9, 10], [11, 12]]) -> MATLAB: ';' row separator
B = [7 8; 9 10; 11 12];

% Python: print('A(1,1) (first row, first col):') -> MATLAB: disp string
disp('A(1,1) (first row, first col):');

% Python: print(A[0, 0]) -> MATLAB: 0-based index -> 1-based (row 1, col 1)
disp(A(1, 1));

% Python: print('A(2,3) (second row, third col):') -> MATLAB: disp string
disp('A(2,3) (second row, third col):');

% Python: print(A[1, 2]) -> MATLAB: 0-based index -> 1-based (row 2, col 3)
disp(A(2, 3));

% Python: print('Matrix multiplication A * B:') -> MATLAB: disp string
disp('Matrix multiplication A * B:');

% Python: print(A @ B) -> MATLAB: '@' is matrix multiplication -> '*'
disp(A * B);

% Python: print('Element-wise multiplication A .* A:') -> MATLAB: disp string
disp('Element-wise multiplication A .* A:');

% Python: print(A * A) -> MATLAB: '*' element-wise in numpy -> '.*' in MATLAB
disp(A .* A);

% Python: print('First row of A via 1-based indexing:') -> MATLAB: disp string
disp('First row of A via 1-based indexing:');

% Python: print(A[0, :]) -> MATLAB: 0-based row 0 with ':' slice -> 1-based row 1
disp(A(1, :));
