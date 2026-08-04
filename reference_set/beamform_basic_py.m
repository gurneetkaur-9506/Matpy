% Python: import numpy as np -> MATLAB: no import; numpy names become built-ins
%
% NOTE: The Python file mixes a function definition with top-level driver
% code. A single MATLAB file may not mix them the same way; this uses a
% script with a local function at the end (R2016b+). Otherwise the driver
% would live in a separate script.

% Python: theta = np.linspace(0, np.pi, 91) -> MATLAB: linspace
% (same semantics: 91 points including both endpoints)
theta = linspace(0, pi, 91);

% Python: af = beamform_basic(N=8, d=0.5, lamb=1.0, theta=theta, theta0=0.0)
%      -> MATLAB: keyword arguments -> positional arguments
af = beamform_basic(8, 0.5, 1.0, theta, 0.0);

% Python: print("array factor shape:", af.shape) -> MATLAB: disp of size()
% AMBIGUITY: numpy ndarray.shape is a tuple of dims; MATLAB size() returns
% a row vector of dims. For a (M,) numpy array vs MATLAB (1,M) row the
% reported shape differs (e.g. [1, 91] vs [91]).
disp('array factor shape:');
disp(size(af));

% Python: print("first 3 values:", af[:3]) -> MATLAB: disp of a 1-based slice
disp('first 3 values:');
disp(af(1:3));

function af = beamform_basic(N, d, lamb, theta, theta0)
    % Python: k = 2 * np.pi / lamb -> MATLAB: 2*pi/lamb (pi is a built-in)
    k = 2 * pi / lamb;

    % Python: phase = k * d * (np.sin(theta) - np.sin(theta0))
    %      -> MATLAB: sin is element-wise on vectors, same as numpy
    phase = k * d * (sin(theta) - sin(theta0));

    % Python: n = np.arange(N) -> MATLAB: colon operator
    % AMBIGUITY: no direct arange() exists. Idiomatic equivalent is the
    % colon 0:N-1; alternatives include linspace(0, N-1, N). For integer
    % steps they agree, but for non-integer steps numpy arange and the
    % MATLAB colon operator can differ in endpoint handling due to
    % floating-point rounding.
    n = 0:N-1;

    % Python: af = np.exp(1j * n[:, np.newaxis] * phase).sum(axis=0)
    %      -> MATLAB: several conversions below, each AMBIGUOUS
    %
    % AMBIGUITY 1: n[:, np.newaxis] reshapes n to a column vector. There
    %   is no single MATLAB equivalent for numpy broadcasting. The outer
    %   product n[:, np.newaxis] * phase (shape (N, M)) can be written as:
    %     (a) implicit expansion: n.' .* phase            (R2016b+)
    %     (b) bsxfun(@times, n.', phase)                   (pre-R2016b)
    %     (c) outer-product matrix multiply: n.' * phase   (used here)
    %   All three produce the same (N, M) phase matrix; choosing one is a
    %   judgment call that depends on the MATLAB release and style.
    %
    % AMBIGUITY 2: 1j -> MATLAB accepts both 1i and 1j as the imaginary
    %   unit; 1i is the idiomatic form.
    %
    % AMBIGUITY 3: .sum(axis=0) sums each column (reduce the N axis).
    %   MATLAB sum(A, 1) is the explicit form; plain sum(A) picks the
    %   first non-singleton dimension, which for an (N, M) matrix (N > 1)
    %   also means dimension 1, so both are valid here.
    af = sum(exp(1i * (n.' * phase)), 1);
end
