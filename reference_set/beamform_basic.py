import numpy as np

# MATLAB: % comment -> Python: # comment (MATLAB comment replaced by Python comment)

# MATLAB: function af = beamform_basic(N, d, lambda, theta, theta0)
#      -> Python: def returns af, same parameter list
def beamform_basic(N, d, lamb, theta, theta0):

    # MATLAB: k = 2 * pi / lambda; -> Python: 'lambda' is a Python keyword,
    #      parameter renamed to 'lamb'; scalar assignment
    k = 2 * np.pi / lamb

    # MATLAB: phase = k * d * (sin(theta) - sin(theta0));
    #      -> Python: numpy element-wise sin, scalars broadcast over theta
    phase = k * d * (np.sin(theta) - np.sin(theta0))

    # MATLAB: af = zeros(size(theta)); -> Python: numpy zeros with same shape
    af = np.zeros(theta.shape)

    # MATLAB: for n = 1:N -> Python: range starts at 0, 1-based -> 0-based
    for n in range(N):

        # MATLAB: af = af + exp(1i * (n - 1) * phase);
        #      -> Python: 1j for imaginary unit, numpy exp, element-wise add
        af = af + np.exp(1j * n * phase)

    # MATLAB: end (for loop) -> Python: dedented loop body

    # MATLAB: end (function) -> Python: return the computed array factor
    return af
