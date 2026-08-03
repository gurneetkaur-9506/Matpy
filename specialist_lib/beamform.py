import numpy as np


def beamform(signal, weights):
    """Apply beamforming weights to an array signal.

    Replaces the MATLAB Phased Array Toolbox function phased.Beamformer
    (equivalently, the conventional beamformer apply() call) which
    combines the signals received by an array into a single beamformed
    output using the supplied complex weights.

    The output is the weighted sum across elements at each time step:
    y[t] = sum_e conj(weights[e]) * signal[e, t].

    Args:
        signal (array_like): Array signal, shape (n_elements, n_samples).
        weights (array_like): Complex beamforming weights, length
            n_elements.

    Returns:
        array_like: Beamformed output with one sample per time step.
    """
    signal = np.asarray(signal)
    weights = np.asarray(weights)

    if signal.ndim != 2 or signal.shape[0] != weights.shape[0]:
        raise ValueError(
            "signal must have shape (n_elements, n_samples) and weights "
            "must have length n_elements"
        )

    return np.conj(weights) @ signal
