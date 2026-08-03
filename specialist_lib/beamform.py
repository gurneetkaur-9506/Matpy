def beamform(signal, weights):
    """Apply beamforming weights to an array signal.

    Replaces the MATLAB Phased Array Toolbox function phased.Beamformer
    (equivalently, the conventional beamformer apply() call) which
    combines the signals received by an array into a single beamformed
    output using the supplied complex weights.

    Args:
        signal (array_like): Array signal, shape (n_elements, n_samples).
        weights (array_like): Complex beamforming weights, length
            n_elements.

    Returns:
        array_like: Beamformed output with one sample per time step.

    Not yet implemented.
    """
    pass
