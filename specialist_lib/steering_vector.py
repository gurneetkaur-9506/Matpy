import numpy as np


def steering_vector(theta, n_elements, spacing):
    """Compute the steering vector of a uniform linear array.

    Replaces the MATLAB Phased Array Toolbox function steervec
    (equivalently, the phased.SteeringVector system object) for a
    uniform linear array with element spacing `spacing` (in wavelengths)
    and `n_elements` elements, steered toward direction `theta`.

    Args:
        theta (float or array_like): Steering angle(s) in radians.
        n_elements (int): Number of array elements.
        spacing (float): Element spacing in wavelengths.

    Returns:
        array_like: Complex steering vector with shape
        (n_elements, len(theta)); for scalar theta the result has shape
        (n_elements, 1).
    """
    theta = np.asarray(theta, dtype=float)
    scalar_input = theta.ndim == 0
    theta = np.atleast_1d(theta)

    element_indices = np.arange(n_elements)
    phase = 2 * np.pi * spacing * np.sin(theta)

    sv = np.exp(1j * np.outer(element_indices, phase))

    if scalar_input:
        return sv[:, 0]
    return sv
