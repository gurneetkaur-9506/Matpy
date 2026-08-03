import numpy as np

from .steering_vector import steering_vector


def array_factor(theta, n_elements, spacing, theta0=0.0):
    """Compute the array factor of a uniform linear array.

    Replaces the MATLAB Phased Array Toolbox pattern-related array
    factor computation, such as the field from steervec-based steering
    vectors or the array factor obtained via phased.ULA pattern methods
    for an array with `n_elements` elements and element spacing
    `spacing` (in wavelengths), evaluated over angle `theta`, optionally
    steered toward `theta0`.

    The array factor is the coherent sum of the steering vector toward
    theta phase-aligned with the steering vector toward theta0:
    AF(theta) = sum_n exp(j*2*pi*spacing*n*(sin(theta)-sin(theta0))).

    Args:
        theta (float or array_like): Evaluation angle(s) in radians.
        n_elements (int): Number of array elements.
        spacing (float): Element spacing in wavelengths.
        theta0 (float, optional): Steering direction in radians.
            Defaults to 0 (boresight).

    Returns:
        array_like: Complex array factor with one value per angle in
        theta; for scalar theta the result is a scalar.
    """
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=float))
    scalar_input = np.asarray(theta).ndim == 0

    sv = steering_vector(theta_arr, n_elements, spacing)
    ref = steering_vector(theta0, n_elements, spacing)

    af = np.sum(sv * np.conj(ref)[:, np.newaxis], axis=0)

    if scalar_input:
        return af[0]
    return af
