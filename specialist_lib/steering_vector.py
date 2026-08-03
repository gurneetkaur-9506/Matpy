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
        array_like: Complex steering vector with one element per array
        element, one column per angle in theta.

    Not yet implemented.
    """
    pass
