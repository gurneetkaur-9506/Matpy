def array_factor(theta, n_elements, spacing):
    """Compute the array factor of a uniform linear array.

    Replaces the MATLAB Phased Array Toolbox pattern-related array
    factor computation, such as the field from steervec-based steering
    vectors or the array factor obtained via phased.ULA pattern methods
    for an array with `n_elements` elements and element spacing
    `spacing` (in wavelengths), evaluated over angle `theta`.

    Args:
        theta (float or array_like): Evaluation angle(s) in radians.
        n_elements (int): Number of array elements.
        spacing (float): Element spacing in wavelengths.

    Returns:
        array_like: Complex array factor with one value per angle in
        theta.

    Not yet implemented.
    """
    pass
