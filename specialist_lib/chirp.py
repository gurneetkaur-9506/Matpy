"""General chirp (frequency sweep) signal generator.

Replaces the MATLAB Signal Processing Toolbox chirp function
(``y = chirp(t, f0, t1, f1, method, phi)``) with an implementation that
mirrors ``scipy.signal.chirp`` formula-for-formula, including its
``method`` parameter and its method-name abbreviations.
"""

import numpy as np

_LINEAR_METHODS = ("linear", "lin", "li")
_QUADRATIC_METHODS = ("quadratic", "quad", "q")
_LOGARITHMIC_METHODS = ("logarithmic", "log", "lo")
_HYPERBOLIC_METHODS = ("hyperbolic", "hyp")

_VALID_METHODS = (
    _LINEAR_METHODS + _QUADRATIC_METHODS + _LOGARITHMIC_METHODS + _HYPERBOLIC_METHODS
)


def _chirp_phase(t, f0, t1, f1, method, vertex_zero):
    """Instantaneous phase (radians, without the phi offset) of the chirp.

    Mirrors ``scipy.signal._waveforms._chirp_phase`` so the output is
    comparable with scipy across every supported method.
    """
    if method in _LINEAR_METHODS:
        beta = (f1 - f0) / t1
        return 2 * np.pi * (f0 * t + 0.5 * beta * t * t)

    if method in _QUADRATIC_METHODS:
        beta = (f1 - f0) / (t1 ** 2)
        if vertex_zero:
            return 2 * np.pi * (f0 * t + beta * t ** 3 / 3)
        return 2 * np.pi * (f1 * t + beta * ((t1 - t) ** 3 - t1 ** 3) / 3)

    if method in _LOGARITHMIC_METHODS:
        if f0 * f1 <= 0.0:
            raise ValueError(
                "For a logarithmic chirp, f0 and f1 must be nonzero and "
                "have the same sign."
            )
        if f0 == f1:
            return 2 * np.pi * f0 * t
        beta = t1 / np.log(f1 / f0)
        return 2 * np.pi * beta * f0 * (np.power(f1 / f0, t / t1) - 1.0)

    if method in _HYPERBOLIC_METHODS:
        if f0 == 0 or f1 == 0:
            raise ValueError(
                "For a hyperbolic chirp, f0 and f1 must be nonzero."
            )
        if f0 == f1:
            return 2 * np.pi * f0 * t
        sing = -f1 * t1 / (f0 - f1)
        return 2 * np.pi * (-sing * f0) * np.log(np.abs(1 - t / sing))

    raise ValueError(
        "method must be 'linear', 'quadratic', 'logarithmic', or "
        "'hyperbolic', but a value of %r was given." % method
    )


def chirp(t, f0, t1, f1, method="linear", phi=0, vertex_zero=True):
    """Evaluate a frequency-swept cosine (chirp) signal.

    Replaces the MATLAB Signal Processing Toolbox chirp function. The
    instantaneous frequency sweeps from ``f0`` at ``t=0`` to ``f1`` at
    ``t=t1`` following ``method``:

    - ``linear``: f(t) = f0 + (f1 - f0) * t / t1
    - ``quadratic``: f(t) = f0 + (f1 - f0) * (t / t1)**2 when
      ``vertex_zero`` is True, otherwise f(t) = f1 - (f1 - f0) * (t1 - t)**2
    - ``logarithmic``: f(t) = f0 * (f1 / f0) ** (t / t1)
    - ``hyperbolic``: f(t) = f0 * f1 * t1 / ((f0 - f1) * t + f1 * t1)

    Args:
        t (array_like): Times at which to evaluate the chirp.
        f0 (float): Frequency at time 0 (Hz).
        t1 (float): Time at which the sweep reaches ``f1`` (seconds).
        f1 (float): Frequency at time ``t1`` (Hz).
        method (str, optional): Sweep law: "linear", "quadratic",
            "logarithmic" or "hyperbolic" (abbreviations like "lin",
            "quad", "log", "hyp" are accepted, as in scipy). Defaults to
            "linear".
        phi (float, optional): Phase offset in degrees (scipy.signal.chirp
            convention). Defaults to 0.
        vertex_zero (bool, optional): Only used for "quadratic": place
            the parabola's vertex at t=0 (True) or at t=t1 (False).
            Defaults to True, matching scipy.signal.chirp.

    Returns:
        array_like: ``cos(phase(t) + deg2rad(phi))`` with one value per
        entry of ``t``; for scalar ``t`` the result is a scalar.
    """
    t_arr = np.asarray(t, dtype=float)
    scalar_input = t_arr.ndim == 0
    t_arr = np.atleast_1d(t_arr)

    if t1 <= 0:
        raise ValueError("t1 must be positive")

    phase = _chirp_phase(t_arr, float(f0), float(t1), float(f1), method, bool(vertex_zero))
    result = np.cos(phase + np.deg2rad(float(phi)))

    if scalar_input:
        return float(result[0])
    return result
