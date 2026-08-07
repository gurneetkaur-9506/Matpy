"""General 1-D convolution with MATLAB shape modes.

Replaces the MATLAB Signal Processing Toolbox conv function
(``w = conv(u, v, shape)``), mapping its 'full', 'same' and 'valid'
shape modes onto ``numpy.convolve``.
"""

import numpy as np

_VALID_SHAPES = ("full", "same", "valid")


def _as_vector(a, name):
    arr = np.asarray(a)
    if arr.ndim > 1:
        raise ValueError(
            "conv requires one-dimensional inputs; got %d-D %s"
            % (arr.ndim, name)
        )
    return np.atleast_1d(arr)


def conv(u, v, shape="full"):
    """Discrete 1-D convolution of ``u`` and ``v``.

    Replaces the MATLAB Signal Processing Toolbox conv function. The
    ``shape`` argument selects which portion of the full convolution to
    return, matching MATLAB's behavior:

    - "full": full convolution, length ``len(u) + len(v) - 1`` (default)
    - "same": central part with the same length as ``u``, aligned so the
      convolution's center is at index ``(len(v) - 1) // 2``
    - "valid": only where ``u`` and ``v`` fully overlap, length
      ``len(u) - len(v) + 1`` when ``len(u) >= len(v)``

    Args:
        u (array_like): First input, a 1-D sequence.
        v (array_like): Second input, a 1-D sequence.
        shape (str, optional): Output region: "full", "same" or "valid".
            Defaults to "full".

    Returns:
        array_like: Convolution of ``u`` and ``v`` with the requested
        shape.
    """
    if shape not in _VALID_SHAPES:
        raise ValueError(
            "shape must be one of %s" % ", ".join(repr(s) for s in _VALID_SHAPES)
        )
    u = _as_vector(u, "u")
    v = _as_vector(v, "v")
    return np.convolve(u, v, mode=shape)
