"""MATLAB Signal Processing Toolbox functions backed by scipy.signal.

Each wrapper exists because the raw scipy call does not preserve the
MATLAB argument semantics, so a thin translation layer is needed:

- ``square(t, duty)``     -- MATLAB expresses ``duty`` as a percentage
  (0..100, default 50) while ``scipy.signal.square`` expects a fraction
  in [0, 1].
- ``findpeaks(x)``        -- MATLAB returns ``[pks, locs]`` where ``locs``
  are 1-based sample indices; ``scipy.signal.find_peaks`` returns only the
  0-based indices (with properties).
- ``xcorr(x, y)``         -- MATLAB returns ``[r, lags]``; scipy only gives
  the correlation values, so the lag vector is reconstructed here.
- ``detrend(x)``          -- MATLAB's default detrend removes the constant
  (mean) component, whereas ``scipy.signal.detrend`` defaults to a linear
  fit.  The wrapper keeps the MATLAB default and its ``(x, type, bp)``
  argument order (scipy's second positional argument is ``axis``).
- ``medfilt1(x, n)``      -- MATLAB's window width may be even; scipy's
  kernel width must be odd, so the wrapper nudges even widths up by one.
- ``filter_with_state(b, a, x)`` -- MATLAB ``[y, zf] = filter(...)`` returns
  the final state with zero initial conditions; scipy only returns ``y``
  unless initial conditions are supplied, so this wrapper fills them in.
"""

import numpy as np
from scipy import signal as _signal

__all__ = [
    "detrend",
    "filter_with_state",
    "findpeaks",
    "freqz",
    "medfilt1",
    "square",
    "xcorr",
]


def square(t, duty=50):
    """Square wave with a MATLAB-style duty cycle in percent.

    ``scipy.signal.square`` takes ``duty`` in [0, 1]; MATLAB takes it in
    [0, 100] with a default of 50 (a symmetric square wave).  The wrapper
    converts percent to a fraction before delegating.
    """
    return _signal.square(np.asarray(t, dtype=float), duty=float(duty) / 100.0)


def findpeaks(x):
    """Local maxima of a signal, MATLAB-style.

    Returns ``(pks, locs)`` matching MATLAB's ``[pks, locs] = findpeaks(x)``:
    the peak values and their 1-based sample locations.
    """
    x = np.asarray(x)
    locs, _ = _signal.find_peaks(x)
    return x[locs], locs + 1


def xcorr(x, y=None, maxlag=None, scaleopt="none"):
    """Cross-correlation and lag vector, MATLAB-style.

    Returns ``(r, lags)`` matching MATLAB's ``[r, lags] = xcorr(x, y)``.
    ``maxlag`` clips the output to lags in ``[-maxlag, maxlag]`` and
    ``scaleopt`` selects one of MATLAB's normalizations: ``'none'``
    (default), ``'biased'``, ``'unbiased'`` or ``'coeff'``.
    """
    x = np.asarray(x)
    y = x if y is None else np.asarray(y)
    r = _signal.correlate(x, y, mode="full")
    lags = np.arange(-(len(y) - 1), len(x))

    if scaleopt == "biased":
        r = r / len(x)
    elif scaleopt == "unbiased":
        r = r / (len(x) - np.abs(lags))
    elif scaleopt == "coeff":
        r = r / np.sqrt(
            np.sum(np.abs(x) ** 2) * np.sum(np.abs(y) ** 2)
        )

    if maxlag is not None:
        keep = np.abs(lags) <= maxlag
        r, lags = r[keep], lags[keep]
    return r, lags


def detrend(x, type="constant", bp=0):
    """Remove a trend, keeping MATLAB's default of a constant (mean) fit.

    ``scipy.signal.detrend`` defaults to a linear fit and takes ``axis`` as
    its second positional argument; MATLAB's signature is
    ``detrend(x, type, bp)``.  This wrapper preserves the MATLAB argument
    order and default.
    """
    return _signal.detrend(
        np.asarray(x, dtype=float), axis=-1, type=type, bp=bp
    )


def medfilt1(x, n=3):
    """One-dimensional median filter, MATLAB-style.

    MATLAB's window width may be even; scipy requires an odd kernel width,
    so even widths are nudged up by one to stay faithful to the intent.
    """
    x = np.asarray(x)
    kernel = int(n)
    if kernel % 2 == 0:
        kernel += 1
    return _signal.medfilt(x, kernel_size=kernel)


def filter_with_state(b, a, x):
    """IIR filter plus final conditions, MATLAB-style.

    Emulates MATLAB's ``[y, zf] = filter(b, a, x)``: the filter starts from
    zero initial conditions (length ``max(len(a), len(b)) - 1``) and returns
    both the filtered signal and the final state ``zf``.
    """
    b = np.atleast_1d(np.asarray(b, dtype=float))
    a = np.atleast_1d(np.asarray(a, dtype=float))
    x = np.asarray(x)
    zi = np.zeros(max(len(a), len(b)) - 1)
    y, zf = _signal.lfilter(b, a, x, zi=zi)
    return y, zf


def freqz(b, a=1, n=512):
    """Frequency response, MATLAB-style return order.

    MATLAB's ``[h, w] = freqz(b, a, n)`` returns the complex response first
    and the frequencies second, while ``scipy.signal.freqz`` returns
    ``(w, h)``.  This wrapper swaps them back to MATLAB's order.
    """
    w, h = _signal.freqz(
        np.atleast_1d(np.asarray(b, dtype=float)),
        np.atleast_1d(np.asarray(a, dtype=float)),
        worN=n,
    )
    return h, w
