"""Additive white Gaussian noise (AWGN) with configurable signal power.

Replaces the MATLAB Communications Toolbox awgn function
(``y = awgn(x, snr, sigpower, mode, seed)``). Supports both measuring
the input's signal power ('measured') and supplying an explicit signal
power, in either dB or linear SNR units.
"""

import numpy as np


def awgn(x, snr, sigpower="measured", mode="dB", seed=None):
    """Add white Gaussian noise to ``x`` at a target SNR.

    Replaces the MATLAB Communications Toolbox awgn function. The noise
    power is ``signal_power / snr_linear``. For real ``x`` the noise is
    real with that variance; for complex ``x`` the real and imaginary
    parts are each zero-mean Gaussian with half that variance, so the
    total complex noise power equals the target.

    Args:
        x (array_like): Input signal.
        snr (float): Signal-to-noise ratio. In "dB" mode this is in
            decibels; in "linear" mode it is a power ratio.
        sigpower (float or str, optional): Signal power used to size the
            noise. Either the string "measured" to measure
            ``mean(|x|**2)`` from the input, or a number giving the
            signal power: in "dB" mode it is in dBW (converted as
            ``10**(sigpower/10)``), in "linear" mode it is used
            directly. Defaults to "measured".
        mode (str, optional): Units for ``snr`` (and for a numeric
            ``sigpower``): "dB" (decibel) or "linear" (power ratio).
            Defaults to "dB".
        seed (int, optional): Seed for the noise generator, for
            reproducible output. Defaults to None (nondeterministic).

    Returns:
        array_like: ``x + noise``, same shape and dtype family as ``x``.
    """
    x = np.asarray(x)
    if mode not in ("dB", "linear"):
        raise ValueError("mode must be 'dB' or 'linear'")

    if snr is None:
        raise ValueError("snr must be provided")

    snr_lin = 10.0 ** (float(snr) / 10.0) if mode == "dB" else float(snr)
    if snr_lin <= 0:
        raise ValueError("snr must be positive")

    if isinstance(sigpower, str):
        if sigpower != "measured":
            raise ValueError(
                "sigpower must be 'measured' or a numeric signal power"
            )
        signal_power = np.mean(np.abs(x) ** 2)
    else:
        signal_power = float(sigpower)
        if mode == "dB":
            signal_power = 10.0 ** (signal_power / 10.0)
        if signal_power < 0:
            raise ValueError("sigpower must be non-negative")

    noise_var = signal_power / snr_lin
    rng = np.random.default_rng(seed)

    if np.iscomplexobj(x):
        std = np.sqrt(noise_var / 2.0)
        noise = std * (
            rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
        )
    else:
        noise = np.sqrt(noise_var) * rng.standard_normal(x.shape)

    return x + noise
