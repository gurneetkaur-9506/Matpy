"""Reverse lookup stub: numpy operations -> MATLAB Phased Array Toolbox candidates.

Given a numpy operation found in a Python translation, suggest the MATLAB
Phased Array Toolbox function(s) it might correspond to. More than one
candidate is returned because a single numpy operation is typically reused
across many toolbox functions; the caller is expected to narrow the choice
using surrounding context (the specialist library's steering_vector,
array_factor and beamform modules mirror the toolbox idioms shown here).

The one-sided (Python -> MATLAB) direction is the harder case: numpy idioms
such as broadcasting (n[:, np.newaxis]), reduction methods (.sum(axis=0))
and attribute access (x.shape) have no unique MATLAB equivalent, so their
candidate lists are deliberately ambiguous. This is expected and correct.
"""

import re

# numpy operation -> candidate MATLAB Phased Array Toolbox functions.
# The candidates are derived from the operations used by the specialist
# library and the toolbox functions those specialists replace, with the
# beamform_basic idioms (np.sin, np.arange, np.exp, np.linspace,
# np.newaxis broadcasting, .sum(axis=0), .shape) populated explicitly.
_REVERSE_LOOKUP_TABLE = {
    "np.exp": ["steervec", "phased.SteeringVector", "phased.ULA"],
    "np.outer": ["steervec", "phased.SteeringVector"],
    "np.conj": [
        "steervec",
        "phased.Beamformer",
        "phased.MVDRBeamformer",
        "phased.LCMVBeamformer",
    ],
    "np.sum": ["phased.ULA", "phased.ArrayGain", "phased.Beamformer"],
    "np.sin": ["steervec", "phased.SteeringVector", "phased.ULA"],
    "np.zeros": ["phased.Beamformer", "phased.ArrayGain"],
    "np.abs": ["beamscan", "phased.BeamscanEstimator", "phased.Beamformer"],
    "np.fft.fft": ["beamscan", "phased.BeamscanEstimator"],
    "np.dot": ["phased.Beamformer", "phased.MVDRBeamformer", "phased.LCMVBeamformer"],
    "np.matmul": ["phased.Beamformer", "phased.MVDRBeamformer", "phased.LCMVBeamformer"],
    "np.linalg.inv": ["phased.MVDRBeamformer", "phased.LCMVBeamformer"],
    "np.linalg.pinv": ["phased.MVDRBeamformer", "phased.LCMVBeamformer"],
    "np.linalg.eig": ["phased.MVDRBeamformer", "phased.ESPRITEstimator"],
    "np.linalg.eigh": ["phased.MVDRBeamformer"],
    "np.linalg.svd": ["phased.MUSICEstimator", "phased.ESPRITEstimator"],
    "np.argmax": ["beamscan", "phased.BeamscanEstimator"],
    "np.max": ["beamscan", "phased.BeamscanEstimator"],
    "np.arange": ["steervec", "phased.ULA"],
    "np.linspace": ["beamscan", "phased.BeamscanEstimator", "phased.MUSICEstimator"],
    "np.newaxis": ["steervec", "phased.SteeringVector", "phased.ULA"],
    ".shape": ["size", "numel"],
}

_NUMPY_CALL_RE = re.compile(
    r"\b(?:np|numpy)\.[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
)

# numpy array methods treated as operations and normalized to their
# np.<method> function equivalents (x.sum(axis=0) <-> np.sum(x, axis=0)).
_NUMPY_METHODS = ("sum", "mean", "max", "min", "reshape", "dot", "abs", "conj")


def _normalize(name):
    if name.startswith("numpy."):
        return "np." + name[len("numpy."):]
    return name


def _collect_operations(text):
    ops = []
    seen = set()
    for match in _NUMPY_CALL_RE.finditer(text):
        name = _normalize(match.group(0))
        if name not in seen:
            seen.add(name)
            ops.append(name)
    for method in _NUMPY_METHODS:
        if re.search(r"\.%s\s*\(" % re.escape(method), text):
            name = "np." + method
            if name not in seen:
                seen.add(name)
                ops.append(name)
    if re.search(r"\.shape\b", text):
        if ".shape" not in seen:
            seen.add(".shape")
            ops.append(".shape")
    return ops


def collect_numpy_operations(text):
    """Return the distinct numpy operations found in *text*.

    Args:
        text (str): Source text such as a Python function body.

    Returns:
        list: Distinct numpy operation names in order of appearance
        (e.g. "np.exp", "np.sum", ".shape"), without candidates.
    """
    if not isinstance(text, str):
        return []
    return _collect_operations(text)


def reverse_lookup(numpy_call):
    """Return candidate MATLAB Phased Array Toolbox functions for a numpy operation.

    Args:
        numpy_call (str): A numpy operation, either a bare function name
            such as "np.conj", an array method call such as
            "np.exp(1j * n[:, np.newaxis] * phase).sum(axis=0)", or a
            full expression.

    Returns:
        list: Candidate MATLAB Phased Array Toolbox function names,
        deduplicated and ordered roughly by likelihood. Empty when no
        numpy operation is recognized or none has a known toolbox
        counterpart.
    """
    if not isinstance(numpy_call, str):
        return []
    candidates = []
    seen = set()
    for key in _collect_operations(numpy_call):
        for name in _REVERSE_LOOKUP_TABLE.get(key, []):
            if name not in seen:
                seen.add(name)
                candidates.append(name)
    return candidates
