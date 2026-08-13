"""Run the original MATLAB source against a live MATLAB Engine (optional).

This module is the real-MATLAB counterpart of :mod:`checker.run_matlab_mock`.
It only imports ``matlab.engine`` lazily, so the rest of the package works
unchanged when MATLAB is not installed: callers keep using the seeded mock
reference instead.

When a MATLAB Engine is available, ``run_matlab_engine`` starts one shared
engine session, calls the original ``.m`` function through it, converts the
returned MATLAB arrays back to numpy, and produces a result dict in the same
shape as ``run_matlab_mock`` so the rest of the Checker is unchanged.
"""

import contextlib

import numpy as np

from .run_matlab_mock import _parse_signature


def matlab_engine_available():
    """Return True when a ``matlab.engine`` module is importable.

    This only probes importability; the engine itself is started lazily by
    :func:`run_matlab_engine`.
    """
    try:
        import matlab.engine  # noqa: F401
    except ImportError:
        return False
    return True


_ENGINE = None


def start_matlab_engine():
    """Start (or reuse) a single shared MATLAB Engine session."""
    global _ENGINE
    if _ENGINE is None:
        import matlab.engine

        _ENGINE = matlab.engine.start_matlab()
    return _ENGINE


def close_matlab_engine():
    """Quit the shared MATLAB Engine session, if one is running."""
    global _ENGINE
    if _ENGINE is not None:
        try:
            _ENGINE.quit()
        except Exception:
            pass
        _ENGINE = None


@contextlib.contextmanager
def matlab_engine_session():
    """Context manager that owns one MATLAB Engine session.

    The engine is started on entry and always quit on exit, so the MATLAB
    process is never leaked, even when the body raises.
    """
    engine = start_matlab_engine()
    try:
        yield engine
    finally:
        close_matlab_engine()


def _to_matlab(value):
    """Convert a numpy/scalar input into a matlab-compatible argument."""
    import matlab

    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if np.iscomplexobj(arr):
        return matlab.complex(arr.real.tolist(), arr.imag.tolist())
    return matlab.double(arr.tolist())


def _to_numpy(value):
    """Convert a MATLAB Engine return value into a numpy array.

    MATLAB stores vectors as 1xN / Nx1 two-dimensional arrays, while the
    translated Python naturally produces 1-D arrays, so singleton dimensions
    are squeezed out before the values reach :func:`checker.compare_outputs`.
    """
    import matlab

    if isinstance(value, matlab.complex):
        real = np.asarray(value.real, dtype=float)
        imag = np.asarray(value.imag, dtype=float)
        return np.squeeze(real + 1j * imag)
    if isinstance(value, matlab.double):
        return np.squeeze(np.asarray(value.tolist(), dtype=float))
    if hasattr(value, "tolist"):
        return np.squeeze(np.asarray(value.tolist()))
    return np.squeeze(np.asarray(value))


def _failure(file_path, func_name, inputs, outputs, notes):
    return {
        "success": False,
        "file": file_path,
        "function": func_name,
        "inputs": {k: np.asarray(v) for k, v in inputs.items()},
        "outputs": outputs,
        "notes": notes,
        "source": "matlab",
    }


def run_matlab_engine(file_path, inputs=None):
    """Run a MATLAB function file through a live MATLAB Engine.

    Args:
        file_path: Path to the original ``.m`` source.
        inputs: Dict of argument values keyed by the MATLAB argument names.

    Returns:
        A result dict in the same shape as ``run_matlab_mock``, with the
        additional ``"source": "matlab"`` marker::

            {"success", "file", "function", "inputs", "outputs", "notes",
             "source"}
    """
    inputs = inputs or {}
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    name, outputs, args = _parse_signature(text)
    if name is None:
        return _failure(
            file_path,
            None,
            inputs,
            {},
            ["could not parse a MATLAB function signature"],
        )

    notes = []
    positional = []
    missing = [arg for arg in args if arg not in inputs]
    if missing:
        for arg in missing:
            notes.append("missing input %r" % arg)
        return _failure(file_path, name, inputs, {}, notes)
    positional = [inputs[arg] for arg in args]

    with matlab_engine_session() as engine:
        try:
            matlab_args = [_to_matlab(value) for value in positional]
        except Exception as exc:
            notes.append("failed to convert inputs for MATLAB: %s" % exc)
            return _failure(file_path, name, inputs, {}, notes)

        try:
            func = getattr(engine, name)
            raw = func(*matlab_args, nargout=len(outputs))
        except Exception as exc:
            notes.append("execution failed: %s" % exc)
            return _failure(file_path, name, inputs, {}, notes)

    if len(outputs) == 0:
        values = ()
    elif len(outputs) == 1:
        values = (raw,)
    else:
        values = raw

    result = {}
    for index, out in enumerate(outputs):
        if index < len(values):
            result[out] = _to_numpy(values[index])
    if len(values) != len(outputs):
        notes.append(
            "MATLAB returned %d outputs but %d were expected"
            % (len(values), len(outputs))
        )

    return {
        "success": True,
        "file": file_path,
        "function": name,
        "inputs": {k: np.asarray(v) for k, v in inputs.items()},
        "outputs": result,
        "notes": notes,
        "source": "matlab",
    }
