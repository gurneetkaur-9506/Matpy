import numpy as np


def compare_outputs(matlab_result, python_result, tolerance=1e-8):
    """Compare structured MATLAB vs Python numeric outputs.

    Args:
        matlab_result: Result dict from run_matlab_mock.
        python_result: Result dict from run_python.
        tolerance: Relative and absolute closeness tolerance passed to
            numpy.allclose.

    Returns:
        "verified" if every matching output is numerically close,
        "failed" if outputs disagree beyond tolerance, or "review needed"
        when the results cannot be decided numerically (execution failures,
        misaligned output names, shape mismatches, or non-finite values).
    """
    if not matlab_result.get("success") or not python_result.get("success"):
        return "review needed"

    matlab_outputs = matlab_result.get("outputs") or {}
    python_outputs = python_result.get("outputs") or {}
    if not matlab_outputs and not python_outputs:
        return "review needed"
    if set(matlab_outputs.keys()) != set(python_outputs.keys()):
        return "review needed"

    for key in matlab_outputs:
        matlab_value = np.asarray(matlab_outputs[key])
        python_value = np.asarray(python_outputs[key])
        if matlab_value.shape != python_value.shape:
            return "review needed"
        if not np.all(np.isfinite(matlab_value)) or not np.all(np.isfinite(python_value)):
            return "review needed"
        try:
            close = np.allclose(
                matlab_value, python_value, rtol=tolerance, atol=tolerance
            )
        except (TypeError, ValueError):
            return "review needed"
        if not close:
            return "failed"
    return "verified"
