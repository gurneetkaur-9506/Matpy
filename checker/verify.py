from .compare_outputs import compare_outputs
from .run_matlab_mock import _parse_signature, run_matlab_mock
from .run_python import python_param_names, run_python
from .run_matlab_real import matlab_engine_available, run_matlab_engine


def _match_inputs(matlab_args, python_params, inputs):
    """Split one input dict into MATLAB- and Python-keyed dicts.

    The user supplies inputs under one naming convention; the translated
    Python may rename a MATLAB keyword (``lambda`` -> ``lamb``), so values are
    matched positionally so both sides can actually run.
    """
    matlab_inputs = {}
    python_inputs = {}
    for index, mname in enumerate(matlab_args):
        if mname in inputs:
            matlab_inputs[mname] = inputs[mname]
        elif index < len(python_params) and python_params[index] in inputs:
            matlab_inputs[mname] = inputs[python_params[index]]
    for index, pname in enumerate(python_params):
        if pname in inputs:
            python_inputs[pname] = inputs[pname]
        elif index < len(matlab_args) and matlab_args[index] in inputs:
            python_inputs[pname] = inputs[matlab_args[index]]
    return matlab_inputs, python_inputs


def verify(matlab_file, python_file, inputs=None, tolerance=1e-8, use_real_matlab=None):
    """Compare a translated Python module against the reference output for its
    original MATLAB source (a live MATLAB Engine when available, otherwise a
    deterministic seeded mock).

    Args:
        matlab_file: Path to the original MATLAB source.
        python_file: Path to the translated Python module.
        inputs: Dict of argument values for both sides.
        tolerance: Relative and absolute closeness tolerance.
        use_real_matlab: When True run the original MATLAB through a live
            MATLAB Engine; when False use the seeded mock reference. When
            None (default) this is decided by whether ``matlab.engine`` is
            importable.

    Returns:
        "verified", "failed", or "review needed" (see compare_outputs).
    """
    inputs = inputs or {}
    if use_real_matlab is None:
        use_real_matlab = matlab_engine_available()

    with open(matlab_file, "r", encoding="utf-8") as f:
        text = f.read()
    _name, _outputs, matlab_args = _parse_signature(text)
    python_params = python_param_names(python_file)
    matlab_inputs, python_inputs = _match_inputs(
        matlab_args, python_params, inputs
    )

    if use_real_matlab:
        matlab_result = run_matlab_engine(matlab_file, matlab_inputs)
    else:
        matlab_result = run_matlab_mock(matlab_file, matlab_inputs)

    output_names = list(matlab_result.get("outputs", {}).keys()) or None
    python_result = run_python(python_file, python_inputs, output_names=output_names)
    return compare_outputs(matlab_result, python_result, tolerance)
