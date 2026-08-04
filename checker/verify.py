from .compare_outputs import compare_outputs
from .run_matlab_mock import run_matlab_mock
from .run_python import run_python


def verify(matlab_file, python_file, inputs=None, tolerance=1e-8):
    matlab_result = run_matlab_mock(matlab_file, inputs)
    output_names = list(matlab_result.get("outputs", {}).keys()) or None
    python_result = run_python(python_file, inputs, output_names=output_names)
    return compare_outputs(matlab_result, python_result, tolerance)
