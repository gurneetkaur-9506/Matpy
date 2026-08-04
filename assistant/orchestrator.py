from reader import MATLAB_TO_PYTHON, PYTHON_TO_MATLAB
from rulebook import UNRESOLVED

from .draft_translation import draft_translation


def draft_unresolved_functions(
    result, specialist_lib_contents, direction=MATLAB_TO_PYTHON
):
    output_key = "matlab" if direction == PYTHON_TO_MATLAB else "python"
    for func in result["functions"]:
        if any(s.get(output_key) == UNRESOLVED for s in func["statements"]):
            func["draft"] = draft_translation(
                func, specialist_lib_contents, direction=direction
            )
    return result
