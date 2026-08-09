import urllib.error

from reader import MATLAB_TO_PYTHON, PYTHON_TO_MATLAB
from rulebook import UNRESOLVED

from .draft_translation import draft_translation

OLLAMA_UNAVAILABLE_MESSAGE = (
    "Assistant unavailable — Ollama not running. "
    "This is not a missing rule: the rulebook left this function UNRESOLVED "
    "and the Assistant could not draft it because the Ollama server is not "
    "reachable. Start it with `ollama serve` and re-run to get a draft."
)


def _is_ollama_unavailable(exc):
    """True when the failure is a connection error, not a model/runtime one."""
    return isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError))


def draft_unresolved_functions(
    result, specialist_lib_contents, direction=MATLAB_TO_PYTHON
):
    output_key = "matlab" if direction == PYTHON_TO_MATLAB else "python"
    for func in result["functions"]:
        if any(s.get(output_key) == UNRESOLVED for s in func["statements"]):
            try:
                func["draft"] = draft_translation(
                    func, specialist_lib_contents, direction=direction
                )
            except Exception as exc:
                if _is_ollama_unavailable(exc):
                    func["draft_error"] = OLLAMA_UNAVAILABLE_MESSAGE
                else:
                    func["draft_error"] = str(exc)
    return result
