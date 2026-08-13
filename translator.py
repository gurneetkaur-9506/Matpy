"""Top-level translation pipeline: Reader -> Rulebook -> Checker."""

import os
import tempfile

from checker import verify
from checker.run_matlab_real import matlab_engine_available as _engine_probe
from reader import (
    MATLAB_TO_PYTHON,
    PYTHON_TO_MATLAB,
    load_structure_from_source,
)
from rulebook import UNRESOLVED, translate_with_rulebook, translate_with_rulebook_reverse


def _parse(source, direction):
    return load_structure_from_source(source, direction)


def matlab_engine_available():
    """Return True when a live MATLAB Engine session can be started.

    The probe imports ``matlab.engine`` lazily and only reports True on
    machines where the MATLAB Engine for Python is installed.  When True the
    Checker runs the original MATLAB source through a live engine and a
    "failed" comparison is a genuine mismatch.  When False the Checker falls
    back to the seeded mock reference and any "failed" comparison is
    downgraded to the inconclusive "inconclusive_no_matlab" verdict.
    """
    return _engine_probe()


def _emit_block(statements, lines, indent="", problems=None):
    for stmt in statements:
        if stmt["kind"] == "command" and not stmt["python"]:
            continue
        if stmt["python"] == UNRESOLVED:
            if problems is not None:
                problems.append(len(lines))
            _emit_unresolved_comment(stmt.get("source"), lines, indent, "#")
            continue
        for comment in stmt.get("renamed") or ():
            lines.append(indent + comment)
        if "\n" in stmt["python"]:
            for sub in stmt["python"].split("\n"):
                lines.append(indent + sub)
        else:
            lines.append(indent + stmt["python"])
        if stmt["kind"] == "loop":
            _emit_block(
                stmt.get("body", []), lines, indent=indent + "    ", problems=problems
            )


def _emit_unresolved_comment(source, lines, indent, marker):
    """Emit a whole block construct as a single UNRESOLVED comment.

    Every line of the raw source is prefixed so no MATLAB syntax (a stray
    'end', an 'else' clause, half-converted statements) can leak into the
    output: the block is one atomic, fully-commented unit."""
    src_lines = (source or "").splitlines() or [""]
    lines.append(indent + marker + " UNRESOLVED: " + src_lines[0])
    for sub in src_lines[1:]:
        lines.append(indent + marker + " " + sub)


def _emit_function(func, lines, problems=None):
    lines.append("")
    parameters = func.get("parameters") or ()
    signature = ", ".join(parameters) if parameters else "*args, **kwargs"
    lines.append("def %s(%s):" % (func["name"], signature))
    _emit_block(func["statements"], lines, indent="    ", problems=problems)
    outputs = func.get("outputs") or ()
    if outputs:
        if len(outputs) == 1:
            lines.append("    return %s" % outputs[0])
        else:
            lines.append("    return (%s)" % ", ".join(outputs))


def code_for_result(result, problems=None):
    lines = ["import numpy as np", ""]
    _emit_block(result["statements"], lines, problems=problems)
    for func in result["functions"]:
        _emit_function(func, lines, problems=problems)
    return "\n".join(lines) + "\n"


def _emit_block_reverse(statements, lines, indent="", problems=None):
    for stmt in statements:
        matlab = stmt.get("matlab")
        if matlab == UNRESOLVED:
            if problems is not None:
                problems.append(len(lines))
            _emit_unresolved_comment(stmt.get("source"), lines, indent, "%")
            continue
        if not matlab:
            continue
        if "\n" in matlab:
            for sub in matlab.split("\n"):
                lines.append(indent + sub + ";")
        else:
            lines.append(indent + matlab + ";")


def _emit_function_reverse(func, lines, problems=None):
    lines.append("")
    parameters = func.get("parameters") or ()
    if parameters:
        lines.append("function %s(%s)" % (func["name"], ", ".join(parameters)))
    else:
        lines.append("%% function %s(*args): signature unresolved" % func["name"])
    _emit_block_reverse(func["statements"], lines, indent="    ", problems=problems)


def code_for_result_reverse(result, problems=None):
    lines = []
    _emit_block_reverse(result["statements"], lines, problems=problems)
    for func in result["functions"]:
        _emit_function_reverse(func, lines, problems=problems)
    return "\n".join(lines) + "\n"


def translate_file(
    matlab_path, inputs=None, tolerance=1e-8, direction=MATLAB_TO_PYTHON
):
    """Translate a MATLAB/Python source file on disk.

    Reads the file and delegates to :func:`translate_source`.  The result's
    ``file`` is the original path and ``source`` holds the file contents.
    """
    try:
        with open(matlab_path, "r", encoding="utf-8") as f:
            source = f.read()
    except OSError as exc:
        result = {
            "file": matlab_path,
            "direction": direction,
            "status": "error",
            "python": "",
            "functions": [],
            "sections": {},
        }
        result["sections"]["reader"] = {"status": "error", "detail": str(exc)}
        return result
    return translate_source(
        source,
        inputs=inputs,
        tolerance=tolerance,
        direction=direction,
        name=matlab_path,
    )


def translate_source(
    source, inputs=None, tolerance=1e-8, direction=MATLAB_TO_PYTHON, name=None
):
    """Translate MATLAB/Python source text directly (no file required).

    ``name`` is an optional label for the result's ``file`` field (used for
    reporting and file naming); it does not need to be a real path.

    Returns the same result shape as :func:`translate_file`.
    """
    if name:
        file_label = name
    elif direction == PYTHON_TO_MATLAB:
        file_label = "input.py"
    else:
        file_label = "input.m"

    result = {
        "file": file_label,
        "source": source,
        "direction": direction,
        "status": "ok",
        "python": "",
        "functions": [],
        "sections": {},
    }

    try:
        structure = _parse(source, direction)
    except Exception as exc:
        result["status"] = "error"
        result["sections"]["reader"] = {"status": "error", "detail": str(exc)}
        return result
    result["sections"]["reader"] = {
        "status": "ok",
        "functions": [f.name for f in structure.functions],
        "statements": len(structure.statements),
    }

    reverse = direction == PYTHON_TO_MATLAB
    rulebook_result = (
        translate_with_rulebook_reverse(structure)
        if reverse
        else translate_with_rulebook(structure)
    )
    output_key = "matlab" if reverse else "python"
    all_statements = rulebook_result["statements"] + [
        s for fn in rulebook_result["functions"] for s in fn["statements"]
    ]
    unresolved_count = sum(1 for s in all_statements if s.get(output_key) == UNRESOLVED)
    result["sections"]["rulebook"] = {
        "status": "unresolved" if unresolved_count else "ok",
        "unresolved": unresolved_count,
        "total": len(all_statements),
    }
    if unresolved_count:
        result["status"] = "unresolved"

    result["functions"] = rulebook_result["functions"]
    result["statements"] = rulebook_result["statements"]

    problems = []
    result["python"] = (
        code_for_result_reverse(rulebook_result, problems=problems)
        if reverse
        else code_for_result(rulebook_result, problems=problems)
    )
    result["problems"] = problems

    if not inputs:
        if matlab_engine_available():
            result["sections"]["checker"] = {
                "status": "skipped",
                "detail": "no inputs provided for the reference comparison",
            }
        else:
            result["sections"]["checker"] = {
                "status": "inconclusive_no_matlab",
                "detail": (
                    "no inputs provided for the reference comparison, so "
                    "no numeric verdict was reached"
                ),
            }
        return result

    with tempfile.TemporaryDirectory() as tmp:
        stem = os.path.basename(file_label).rsplit(".", 1)[0]
        if reverse:
            matlab_path = os.path.join(tmp, stem + ".m")
            py_path = os.path.join(tmp, stem + ".py")
            with open(matlab_path, "w", encoding="utf-8") as f:
                f.write(result["python"])
            with open(py_path, "w", encoding="utf-8") as f:
                f.write(source)
        else:
            matlab_path = os.path.join(tmp, stem + ".m")
            with open(matlab_path, "w", encoding="utf-8") as f:
                f.write(source)
            py_path = os.path.join(tmp, stem + ".py")
            with open(py_path, "w", encoding="utf-8") as f:
                f.write(result["python"])
        try:
            verdict = verify(matlab_path, py_path, inputs, tolerance=tolerance)
            if verdict == "failed" and not matlab_engine_available():
                result["sections"]["checker"] = {
                    "status": "inconclusive_no_matlab",
                    "detail": (
                        "the translated output disagrees with the reference, "
                        "but the reference is only a seeded mock rather than "
                        "real MATLAB output; the verdict is inconclusive"
                    ),
                }
            else:
                result["sections"]["checker"] = {"status": verdict}
        except Exception as exc:
            result["sections"]["checker"] = {
                "status": "review needed",
                "detail": str(exc),
            }

    return result
