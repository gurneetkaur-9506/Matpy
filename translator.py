"""Top-level translation pipeline: Reader -> Rulebook -> Assistant -> Checker."""

import inspect
import os
import tempfile

from assistant import draft_unresolved_functions
from checker import verify
from reader import MATLAB_TO_PYTHON, PYTHON_TO_MATLAB, load_structure
from rulebook import UNRESOLVED, translate_with_rulebook, translate_with_rulebook_reverse
from specialist_lib import __all__ as SPECIALIST_NAMES

import specialist_lib


LOW_CONFIDENCE = 0.5


def _specialist_lib_contents():
    return {
        name: inspect.getsource(getattr(specialist_lib, name))
        for name in SPECIALIST_NAMES
    }


def _parse(path, direction):
    return load_structure(path, direction)


def _emit_block(statements, lines, indent="", problems=None):
    for stmt in statements:
        if stmt["kind"] == "command" and not stmt["python"]:
            continue
        if stmt["python"] == UNRESOLVED:
            if problems is not None:
                problems.append(len(lines))
            lines.append(indent + "# UNRESOLVED: %s" % stmt["source"])
            continue
        lines.append(indent + stmt["python"])
        if stmt["kind"] == "loop":
            _emit_block(
                stmt.get("body", []), lines, indent=indent + "    ", problems=problems
            )


def _emit_function(func, lines, problems=None):
    lines.append("")
    parameters = func.get("parameters") or ()
    signature = ", ".join(parameters) if parameters else "*args, **kwargs"
    lines.append("def %s(%s):" % (func["name"], signature))
    _emit_block(func["statements"], lines, indent="    ", problems=problems)
    draft = func.get("draft")
    if draft:
        notes = "; ".join(draft["notes"]) if draft["notes"] else "none"
        lines.append(
            "    # Assistant draft: confidence=%.2f notes=%s"
            % (draft["confidence"], notes)
        )
        low_confidence = problems is not None and draft["confidence"] < LOW_CONFIDENCE
        if low_confidence:
            problems.append(len(lines) - 1)
        if draft["code"]:
            for line in draft["code"].splitlines():
                lines.append("    " + line)
                if low_confidence:
                    problems.append(len(lines) - 1)
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
            lines.append(indent + "%% UNRESOLVED: %s" % stmt["source"])
            continue
        if not matlab:
            continue
        lines.append(indent + matlab + ";")


def _emit_function_reverse(func, lines, problems=None):
    lines.append("")
    parameters = func.get("parameters") or ()
    if parameters:
        lines.append("function %s(%s)" % (func["name"], ", ".join(parameters)))
    else:
        lines.append("%% function %s(*args): signature unresolved" % func["name"])
    _emit_block_reverse(func["statements"], lines, indent="    ", problems=problems)
    draft = func.get("draft")
    if draft:
        notes = "; ".join(draft["notes"]) if draft["notes"] else "none"
        lines.append(
            "    %% Assistant draft: confidence=%.2f notes=%s"
            % (draft["confidence"], notes)
        )
        low_confidence = problems is not None and draft["confidence"] < LOW_CONFIDENCE
        if low_confidence:
            problems.append(len(lines) - 1)
        if draft["code"]:
            for line in draft["code"].splitlines():
                lines.append("    " + line)
                if low_confidence:
                    problems.append(len(lines) - 1)


def code_for_result_reverse(result, problems=None):
    lines = []
    _emit_block_reverse(result["statements"], lines, problems=problems)
    for func in result["functions"]:
        _emit_function_reverse(func, lines, problems=problems)
    return "\n".join(lines) + "\n"


def translate_file(
    matlab_path, inputs=None, tolerance=1e-8, direction=MATLAB_TO_PYTHON
):
    result = {
        "file": matlab_path,
        "direction": direction,
        "status": "ok",
        "python": "",
        "functions": [],
        "sections": {},
    }

    try:
        structure = _parse(matlab_path, direction)
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

    draft_unresolved_functions(
        rulebook_result, _specialist_lib_contents(), direction=direction
    )
    drafted = [f["name"] for f in rulebook_result["functions"] if "draft" in f]
    errored = [f["name"] for f in rulebook_result["functions"] if "draft_error" in f]
    result["functions"] = rulebook_result["functions"]
    result["statements"] = rulebook_result["statements"]
    result["sections"]["assistant"] = {
        "status": "drafted" if drafted else "errored" if errored else "none",
        "drafted": drafted,
        "errors": errored,
    }

    problems = []
    result["python"] = (
        code_for_result_reverse(rulebook_result, problems=problems)
        if reverse
        else code_for_result(rulebook_result, problems=problems)
    )
    result["problems"] = problems

    if not inputs:
        result["sections"]["checker"] = {
            "status": "skipped",
            "detail": "no inputs provided for numeric cross-check",
        }
        return result

    with tempfile.TemporaryDirectory() as tmp:
        stem = os.path.basename(matlab_path).rsplit(".", 1)[0]
        if reverse:
            matlab_path = os.path.join(tmp, stem + ".m")
            py_path = os.path.join(tmp, stem + ".py")
            with open(matlab_path, "w", encoding="utf-8") as f:
                f.write(result["python"])
            with open(py_path, "w", encoding="utf-8") as f:
                with open(result["file"], "r", encoding="utf-8") as src:
                    f.write(src.read())
        else:
            py_path = os.path.join(tmp, stem + ".py")
            with open(py_path, "w", encoding="utf-8") as f:
                f.write(result["python"])
        try:
            verdict = verify(matlab_path, py_path, inputs, tolerance=tolerance)
            result["sections"]["checker"] = {"status": verdict}
        except Exception as exc:
            result["sections"]["checker"] = {
                "status": "review needed",
                "detail": str(exc),
            }

    return result
