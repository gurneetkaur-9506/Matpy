"""Top-level translation pipeline: Reader -> Rulebook -> Assistant -> Checker."""

import inspect
import os
import tempfile

from tree_sitter import Language, Parser
from tree_sitter_matlab import language

from assistant import draft_unresolved_functions
from checker import verify
from reader import build_structure, load_matlab_file
from rulebook import UNRESOLVED, translate_with_rulebook
from specialist_lib import __all__ as SPECIALIST_NAMES

import specialist_lib

_parser = Parser(Language(language()))


def _specialist_lib_contents():
    return {
        name: inspect.getsource(getattr(specialist_lib, name))
        for name in SPECIALIST_NAMES
    }


def _parse(path):
    tree = _parser.parse(load_matlab_file(path).encode("utf-8"))
    return build_structure(tree)


def _emit_block(statements, lines, indent=""):
    dropped = []
    for stmt in statements:
        if stmt["kind"] == "command" and not stmt["python"]:
            dropped.append(stmt["source"])
            continue
        if dropped:
            lines.append(
                indent
                + "# MATLAB: %s -> Python: re-initialize state (no-op here)"
                % "; ".join(dropped)
            )
            dropped = []
        if stmt["python"] == UNRESOLVED:
            lines.append(indent + "# UNRESOLVED: %s" % stmt["source"])
            continue
        lines.append(indent + stmt["comment"])
        lines.append(indent + stmt["python"])
    if dropped:
        lines.append(
            indent
            + "# MATLAB: %s -> Python: re-initialize state (no-op here)"
            % "; ".join(dropped)
        )


def _emit_function(func, lines):
    lines.append("")
    lines.append("def %s(*args, **kwargs):" % func["name"])
    _emit_block(func["statements"], lines, indent="    ")
    draft = func.get("draft")
    if draft:
        notes = "; ".join(draft["notes"]) if draft["notes"] else "none"
        lines.append(
            "    # Assistant draft: confidence=%.2f notes=%s"
            % (draft["confidence"], notes)
        )
        if draft["code"]:
            for line in draft["code"].splitlines():
                lines.append("    " + line)


def code_for_result(result):
    lines = ["import numpy as np", ""]
    _emit_block(result["statements"], lines)
    for func in result["functions"]:
        _emit_function(func, lines)
    return "\n".join(lines) + "\n"


def translate_file(matlab_path, inputs=None, tolerance=1e-8):
    result = {
        "file": matlab_path,
        "status": "ok",
        "python": "",
        "functions": [],
        "sections": {},
    }

    try:
        structure = _parse(matlab_path)
    except Exception as exc:
        result["status"] = "error"
        result["sections"]["reader"] = {"status": "error", "detail": str(exc)}
        return result
    result["sections"]["reader"] = {
        "status": "ok",
        "functions": [f.name for f in structure.functions],
        "statements": len(structure.statements),
    }

    rulebook_result = translate_with_rulebook(structure)
    all_statements = rulebook_result["statements"] + [
        s for fn in rulebook_result["functions"] for s in fn["statements"]
    ]
    unresolved_count = sum(1 for s in all_statements if s["python"] == UNRESOLVED)
    result["sections"]["rulebook"] = {
        "status": "unresolved" if unresolved_count else "ok",
        "unresolved": unresolved_count,
        "total": len(all_statements),
    }
    if unresolved_count:
        result["status"] = "unresolved"

    draft_unresolved_functions(rulebook_result, _specialist_lib_contents())
    drafted = [f["name"] for f in rulebook_result["functions"] if "draft" in f]
    result["functions"] = rulebook_result["functions"]
    result["sections"]["assistant"] = {
        "status": "drafted" if drafted else "none",
        "drafted": drafted,
    }

    result["python"] = code_for_result(rulebook_result)

    if not inputs:
        result["sections"]["checker"] = {
            "status": "skipped",
            "detail": "no inputs provided for numeric cross-check",
        }
        return result

    with tempfile.TemporaryDirectory() as tmp:
        py_path = os.path.join(
            tmp, os.path.basename(matlab_path).rsplit(".", 1)[0] + ".py"
        )
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
